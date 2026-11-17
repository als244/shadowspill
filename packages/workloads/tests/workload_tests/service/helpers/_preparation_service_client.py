"""NumPy-only measurement/export client, also runnable on a separate laptop."""

import hashlib
import importlib.abc
import json
import runpy
import socket
import subprocess
import sys
import zipfile
from dataclasses import replace
from pathlib import Path
from tempfile import mkdtemp

import numpy as np

from shadowspill_client import (
    SSH,
    BatchConfig,
    BuildCandidates,
    Memory,
    Model,
    Objective,
    Optimizer,
    PlanningOptions,
    Preparation,
    ProfilingOptions,
    ServiceClient,
    Weights,
)
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.requests import ArtifactRef


def check_reused_training(client, settings, prepared, initial, *, local=None):
    checks = runpy.run_path(
        str(Path(__file__).with_name("_regression_service_client.py"))
    )
    training_settings = {
        **settings,
        "model": replace(settings["model"], initialize=Weights.random(seed=0)),
        "memory": Memory(device_gib=3, spill_gib=0.125),
        "prepare": replace(
            settings["prepare"],
            planning=PlanningOptions(microbatch_counts=[1, 2] if local else [1]),
            reuse=prepared.reuse,
        ),
    }
    if local:
        root, options = local
        preview = client.describe_training(**training_settings).result(30)
        (root / "config.json").write_text(json.dumps(preview.config.to_dict()))
        planned = subprocess.run(
            [
                options["planner_python"],
                "-I",
                str(Path(__file__).with_name("_submit_local_plans.py")),
                str(root),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert planned.returncode == 0, planned.stdout + planned.stderr
        proposed = json.loads((root / "proposal.json").read_text())
        uploaded = client.artifacts.upload(
            root / "local-plans.ssp",
            reference=ArtifactRef.from_dict(proposed["reference"]),
        ).result(60)
        training = client.create_training_client(
            config=preview.config, plans=uploaded
        ).result(120)
        assert training.plans.reference != uploaded.reference
    else:
        training = client.create_training_client(**training_settings).result(300)
    memory = BufferBudget(64 << 20, 8)
    checkpoint = training.save_checkpoint().result(60)
    rows = []
    for split in (False, True):
        if split:
            training.load_checkpoint(checkpoint.id).result(60)
        oracle = checks["Oracle"](initial, "float32")
        batch = checks["records"]()
        if split:
            for start, stop in ((0, 2), (2, 5)):
                result = training.forward_backward(batch[start:stop]).result(60)
                np.testing.assert_allclose(
                    result.loss,
                    oracle.forward_backward(slice(start, stop))[1],
                    rtol=2e-5,
                    atol=1e-6,
                )
            result = training.optim_step().result(60)
        else:
            result = training.step(batch).result(60)
            np.testing.assert_allclose(
                result.loss, oracle.forward_backward()[1], rtol=2e-5, atol=1e-6
            )
        oracle.update(oracle.forward_backward()[2])
        checks["check"](training, oracle, memory, counter=2 if split else 1)
        assert result.completed_updates == 1
        rows.append(result.record.to_dict())
    if local:
        checkpoint = training.save_checkpoint().result(60)
        plans = training.plans
        training.close().result(60)
        training = client.create_training_client(
            plans=plans, checkpoint=checkpoint
        ).result(120)
        checks["check"](training, oracle, memory, counter=2)
    training.close().result(60)
    assert memory.used == (0, 0)
    return {
        "reused_training": True,
        "whole_split_oracles": rows,
        "training_run": training.run_id,
        **(
            {"local_planning": proposed, "native_saved_plan_resume": True}
            if local
            else {}
        ),
    }


def main():
    class Block(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, *args):
            if fullname.split(".")[0] in {
                "torch",
                "triton",
                "shadowspill",
                "shadowspill_workloads",
                "mlops",
            }:
                raise AssertionError("framework/engine import in client: " + fullname)

    sys.meta_path.insert(0, Block())
    options = json.loads(sys.stdin.buffer.read(1 << 20))
    connection = (
        {"ssh": SSH(options["ssh_host"], options["port"])}
        if options.get("ssh_host")
        else {"base_url": f"http://127.0.0.1:{options['port']}"}
    )
    root = Path(mkdtemp(prefix="shadowspill-preparation-client-"))
    with ServiceClient(
        **connection, token=options["token"], session_id="measurements", timeout=30
    ) as client:
        settings = dict(
            model=Model("shadowspill/regression@1"),
            objective=Objective("shadowspill/weighted_squared_error@1"),
            optimizer=Optimizer(
                "mlops/adamw@1",
                precision={
                    "gradient_dtype": "float32",
                    "opt_state_dtype": "float32",
                },
            ),
            batch=BatchConfig(limits={"max_examples": 8}),
            memory=Memory(device_gib=2, spill_gib=0.0625),
            operations={
                "step": ["metrics"],
                "forward_backward": ["metrics"],
                "forward": ["predictions"],
            },
            prepare=Preparation(
                candidates=BuildCandidates(microbatch_sizes=[4, 8]),
                planning=PlanningOptions(microbatch_counts=[1, 2]),
                profiling=ProfilingOptions(
                    conditioning_seconds=0.001, measurement_seconds=0.002
                ),
            ),
        )
        prepared = client.prepare_training(**settings).result(300)
        assert prepared.info.state == "prepared"
        assert prepared.info.plans is prepared.info.initial_history_id is None
        export = prepared.export().result(30)
        path = export.download(root / "measurements.zip").result(60)
        with zipfile.ZipFile(path) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            assert manifest["configuration"]["model"]["initialize"] is None
            assert manifest["captures"] and not manifest["rejected"]
            assert all(case["required"] for case in manifest["captures"].values())
            for name, expected in manifest["files"].items():
                body = archive.read(name)
                assert expected == {
                    "size_bytes": len(body),
                    "sha256": hashlib.sha256(body).hexdigest(),
                }
        rejoined = client.get_preparation(prepared.info.run_id).result(30)
        assert rejoined.info == prepared.info
        reuse = (
            check_reused_training(
                client,
                settings,
                prepared,
                options["initial"],
                local=(root, options) if options.get("local_planning") else None,
            )
            if options.get("reuse")
            else {}
        )
    assert client._memory.used == client._work.pending.used == (0, 0)
    print(
        json.dumps(
            {
                "client_host": socket.gethostname(),
                "all_client_claims_released": True,
                "archive": str(path),
                "artifact": export.reference.to_dict(),
                "captures": len(manifest["captures"]),
                "files": len(manifest["files"]),
                "run": prepared.info.to_dict(),
                **reuse,
            }
        )
    )


if __name__ == "__main__":
    main()
