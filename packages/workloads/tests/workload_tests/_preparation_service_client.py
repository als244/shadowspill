"""NumPy-only measurement/export client, also runnable on a separate laptop."""

import hashlib
import importlib.abc
import json
import socket
import sys
import zipfile
from pathlib import Path
from tempfile import mkdtemp

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
)


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
        prepared = client.prepare_training(
            model=Model("shadowspill/regression@1"),
            objective=Objective("shadowspill/weighted_squared_error@1"),
            optimizer=Optimizer("mlops/adamw@1"),
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
        ).result(300)
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
            }
        )
    )


if __name__ == "__main__":
    main()
