"""Framework-free GLM client: packed updates, evaluation and saved-state restart."""

import importlib.abc
import json
import math
import sys
import tarfile
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from shadowspill_client import (
    BatchConfig,
    BuildCandidates,
    Datum,
    LoRA,
    Memory,
    Model,
    Objective,
    Optimizer,
    PlanningOptions,
    Preparation,
    ProfilingOptions,
    ServiceClient,
)
from workload_tests._text_service_client import close


def state(checkpoint, directory):
    root = checkpoint.download(directory).result(60)
    with tarfile.open(root / "ranks/rank-00000.ssp") as archive:
        manifest = json.loads(archive.extractfile("manifest.json").read())
        values = {}
        for row in manifest["tensors"]:
            dtype = row["dtype"].removeprefix("torch.")
            value = np.frombuffer(
                archive.extractfile(row["member"]).read(),
                dtype=np.dtype("uint16" if dtype == "bfloat16" else dtype).newbyteorder(
                    "<" if manifest["byteorder"] == "little" else ">"
                ),
            )
            if dtype == "bfloat16":
                value = (value.astype(np.uint32) << 16).view(np.float32)
            values[row["name"]] = value.copy()
    assert "embed_tokens.weight" in manifest["omitted_parameters"]
    assert all("lora" not in name for name in manifest["omitted_parameters"])
    return values


def qualify(client, options):
    training = client.create_training_client(
        model=Model.from_pretrained(options["checkpoint"], lora=LoRA(4)),
        objective=Objective("cross_entropy"),
        optimizer=Optimizer(
            "adamw",
            precision={
                "gradient_dtype": "float32",
                "master_dtype": "float32",
                "opt_state_dtype": "float32",
            },
        ),
        batch=BatchConfig(
            limits={
                "token_capacity": 32,
                "max_sequence_length": 16,
                "max_sequences": 8,
            }
        ),
        memory=Memory(6, 0.5),
        operations={
            "step": ["metrics"],
            "forward_backward": ["metrics"],
            "forward": ["metrics"],
        },
        prepare=Preparation(
            candidates=BuildCandidates(microbatch_sizes=[32]),
            planning=PlanningOptions(microbatch_counts=[1], workers=2),
            profiling=ProfilingOptions(
                conditioning_seconds=0.001, measurement_seconds=0.002
            ),
        ),
    ).result(600)
    output = Path(options["state_directory"])
    output.mkdir()

    def snapshot(run, name):
        checkpoint = run.save_checkpoint(name=name).result(120)
        return state(checkpoint, output / name)

    plans = training.plans
    report = plans.report().result(60)
    records = [
        Datum(
            {"tokens": list(range(start, start + length))},
            {
                "target_tokens": list(range(start + 1, start + length + 1)),
                "weights": [1 / 25] * length,
            },
        )
        for start, length in ((1, 16), (33, 9))
    ]
    initial = training.save_checkpoint(name="initial").result(120)
    original = state(initial, output / "initial")
    optim = {"lr": 0.001}
    first = training.step(records, optim=optim).result(120)
    assert first.completed_updates == 1 and not first.accumulation.open
    snapshot(training, "whole")
    training.load_checkpoint(initial.id).result(120)
    parts = [
        training.forward_backward(part).result(120).loss
        for part in (records[:1], records[1:])
    ]
    split = training.optim_step(optim=optim).result(120)
    assert split.completed_updates == 1 and not split.accumulation.open
    snapshot(training, "split")
    training.load_checkpoint(initial.id).result(120)
    losses = []
    for index in range(4):
        result = training.step(records, optim=optim, runtime_trace=index == 3).result(
            120
        )
        assert result.completed_updates == index + 1 and math.isfinite(result.loss)
        losses.append(result.loss)
        if index == 2:
            resume = training.save_checkpoint(name="after-three").result(120)
    final = snapshot(training, "after-four")
    assert losses[-1] < losses[0], losses
    assert any(
        not np.array_equal(final[name], value)
        for name, value in original.items()
        if "lora" in name
    )
    evaluated = training.forward(records).result(120)
    after_evaluation = snapshot(training, "after-evaluation")
    assert after_evaluation.keys() == final.keys()
    for name, value in final.items():
        np.testing.assert_array_equal(after_evaluation[name], value, err_msg=name)
    assert len(result.traces) == 1
    runtime = result.traces[0]
    with TemporaryDirectory(prefix="shadowspill-glm-reports-") as directory:
        offline = report.download(Path(directory) / "plan.html").result(60)
        actual = runtime.download_report(Path(directory) / "runtime.html").result(60)
        assert "<html" in offline.read_text().lower()
        assert "<html" in actual.read_text().lower()
    close(client, training)
    resumed = client.create_training_client(plans=plans, checkpoint=resume).result(240)
    again = resumed.step(records, optim=optim).result(120)
    assert again.completed_updates == 4
    snapshot(resumed, "resumed-four")
    evaluation = resumed.forward(records).result(120)
    close(client, resumed)
    return {
        "losses": losses,
        "loss_comparisons": {
            "whole_split": [first.loss, sum(parts)],
            "checkpoint_resume": [losses[-1], again.loss],
            "evaluation_resume": [evaluated.loss, evaluation.loss],
        },
        "evaluation_isolation": True,
        "saved_plan_restart": True,
        "plan_report": report.reference.to_dict(),
        "runtime_report": runtime.reference.to_dict(),
    }


def main():
    class Block(importlib.abc.MetaPathFinder):
        def find_spec(self, name, *args):
            if name.split(".")[0] in {
                "torch",
                "triton",
                "mlops",
                "shadowspill",
                "shadowspill_workloads",
            }:
                raise AssertionError("framework or engine in client: " + name)

    sys.meta_path.insert(0, Block())
    options = json.load(sys.stdin)
    with ServiceClient(
        base_url=f"http://127.0.0.1:{options['port']}",
        token=options["token"],
        session_id="glm",
    ) as client:
        receipt = qualify(client, options)
    assert client._memory.used == client._work.pending.used == (0, 0)
    receipt["all_client_claims_released"] = True
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
