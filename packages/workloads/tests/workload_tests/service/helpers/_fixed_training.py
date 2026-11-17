"""Fresh Executor worker for fixed-window text updates and checkpoint replay."""

import json
import sys
from pathlib import Path

import torch

from engine_tests.execution._cold import forbid_preparation

forbid_preparation()

from qualification.reference.training.data.calls import Inputs  # noqa: E402
from shadowspill import Executor, Plan, load_checkpoint, save_checkpoint  # noqa: E402
from shadowspill.checkpoint import metadata_values  # noqa: E402
from shadowspill_client.schedules import WarmupCosine  # noqa: E402

root, output = map(Path, sys.argv[1:3])
options = json.loads((root / "config.json").read_text())
initial = torch.load(root / "initial.pt", weights_only=True)


def fill(request, destination):
    value = initial[request.name].view(-1)
    destination.copy_(value[request.offset : request.offset + destination.numel()])


frozen = "fixed-text-base-v1" if options["partial_checkpoint"] else None
schedule = WarmupCosine(0.01, 0.001, 1, 5)
output.mkdir()
checkpoints = output / "checkpoints"
checkpoints.mkdir()
data = Inputs(options)

with Executor(
    Plan.load(root / "training.ssp"),
    plans={"evaluation": Plan.load(root / "evaluation.ssp")},
) as executor:
    position = 0
    if len(sys.argv) > 3:
        metadata = load_checkpoint(
            executor, sys.argv[3], fill_tensor=fill, frozen_state_id=frozen
        )
        position = metadata["step"]
        data.load_state_dict(metadata_values.decode(metadata["data"]))
    else:
        executor.initialize(fill)
    while position < 5:
        rate = schedule(position)
        result = executor.run(
            **next(data), hyperparams={"optimizer_group.0.lr": rate}
        ).result()
        loss = sum(float(value) for value in result.losses)
        del result
        executor.synchronize()
        position += 1
        with (output / "metrics.jsonl").open("a") as stream:
            stream.write(
                json.dumps(
                    {
                        "step": position,
                        "train/loss": loss,
                        "hyperparameters/optimizer_group.0.lr": rate,
                    }
                )
                + "\n"
            )
        if position % 2 == 0 or position == 5:
            validation = Inputs(options, split="val")
            total = 0.0
            for _ in range(2):
                calls = next(validation)
                for args, kwargs in zip(calls["args"], calls["kwargs"], strict=True):
                    total += float(
                        executor.run(
                            plan="evaluation",
                            args=[args],
                            kwargs=[kwargs],
                        ).result()
                    )
            with (output / "evaluation.jsonl").open("a") as stream:
                stream.write(
                    json.dumps({"step": position, "eval/loss": total / 2}) + "\n"
                )
            save_checkpoint(
                executor,
                checkpoints / f"step_{position:08d}.ssp",
                frozen_state_id=frozen,
                metadata={
                    "step": position,
                    "data": metadata_values.encode(data.state_dict()),
                },
            )
            for path in sorted(checkpoints.glob("step_*.ssp"))[:-2]:
                path.unlink()
    assert data.data.next_step == position == 5
