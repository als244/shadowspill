"""Fresh execution worker for fixed-window text training and resume checks."""

import importlib.abc
import json
import sys
from pathlib import Path

import torch


class BlockPreparation(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(
            fullname == name or fullname.startswith(name + ".")
            for name in (
                "shadowspill.build",
                "shadowspill.profiling.measure",
                "shadowspill.planning.plan",
                "shadowspill.training._legacy_trainer",
                "shadowspill.training.backends",
            )
        ):
            raise AssertionError("execution imported " + fullname)


sys.meta_path.insert(0, BlockPreparation())

from shadowspill import Plan, Trainer  # noqa: E402
from shadowspill.training.schedules import WarmupCosine  # noqa: E402
from workloads.recipes.text.data.calls import Inputs  # noqa: E402

root, output = map(Path, sys.argv[1:3])
options = json.loads((root / "config.json").read_text())
initial = torch.load(root / "initial.pt", weights_only=True)


def fill(request, destination):
    value = initial[request.name].view(-1)
    destination.copy_(value[request.offset : request.offset + destination.numel()])


with Trainer(
    Plan.load(root / "training.ssp"),
    evaluation_plan=Plan.load(root / "evaluation.ssp"),
    schedules={"optimizer_group.0.lr": WarmupCosine(0.01, 0.001, 1, 5)},
    frozen_state_id="fixed-text-base-v1" if options["partial_checkpoint"] else None,
) as trainer:
    if len(sys.argv) > 3:
        trainer.load(sys.argv[3], fill_tensor=fill)
    else:
        trainer.initialize(fill)
    output.mkdir()
    data, logged = Inputs(options), []

    def evaluate(owner, result):
        if result.step % 2 == 0 or result.step == 5:
            validation = Inputs(options, split="val")
            total = 0.0
            for _ in range(2):
                calls = next(validation)
                for args, kwargs in zip(calls["args"], calls["kwargs"], strict=True):
                    total += float(owner.evaluate(args=[args], kwargs=[kwargs]))
            with (output / "evaluation.jsonl").open("a") as stream:
                stream.write(
                    json.dumps({"step": result.step, "eval/loss": total / 2}) + "\n"
                )

    trainer.fit(
        data,
        steps=5,
        run_dir=output,
        callbacks=[evaluate],
        logger=logged.append,
        checkpoint_every=2,
        keep_last=2,
    )
    assert data.data.next_step == trainer.step_count == 5
    assert logged == [
        json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()
    ]
