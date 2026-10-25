"""Train/evaluate saved Plans; never construct a model, profile, or plan here."""

import json
import sys

from shadowspill import Plan, Trainer
from shadowspill.training.schedules import WarmupCosine
from workloads.recipes.phases import settings
from workloads.recipes.text.data.calls import Inputs

from .common import model_config
from .initialize import Initialize


def run(directory):
    root, options = settings(directory)
    schedule = WarmupCosine(
        lr=options["learning_rate"],
        min_lr=options["learning_rate"] / 10,
        warmup_steps=min(1, options["steps"] - 1),
        total_steps=options["steps"],
    )
    with Trainer(
        Plan.load(root / "plans/training.ssp"),
        evaluation_plan=Plan.load(root / "plans/evaluation.ssp"),
        schedules={"optimizer_group.0.lr": schedule},
    ) as trainer:
        source, validation = Inputs(options), Inputs(options, split="val").example()
        if options.get("resume"):
            trainer.load(options["resume"])
        else:
            trainer.initialize(Initialize(model_config(options), seed=options["seed"]))
        with (root / "training/evaluation.jsonl").open("a", buffering=1) as stream:

            def evaluate(owner, result=None):
                value = sum(
                    float(owner.evaluate(args=[args], kwargs=[kwargs]))
                    for args, kwargs in zip(
                        validation["args"], validation["kwargs"], strict=True
                    )
                )
                record = {"step": owner.step_count, "validation_loss": value}
                stream.write(json.dumps(record) + "\n")
                print(record, flush=True)

            evaluate(trainer)
            trainer.fit(
                source,
                steps=options["steps"],
                run_dir=root / "training",
                callbacks=[evaluate],
                checkpoint_every=options.get("checkpoint_every", 5),
            )


if __name__ == "__main__":
    run(sys.argv[1])
