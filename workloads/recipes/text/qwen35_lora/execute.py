"""Execute saved Plans with HF base chunks; no build, measurement or planning."""

import json
import sys
from contextlib import ExitStack

from shadowspill import Plan, Trainer
from shadowspill.training.logging import Wandb
from shadowspill.training.schedules import WarmupCosine
from workloads.recipes.text.data.calls import Inputs
from workloads.recipes.text.phases import settings

from .evaluate import evaluate
from .initialize import Initialize


def run(directory):
    root, options = settings(directory)
    options.update(json.loads((root / "selected.json").read_text()))
    schedule = WarmupCosine(
        lr=options["learning_rate"],
        min_lr=options["learning_rate"] / 10,
        warmup_steps=min(1, options["steps"] - 1),
        total_steps=options["steps"],
    )
    (root / "training").mkdir(exist_ok=True)
    with ExitStack() as stack:
        trainer = stack.enter_context(
            Trainer(
                Plan.load(root / "plans/training.ssp"),
                evaluation_plan=Plan.load(root / "plans/evaluation.ssp"),
                schedules={"optimizer_group.0.lr": schedule},
                frozen_state_id=options["frozen_state_id"],
            )
        )
        with Initialize(options) as initialize:
            if options.get("resume"):
                trainer.load(options["resume"], fill_tensor=initialize)
            else:
                trainer.initialize(initialize)
        source = Inputs(options)
        logger = (
            stack.enter_context(
                Wandb(
                    project=options["wandb_project"],
                    run_dir=root / "training",
                    config=options,
                )
            )
            if options.get("wandb_project")
            else None
        )
        stream = stack.enter_context(
            (root / "training/evaluation.jsonl").open(
                "a",
                buffering=1,
            )
        )

        def validation(owner, result):
            if owner.step_count % 2 == 0 or owner.step_count == options["steps"]:
                evaluate(owner, options, stream, logger)

        trainer.fit(
            source,
            steps=options["steps"],
            run_dir=root / "training",
            logger=logger,
            callbacks=[validation],
            checkpoint_every=2,
            keep_last=2,
        )


if __name__ == "__main__":
    run(sys.argv[1])
