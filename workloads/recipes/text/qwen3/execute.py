"""Fresh Trainer workers consume one shared Plan and bounded rank-local inputs."""

import json
import sys
from contextlib import nullcontext
from datetime import timedelta

import torch.distributed as dist

from shadowspill import Plan, Trainer
from shadowspill.training.logging import DistributedLogger
from shadowspill.training.schedules import WarmupCosine
from workloads.recipes.phases import settings
from workloads.recipes.text.data.calls import Inputs

from ..initialization import Initialize
from .common import model_config


def run(directory):
    root, options = settings(directory)
    selected = json.loads((root / "selected.json").read_text())
    options["sequences_per_example"] = selected["sequences_per_example"]
    ranks = options["ranks"]
    if ranks > 1:
        dist.init_process_group("gloo", timeout=timedelta(seconds=7200))
    try:
        rank = dist.get_rank() if ranks > 1 else 0
        schedule = WarmupCosine(
            lr=options["learning_rate"],
            min_lr=options["learning_rate"] / 10,
            warmup_steps=min(1, options["steps"] - 1),
            total_steps=options["steps"],
        )
        with (
            Trainer(
                Plan.load(root / "plan.ssp"),
                rank_map={i: i for i in range(ranks)},
                schedules={"optimizer_group.0.lr": schedule},
                metric_reducer=lambda result: {
                    "work_units": options["sequences_per_step"]
                    * options["sequence_length"]
                },
            ) as trainer,
            (
                DistributedLogger(dist.group.WORLD, run_dir=root / "training")
                if ranks > 1
                else nullcontext()
            ) as logger,
        ):
            trainer.initialize(
                Initialize("qwen3moe", model_config(options), seed=options["seed"])
            )
            trainer.fit(
                Inputs(options, rank=rank, world_size=ranks),
                steps=options["steps"],
                run_dir=root / f"training/rank-{rank:05d}",
                logger=logger,
                checkpoint_every=options["steps"],
            )
    finally:
        if ranks > 1:
            dist.destroy_process_group()


if __name__ == "__main__":
    run(sys.argv[1])
