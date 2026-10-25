"""Fresh distributed Trainer workers; no model construction or preparation."""

import sys
from datetime import timedelta

import torch.distributed as dist

from shadowspill import Plan, Trainer
from shadowspill.training.logging import DistributedLogger
from shadowspill.training.schedules import Constant
from workloads.recipes.phases import settings

from .data import Inputs
from .initialize import fill_tensor


def run(directory):
    root, options = settings(directory)
    dist.init_process_group("gloo", timeout=timedelta(seconds=120))
    try:
        rank = dist.get_rank()
        wandb = (
            {"project": options["wandb_project"], "mode": options["wandb_mode"]}
            if options.get("wandb_project")
            else None
        )
        with (
            Trainer(
                Plan.load(root / "plan.ssp"),
                rank_map={rank: rank for rank in range(options["ranks"])},
                schedules={"optimizer_group.0.lr": Constant(0.001)},
                metric_reducer=lambda result: {"work_units": 8},
            ) as trainer,
            DistributedLogger(
                dist.group.WORLD, run_dir=root / "training", wandb=wandb
            ) as logger,
        ):
            trainer.initialize(fill_tensor)
            trainer.fit(
                Inputs(rank),
                steps=options["steps"],
                run_dir=root / f"training/rank-{rank:05d}",
                logger=logger,
                checkpoint_every=options["steps"],
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    run(sys.argv[1])
