"""Runnable two-or-more-device, non-text training example using public APIs."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn

from shadowspill.pytorch import ProfilingOptions
from shadowspill.training import Distributed, Trainer
from shadowspill.training.backends import ShadowSpill
from shadowspill.training.logging import DistributedLogger
from shadowspill.training.schedules import Constant


class Regression(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(16, 32), nn.SiLU(), nn.Linear(32, 8))
        self.register_buffer("updates", torch.tensor(0, dtype=torch.int64))

    def forward(self, value):
        self.updates.add_(1)
        return self.network(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--execution-gib", type=float, default=2)
    parser.add_argument("--spill-gib", type=float, default=1)
    parser.add_argument("--profiling-seconds", type=float, default=1)
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-mode", choices=("online", "offline"), default="online")
    args = parser.parse_args()
    from mlops.optim import AdamW

    dist.init_process_group("gloo")
    try:
        rank, world = dist.get_rank(), dist.get_world_size()
        generator = torch.Generator().manual_seed(711 + rank)
        source = [
            (
                torch.randn(8, 16, generator=generator),
                torch.randn(8, 8, generator=generator),
            )
            for _ in range(args.steps)
        ]

        def objective(model, data):
            inputs, targets = data
            # SUM contributions divided by all target elements in the global step.
            return (model(inputs) - targets).square().sum() / (world * 8 * 8)

        wandb = (
            {"project": args.wandb_project, "mode": args.wandb_mode}
            if args.wandb_project
            else None
        )
        with ShadowSpill(
            device="auto",
            execution_gib=args.execution_gib,
            spill_gib=args.spill_gib,
            control_group=dist.group.WORLD,
            artifact_store=args.run_dir / f"rank-{rank:05d}" / "artifacts",
            partition="whole",
            profiling_options=ProfilingOptions(
                conditioning_seconds=args.profiling_seconds,
                measurement_seconds=args.profiling_seconds,
            ),
        ) as backend:
            group = dist.new_group(backend="nccl", device_id=backend.device)
            try:
                torch.manual_seed(12)
                with (
                    Trainer(
                        Regression(),
                        objective=objective,
                        optimizer=AdamW,
                        optimizer_args={"opt_state_dtype": torch.float32},
                        schedules={"lr": Constant(0.001)},
                        backend=backend,
                        distributed=Distributed(group),
                        # Work is application-defined: eight independent examples.
                        metric_reducer=lambda result: {"work_units": 8},
                    ) as trainer,
                    DistributedLogger(
                        dist.group.WORLD, run_dir=args.run_dir, wandb=wandb
                    ) as logger,
                ):
                    trainer.prepare(source[0])
                    trainer.fit(
                        source,
                        steps=args.steps,
                        run_dir=args.run_dir,
                        logger=logger,
                        startup_diagnostics=True,
                        checkpoint_every=args.steps,
                    )
            finally:
                dist.destroy_process_group(group)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
