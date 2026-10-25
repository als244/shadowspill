"""Calibrate declared collectives on all ranks, separately from task profiling."""

import sys
from datetime import timedelta

import torch.distributed as dist

from shadowspill import Program, measure_communication
from workloads.recipes.phases import settings


def run(directory):
    root, options = settings(directory)
    dist.init_process_group("gloo", timeout=timedelta(seconds=120))
    try:
        communication = measure_communication(
            program=Program.load(root / "program.ssp"),
            device="cuda:0",
            warmup_seconds=options["warmup_seconds"],
            measure_seconds=options["measure_seconds"],
        )
        if dist.get_rank() == 0:
            communication.save(root / "communication.ssp")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    run(sys.argv[1])
