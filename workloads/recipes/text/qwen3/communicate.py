"""Measure each candidate's declared communication on the actual participating GPUs."""

import sys
from datetime import timedelta

import torch.distributed as dist

from shadowspill import Program, measure_communication
from workloads.recipes.phases import settings

from .common import batch_sizes


def run(directory):
    root, options = settings(directory)
    dist.init_process_group("gloo", timeout=timedelta(seconds=7200))
    try:
        for size in batch_sizes(options):
            print(f"calibrating batch-{size}", flush=True)
            communication = measure_communication(
                program=Program.load(root / f"programs/batch-{size}.ssp"),
                rank_map={rank: rank for rank in range(options["ranks"])},
                device="cuda:0",
                warmup_seconds=options["warmup_seconds"],
                measure_seconds=options["measure_seconds"],
            )
            if dist.get_rank() == 0:
                communication.save(root / f"communication/batch-{size}.ssp")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    run(sys.argv[1])
