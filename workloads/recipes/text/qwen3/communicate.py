"""Measure each candidate's declared communication on the actual participating GPUs."""

import sys
from datetime import timedelta

import torch.distributed as dist

from shadowspill import (
    ProfiledProgram,
    apply_profiles,
    measure_communication,
    measure_overlap,
)
from workloads.recipes.phases import settings

from .common import batch_sizes


def run(directory):
    root, options = settings(directory)
    dist.init_process_group("gloo", timeout=timedelta(seconds=7200))
    try:
        for size in batch_sizes(options):
            print(f"calibrating batch-{size}", flush=True)
            profiled = ProfiledProgram.load(root / f"profiled/batch-{size}.ssp")
            communication = measure_communication(
                program=profiled.program,
                rank_map={rank: rank for rank in range(options["ranks"])},
                device="cuda:0",
                warmup_seconds=options["warmup_seconds"],
                measure_seconds=options["measure_seconds"],
            )
            overlap = measure_overlap(
                profiled,
                communication,
                rank_map={rank: rank for rank in range(options["ranks"])},
                device="cuda:0",
            )
            if dist.get_rank() == 0:
                communication.save(root / f"communication/batch-{size}.ssp")
                overlap.save(root / f"overlap/batch-{size}.ssp")
                profiles = profiled.profiles.merge(overlap)
                profiles.save(root / f"profiles/batch-{size}.ssp")
                apply_profiles(profiled.program, profiles).save(
                    root / f"profiled/batch-{size}.ssp"
                )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    run(sys.argv[1])
