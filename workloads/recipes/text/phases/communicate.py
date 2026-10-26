"""Calibrate each saved Program's collectives on the participating GPUs."""

from datetime import timedelta

import torch.distributed as dist

from shadowspill import Program, measure_communication
from workloads.recipes.phases import settings

from .candidates import program_paths


def run(directory):
    root, options = settings(directory)
    dist.init_process_group("gloo", timeout=timedelta(seconds=7200))
    try:
        for relative in program_paths(options):
            print(f"calibrating {relative}", flush=True)
            communication = measure_communication(
                program=Program.load(root / "programs" / relative),
                rank_map={rank: rank for rank in range(options["ranks"])},
                device="cuda:0",
                warmup_seconds=options["warmup_seconds"],
                measure_seconds=options["measure_seconds"],
            )
            if dist.get_rank() == 0:
                communication.save(root / "communication" / relative)
    finally:
        dist.destroy_process_group()
