"""Calibrate each saved Program's collectives on the participating GPUs."""

from datetime import timedelta

import torch.distributed as dist

from shadowspill import (
    ProfiledProgram,
    apply_profiles,
    measure_communication,
    measure_overlap,
)
from workloads.recipes.phases import settings

from .candidates import program_paths


def run(directory):
    root, options = settings(directory)
    dist.init_process_group("gloo", timeout=timedelta(seconds=7200))
    try:
        for relative in program_paths(options):
            print(f"calibrating {relative}", flush=True)
            profiled = ProfiledProgram.load(root / "profiled" / relative)
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
                communication.save(root / "communication" / relative)
                overlap.save(root / "overlap" / relative)
                profiles = profiled.profiles.merge(overlap)
                profiles.save(root / "profiles" / relative)
                apply_profiles(profiled.program, profiles).save(
                    root / "profiled" / relative
                )
    finally:
        dist.destroy_process_group()
