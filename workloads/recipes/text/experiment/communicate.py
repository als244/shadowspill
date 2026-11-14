"""Calibrate each candidate Program's collectives on the participating GPUs."""

import json
import sys
from datetime import timedelta

import torch.distributed as dist

from shadowspill import (
    ProfiledProgram,
    apply_profiles,
    measure_communication,
    measure_overlap,
)
from shadowspill.orchestration.profiling import profiling_options
from workloads.recipes.phases import settings

from .group import ranks


def run(directory):
    root, options = settings(directory)
    count = ranks(options)
    inventory = json.loads((root / "candidates.json").read_text())
    names = ["training", *(["evaluation"] if options.get("eval_every", 0) else [])]
    dist.init_process_group("gloo", timeout=timedelta(seconds=7200))
    try:
        for candidate in inventory["candidates"]:
            for name in names:
                relative = f"{candidate}/{name}.ssp"
                print(f"calibrating {relative}", flush=True)
                profiled = ProfiledProgram.load(root / "profiled" / relative)
                communication = measure_communication(
                    program=profiled.program,
                    rank_map={rank: rank for rank in range(count)},
                    device="cuda:0",
                    options=profiling_options(options),
                )
                overlap = measure_overlap(
                    profiled,
                    communication,
                    rank_map={rank: rank for rank in range(count)},
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


if __name__ == "__main__":
    run(sys.argv[1])
