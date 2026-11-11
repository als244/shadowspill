"""Calibrate each saved Program's collectives on the participating GPUs."""

from datetime import timedelta

import torch.distributed as dist

from benchmarking.plan_runner.lifecycle import Lifecycle
from shadowspill import (
    OverlapProfileCache,
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
        cache = OverlapProfileCache(
            root / "cache",
            mode="refresh" if options.get("refresh_profiles", False) else "contribute",
        )
        with Lifecycle(
            root / f"logs/communication-rank-{dist.get_rank():05d}.jsonl",
            rank=dist.get_rank(),
        ) as timing:
            for relative in program_paths(options, root=root):
                print(f"calibrating {relative}", flush=True)
                timing.event("candidate", program=str(relative))
                with timing.phase("profiled.load"):
                    profiled = ProfiledProgram.load(root / "profiled" / relative)
                with timing.phase("communication.measure"):
                    communication = measure_communication(
                        program=profiled.program,
                        rank_map={rank: rank for rank in range(options["ranks"])},
                        device="cuda:0",
                        warmup_seconds=options["warmup_seconds"],
                        measure_seconds=options["measure_seconds"],
                    )
                with timing.phase("overlap.measure"):
                    overlap = measure_overlap(
                        profiled,
                        communication,
                        rank_map={rank: rank for rank in range(options["ranks"])},
                        device="cuda:0",
                        cache=cache,
                    )
                with timing.phase("artifacts.save"):
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
    import sys

    run(sys.argv[1])
