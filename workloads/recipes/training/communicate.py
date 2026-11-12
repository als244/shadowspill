"""Calibrate each saved Program's collectives on the participating GPUs."""

from datetime import timedelta

import torch.distributed as dist

from orchestration.communication import communication_probes
from orchestration.lifecycle import Lifecycle
from orchestration.profiling import profiling_options
from shadowspill import (
    OverlapProfileCache,
    ProfiledProgram,
    Program,
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
            candidates = program_paths(options, root=root)
            if not candidates:
                print(
                    "communicate: no built candidates; skipping calibration", flush=True
                )
                return
            with timing.phase("communication.requests"):
                program, collectives = communication_probes(
                    Program.load(root / "programs" / relative)
                    for relative in candidates
                )
            timing.event(
                "communication.probes",
                candidates=len(candidates),
                collectives=len(collectives),
            )
            with timing.phase("communication.measure"):
                communication = measure_communication(
                    program=program,
                    collectives=collectives,
                    rank_map={rank: rank for rank in range(options["ranks"])},
                    device="cuda:0",
                    options=profiling_options(options),
                )
            del program
            for relative in candidates:
                print(f"calibrating {relative}", flush=True)
                timing.event("candidate", program=str(relative))
                with timing.phase("profiled.load"):
                    profiled = ProfiledProgram.load(root / "profiled" / relative)
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
