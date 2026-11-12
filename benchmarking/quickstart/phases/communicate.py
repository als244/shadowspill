"""Calibrate communication and paired interference for saved candidates."""

from benchmarking.spill import resolve as spill_pool
from orchestration.communication import communication_probes
from orchestration.profiling import profiling_options
from orchestration.workers import workers
from shadowspill import (
    CommunicationInfo,
    OverlapProfileCache,
    ProfiledProgram,
    Program,
    apply_profiles,
)

from .common import paths


def probes(root, index):
    return communication_probes(
        Program.load(root / "programs" / entry["file"])
        for entry in index["candidates"]
        if "failure" not in entry
    )


def run(request):
    root, index = paths(request)
    if request.get("communication_info"):
        CommunicationInfo.load(request["communication_info"]).save(
            root / "communication.ssp"
        )
        return
    if not any("failure" not in entry for entry in index["candidates"]):
        print("communicate: no built candidates; skipping calibration", flush=True)
        return
    program, collectives = probes(root, index)
    from shadowspill import measure_communication, measure_overlap

    with workers(index.get("world_size", 1), rank_map=request.get("rank_map")) as group:
        print(
            f"communicate: {len(collectives)} collective probes, {group.count} workers",
            flush=True,
        )
        communication = measure_communication(
            program=program,
            collectives=collectives,
            rank_map=group.rank_map,
            device=request["device"],
            options=profiling_options(request),
            spill=spill_pool(request),
        )
        if group.rank == 0:
            communication.save(root / "communication.ssp")
        cache = OverlapProfileCache(root / "cache")
        for entry in index["candidates"]:
            path = root / "profiled" / entry["file"]
            if "failure" in entry or not path.is_file():
                continue
            print(f"overlap: {entry['name']}", flush=True)
            profiled = ProfiledProgram.load(path)
            overlap = measure_overlap(
                profiled,
                communication,
                device=request["device"],
                rank_map=group.rank_map,
                cache=cache,
            )
            if group.rank == 0:
                overlap.save(root / "overlap" / entry["file"])
                profiles = profiled.profiles.merge(overlap)
                profiles.save(root / "profiles" / entry["file"])
                apply_profiles(profiled.program, profiles).save(path)
