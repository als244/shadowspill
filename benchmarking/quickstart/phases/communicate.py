"""Calibrate communication and paired interference for saved candidates."""

from benchmarking.spill import resolve as spill_pool
from orchestration.workers import workers
from shadowspill import (
    CommunicationInfo,
    OverlapProfileCache,
    ProfiledProgram,
    Program,
    apply_profiles,
)
from shadowspill.communication.requests import collective_requests

from .common import paths


def probes(root, index):
    programs = [
        Program.load(root / "programs" / entry["file"])
        for entry in index["candidates"]
        if "failure" not in entry
    ]
    first = programs[0]
    if any(program.parallelism != first.parallelism for program in programs):
        raise ValueError("quickstart candidates must declare the same parallelism")
    requests = {
        request
        for program in programs
        for request in collective_requests(program, None)
    }
    return first, [request.to_dict() for request in sorted(requests)]


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
            warmup_seconds=request["warmup_seconds"],
            measure_seconds=request["measure_seconds"],
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
