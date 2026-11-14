"""Ordinary-allocator GPU measurements, independent of execution workers."""

from functools import partial

from shadowspill import Program, TaskProfileCache, apply_profiles, measure_program
from shadowspill.orchestration.candidates import attempt
from shadowspill.orchestration.launch.workers import workers
from shadowspill.orchestration.profiling import profiling_options

from .common import paths, write


def run(request):
    root, index = paths(request)
    count = index.get("preparation_world_size", 1)
    binding = request if count > 1 else {**request, "rank_map": None}
    with workers(count, rank_map=binding.get("rank_map")) as group:
        _profile(request, root, index, group)


def _profile(request, root, index, group):
    if group.rank == 0:
        (root / "profiled" / "index.json").unlink(missing_ok=True)
    timing = {"options": profiling_options(request)}
    cache = TaskProfileCache(root / "cache")
    for entry in index["candidates"]:
        if "failure" in entry:
            continue
        print(f"profile: {entry['name']}", flush=True)
        program = Program.load(root / "programs" / entry["file"])
        measure = partial(
            measure_program,
            program,
            device=request["device"],
            rank_map=group.rank_map if group.count > 1 else None,
            external_headroom_bytes=request["external_headroom_bytes"],
            cache=cache,
            **timing,
        )
        profiles = (
            measure()
            if group.count > 1
            else attempt(
                entry,
                "profile",
                measure,
                request["device"],
            )
        )
        if profiles is not None and group.rank == 0:
            profiles.save(root / "profiles" / entry["file"])
            apply_profiles(program, profiles).save(root / "profiled" / entry["file"])
        del profiles, program
    if group.rank == 0:
        write(root / "profiled" / "index.json", index)
