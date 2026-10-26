"""Ordinary-allocator GPU measurements, independent of execution workers."""

from functools import partial

from shadowspill import Program, apply_profiles, measure_program

from .candidates import attempt
from .common import paths, write


def run(request):
    root, index = paths(request)
    (root / "profiled" / "index.json").unlink(missing_ok=True)
    timing = {
        "warmup_seconds": request["warmup_seconds"],
        "measure_seconds": request["measure_seconds"],
    }
    for entry in index["candidates"]:
        if "failure" in entry:
            continue
        print(f"profile: {entry['name']}", flush=True)
        program = Program.load(root / "programs" / entry["file"])
        profiles = attempt(
            entry,
            "profile",
            partial(
                measure_program,
                program,
                device=request["device"],
                external_headroom_bytes=request["external_headroom_bytes"],
                **timing,
            ),
            request["device"],
        )
        if profiles is not None:
            profiles.save(root / "profiles" / entry["file"])
            apply_profiles(program, profiles).save(root / "profiled" / entry["file"])
        del profiles, program
    write(root / "profiled" / "index.json", index)
