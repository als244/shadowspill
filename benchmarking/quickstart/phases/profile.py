"""Ordinary-allocator GPU measurements, independent of execution workers."""

from shadowspill import Program, apply_profiles, measure_program

from .common import paths


def run(request):
    root, index = paths(request)
    timing = {
        "warmup_seconds": request["warmup_seconds"],
        "measure_seconds": request["measure_seconds"],
    }
    for entry in index["candidates"]:
        print(f"profile: {entry['name']}", flush=True)
        program = Program.load(root / "programs" / entry["file"])
        profiles = measure_program(
            program,
            device=request["device"],
            external_headroom_bytes=request["external_headroom_bytes"],
            **timing,
        )
        profiles.save(root / "profiles" / entry["file"])
        apply_profiles(program, profiles).save(root / "profiled" / entry["file"])
