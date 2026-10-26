"""Measure local tasks on one GPU without creating runtime pools or DP groups."""

from shadowspill import Program, apply_profiles, measure_communication, measure_program
from workloads.recipes.phases import settings

from .candidates import program_paths


def run(directory):
    root, options = settings(directory)
    timing = {key: options[key] for key in ("warmup_seconds", "measure_seconds")}
    for relative in program_paths(options):
        print(f"profiling {relative}", flush=True)
        program = Program.load(root / "programs" / relative)
        profiles = measure_program(
            program,
            external_headroom_bytes=options.get("external_headroom_mib", 512) << 20,
            **timing,
        )
        profiles.save(root / "profiles" / relative)
        apply_profiles(program, profiles).save(root / "profiled" / relative)
    if options.get("ranks", 1) == 1:
        communication = measure_communication(**timing)
        for relative in program_paths(options):
            communication.save(root / "communication" / relative)
