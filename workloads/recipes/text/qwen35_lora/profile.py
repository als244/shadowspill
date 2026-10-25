"""Ordinary-allocator profiling; model initialization and runtime pools stay out."""

import sys

from shadowspill import Program, apply_profiles, measure_communication, measure_program
from workloads.recipes.phases import settings

from .candidates import batch_sizes


def run(directory):
    root, options = settings(directory)
    timing = {
        "warmup_seconds": options["warmup_seconds"],
        "measure_seconds": options["measure_seconds"],
    }
    for size in batch_sizes(options):
        for name in ("training", "evaluation"):
            relative = f"batch-{size}/{name}.ssp"
            print(f"profiling {relative}", flush=True)
            program = Program.load(root / "programs" / relative)
            profiles = measure_program(program, **timing)
            profiles.save(root / "profiles" / relative)
            apply_profiles(program, profiles).save(root / "profiled" / relative)
    measure_communication(**timing).save(root / "communication.ssp")


if __name__ == "__main__":
    run(sys.argv[1])
