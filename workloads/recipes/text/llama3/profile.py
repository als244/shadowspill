"""Measure ordinary-allocator tasks and host copies in a separate GPU process."""

import sys

from shadowspill import Program, apply_profiles, measure_communication, measure_program
from workloads.recipes.phases import settings


def run(directory):
    root, options = settings(directory)
    timing = {
        "warmup_seconds": options["warmup_seconds"],
        "measure_seconds": options["measure_seconds"],
    }
    for name in ("training", "evaluation"):
        print(f"profiling {name}", flush=True)
        program = Program.load(root / f"programs/{name}.ssp")
        profiles = measure_program(program, **timing)
        profiles.save(root / f"profiles/{name}.ssp")
        apply_profiles(program, profiles).save(root / f"profiled/{name}.ssp")
    measure_communication(**timing).save(root / "communication.ssp")


if __name__ == "__main__":
    run(sys.argv[1])
