"""Measure the shared Program's local tasks on one representative GPU."""

import sys

from shadowspill import Program, apply_profiles, measure_program
from workloads.recipes.phases import settings


def run(directory):
    root, options = settings(directory)
    program = Program.load(root / "program.ssp")
    profiles = measure_program(
        program,
        warmup_seconds=options["warmup_seconds"],
        measure_seconds=options["measure_seconds"],
    )
    profiles.save(root / "profiles.ssp")
    apply_profiles(program, profiles).save(root / "profiled.ssp")


if __name__ == "__main__":
    run(sys.argv[1])
