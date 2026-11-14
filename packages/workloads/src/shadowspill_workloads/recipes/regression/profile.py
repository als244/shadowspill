"""Measure the shared Program's local tasks on one representative GPU."""

import sys

from shadowspill import Program, apply_profiles, measure_program
from shadowspill.orchestration.profiling import profiling_options
from shadowspill.training.preparation.phases import settings


def run(directory):
    root, options = settings(directory)
    program = Program.load(root / "program.ssp")
    profiles = measure_program(
        program,
        options=profiling_options(options),
    )
    profiles.save(root / "profiles.ssp")
    apply_profiles(program, profiles).save(root / "profiled.ssp")


if __name__ == "__main__":
    run(sys.argv[1])
