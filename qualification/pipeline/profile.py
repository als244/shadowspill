"""Ordinary-allocator GPU measurement, independent of model construction."""

import json
import sys
from pathlib import Path

from mlops.dispatch import deterministic_kernels

from benchmarking.spill import resolve
from shadowspill import Program, apply_profiles, measure_communication, measure_program
from shadowspill.orchestration.profiling import profiling_options


def run(directory: Path) -> None:
    settings = json.loads((directory / "settings.json").read_text())
    program = Program.load(directory / "program.ssp")
    timing = {"options": profiling_options(settings)}
    with deterministic_kernels(settings.get("deterministic", False)):
        profiles = measure_program(
            program,
            external_headroom_bytes=settings.get("external_headroom_bytes", 512 << 20),
            **timing,
        )
    profiles.save(directory / "profiles.ssp")
    apply_profiles(program, profiles).save(directory / "profiled.ssp")
    measure_communication(spill=resolve(settings), **timing).save(
        directory / "communication.ssp"
    )


if __name__ == "__main__":
    run(Path(sys.argv[1]))
