"""Measure saved tasks with the ordinary allocator; no execution pools."""

import json
import sys

from shadowspill import Program, apply_profiles, measure_communication, measure_program
from workloads.recipes.phases import settings


def run(directory):
    root, options = settings(directory)
    timing = {
        "warmup_seconds": options.get("warmup_seconds", 0.05),
        "measure_seconds": options.get("measure_seconds", 0.18),
    }
    inventory = json.loads((root / "candidates.json").read_text())
    names = ["training", *(["evaluation"] if options.get("eval_every", 0) else [])]
    for relative in (
        f"{candidate}/{name}.ssp"
        for candidate in inventory["candidates"]
        for name in names
    ):
        print(f"profiling {relative}", flush=True)
        program = Program.load(root / "programs" / relative)
        profiles = measure_program(
            program,
            external_headroom_bytes=options.get("external_headroom_mib", 512) << 20,
            **timing,
        )
        profiles.save(root / "profiles" / relative)
        apply_profiles(program, profiles).save(root / "profiled" / relative)
    measure_communication(**timing).save(root / "communication.ssp")


if __name__ == "__main__":
    run(sys.argv[1])
