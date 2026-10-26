"""Measure local tasks on one representative GPU; no distributed rendezvous."""

import sys

from shadowspill import Program, apply_profiles, measure_communication, measure_program
from workloads.recipes.phases import settings

from .common import batch_sizes


def run(directory):
    root, options = settings(directory)
    timing = {key: options[key] for key in ("warmup_seconds", "measure_seconds")}
    for size in batch_sizes(options):
        name = f"batch-{size}.ssp"
        print(f"profiling {name}", flush=True)
        program = Program.load(root / "programs" / name)
        profiles = measure_program(
            program,
            external_headroom_bytes=options.get("external_headroom_mib", 768) << 20,
            **timing,
        )
        profiles.save(root / "profiles" / name)
        apply_profiles(program, profiles).save(root / "profiled" / name)
    if options["ranks"] == 1:
        communication = measure_communication(**timing)
        for size in batch_sizes(options):
            communication.save(root / f"communication/batch-{size}.ssp")


if __name__ == "__main__":
    run(sys.argv[1])
