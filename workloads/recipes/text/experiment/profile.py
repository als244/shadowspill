"""Measure saved tasks with the ordinary allocator; no execution pools."""

import json
import sys
from pathlib import Path

from shadowspill import Program, apply_profiles, measure_communication, measure_program
from workloads.recipes.phases import settings

from . import targets
from .inputs import ranks


def hardware():
    """The profiled device's memory and the host's, for budgets a config
    leaves out; planning runs without a device and reads the record."""
    import torch

    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    return {
        "device": properties.name,
        "device_total_bytes": int(properties.total_memory),
        **host_memory(),
    }


def host_memory(meminfo="/proc/meminfo"):
    """``MemTotal`` and ``MemAvailable`` of this host, in bytes."""
    values = {}
    for line in Path(meminfo).read_text().splitlines():
        key, _, rest = line.partition(":")
        if key in {"MemTotal", "MemAvailable"}:
            values[key] = int(rest.split()[0]) << 10
    return {
        "host_total_bytes": values["MemTotal"],
        "host_available_bytes": values["MemAvailable"],
    }


def run(directory):
    root, options = settings(directory)
    root = targets.artifacts(root, options)
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
    if ranks(options) == 1:
        measure_communication(**timing).save(root / "communication.ssp")
    (root / "hardware.json").write_text(json.dumps(hardware(), indent=2) + "\n")


if __name__ == "__main__":
    run(sys.argv[1])
