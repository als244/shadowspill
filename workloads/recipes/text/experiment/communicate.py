"""Calibrate each candidate Program's collectives on the participating GPUs."""

import json
import sys
from datetime import timedelta

import torch.distributed as dist

from shadowspill import Program, measure_communication
from workloads.recipes.phases import settings

from . import targets
from .inputs import ranks


def run(directory):
    root, options = settings(directory)
    target = targets.read(root, options)
    root = targets.artifacts(root, options)
    count = ranks(options)
    inventory = json.loads((root / "candidates.json").read_text())
    names = ["training", *(["evaluation"] if options.get("eval_every", 0) else [])]
    dist.init_process_group("gloo", timeout=timedelta(seconds=7200))
    try:
        for candidate in inventory["candidates"]:
            for name in names:
                relative = f"{candidate}/{name}.ssp"
                print(f"calibrating {relative}", flush=True)
                communication = measure_communication(
                    program=Program.load(root / "programs" / relative),
                    rank_map={rank: rank for rank in range(count)},
                    device="cuda:0",
                    warmup_seconds=options.get("warmup_seconds", 0.05),
                    measure_seconds=options.get("measure_seconds", 0.18),
                )
                # The first worker of each host saves its target's calibration.
                if dist.get_rank() == targets.first_rank(options, target["host"]):
                    communication.save(root / "communication" / relative)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    run(sys.argv[1])
