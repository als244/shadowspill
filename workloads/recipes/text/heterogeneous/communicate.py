"""Calibrate each node's own Programs' collectives with every rank taking part."""

import json
import sys
from datetime import timedelta

import torch.distributed as dist

from shadowspill import Program, measure_communication
from workloads.recipes.phases import settings
from workloads.recipes.text.launch import first_rank

from .run import check_section


def run(directory):
    root, options = settings(directory)
    nodes, _, _ = check_section(options)
    count = options["ranks"]
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
                # Each node's first worker saves the node's own calibration.
                if dist.get_rank() == first_rank(nodes, options["node"]):
                    communication.save(root / "communication" / relative)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    run(sys.argv[1])
