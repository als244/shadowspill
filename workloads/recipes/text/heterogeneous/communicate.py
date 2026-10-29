"""Calibrate each node's own Programs' collectives with every rank taking part.

Nodes search their own geometries, so they hold different candidate sets;
every measurement is a collective over all ranks, so the ranks walk the
union of the nodes' slots together. For a slot a node lacks it measures its
own Program of the same kind, whose collectives are the same, and saves
nothing; for its own slots it saves the calibration its first worker made.
"""

import json
import sys
from datetime import timedelta

import torch.distributed as dist

from shadowspill import Program, measure_communication
from workloads.recipes.phases import settings
from workloads.recipes.text.launch import first_rank

from .run import check_section


def slots(options, inventory):
    """This node's ``candidate/kind`` slots, in order."""
    kinds = ["training", *(["evaluation"] if options.get("eval_every", 0) else [])]
    return [
        f"{candidate}/{kind}" for candidate in inventory["candidates"] for kind in kinds
    ]


def shared_slots(everyone):
    """Every node's slots as one ordered list, each once."""
    return sorted({slot for mine in everyone for slot in mine})


def stand_in(slot, mine):
    """The node's own slot of the same kind, for a slot it does not hold."""
    kind = slot.rsplit("/", 1)[1]
    for own in mine:
        if own.rsplit("/", 1)[1] == kind:
            return own
    raise ValueError(f"this node built no {kind} Program to measure {slot} with")


def run(directory):
    root, options = settings(directory)
    nodes, _, _ = check_section(options)
    count = options["ranks"]
    mine = slots(options, json.loads((root / "candidates.json").read_text()))
    dist.init_process_group("gloo", timeout=timedelta(seconds=7200))
    try:
        everyone = [None] * dist.get_world_size()
        dist.all_gather_object(everyone, mine)
        saving = dist.get_rank() == first_rank(nodes, options["node"])
        for slot in shared_slots(everyone):
            own = slot in mine
            measured = slot if own else stand_in(slot, mine)
            print(
                f"calibrating {slot}" + ("" if own else f" with {measured}"), flush=True
            )
            communication = measure_communication(
                program=Program.load(root / "programs" / f"{measured}.ssp"),
                rank_map={rank: rank for rank in range(count)},
                device="cuda:0",
                warmup_seconds=options.get("warmup_seconds", 0.05),
                measure_seconds=options.get("measure_seconds", 0.18),
            )
            if own and saving:
                communication.save(root / "communication" / f"{slot}.ssp")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    run(sys.argv[1])
