"""Build one shared Program using one GPU and no process group."""

import sys
from functools import partial

import torch
from mlops.optim import AdamW

from shadowspill import Parallelism, build_step
from workloads.recipes.phases import settings

from .data import Inputs
from .model import Regression, objective


def run(directory):
    root, options = settings(directory)
    with torch.device("meta"):
        model = Regression()
    example = Inputs(0).example()
    program = build_step(
        model,
        optimizer=AdamW(model.parameters(), lr=0.001, opt_state_dtype=torch.float32),
        objective=partial(objective, ranks=options["ranks"]),
        example_args=example["args"][0],
        example_kwargs=example["kwargs"][0],
        parallelism=Parallelism.dp(ranks_per_replica=options["ranks"]),
        partition="whole",
    )
    program.save(root / "program.ssp")


if __name__ == "__main__":
    run(sys.argv[1])
