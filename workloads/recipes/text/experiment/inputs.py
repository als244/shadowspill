"""Packed examples and their changing token normalizer as ordinary call inputs."""

import torch

from ..data.source import PackedUpdates, candidate_data, validation_update
from .group import ranks


def candidates(options, data):
    bounds = {
        key: options.get("planning_" + key)
        for key in ("min_tokens_per_microbatch", "max_tokens_per_microbatch")
    }
    fixed = options.get("max_tokens_per_microbatch")
    if fixed is not None:
        if any(value is not None for value in bounds.values()):
            raise ValueError("choose a fixed text microbatch size or search bounds")
        bounds = {key: fixed for key in bounds}
    return candidate_data(
        data,
        max_seq_len=options["max_seq_len"],
        max_tokens_per_step=options["max_tokens_per_step"],
        **bounds,
    )


def call_inputs(update, name):
    normalizer = update["normalizer"]
    if normalizer <= 0:
        raise ValueError("the packed update contains no valid targets")
    return {
        "args": [
            (values, torch.tensor(1.0 / normalizer, dtype=torch.float32))
            for values in update["candidates"][name]
        ]
    }


class Inputs(PackedUpdates):
    def __next__(self):
        return call_inputs(super().__next__(), self.name)


def source(data, options, choice, rank=0):
    """This rank's training source: ``choice`` counts one rank's microbatches,
    so the step holds ``ranks`` times as many."""
    return Inputs(
        data,
        name=choice["candidate"],
        tokens=choice["tokens_per_microbatch"],
        counts=[choice["microbatches"]] * ranks(options),
        max_seq_len=options["max_seq_len"],
        rank=rank,
    )


def validation(data, options, choice, rank=0):
    count = options.get("eval_microbatches", 1)
    if count < 1:
        raise ValueError("eval_microbatches must be positive")
    update = validation_update(
        data,
        name=choice["candidate"],
        tokens=choice["tokens_per_microbatch"],
        max_seq_len=options["max_seq_len"],
        microbatches=count,
        rank=rank,
        world_size=ranks(options),
    )
    return call_inputs(update, choice["candidate"])["args"]
