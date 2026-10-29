"""Packed examples and their changing token normalizer as ordinary call inputs."""

import torch

from ..data.source import PackedUpdates, candidate_data, validation_update


def ranks(options):
    """How many workers the run uses; one unless the config says ``ranks``."""
    count = options.get("ranks", 1)
    if type(count) is not int or count < 1:
        raise ValueError("ranks must be a positive integer")
    return count


def shares(options):
    """How the step's microbatches divide among the ranks: ``shares``, one
    positive integer per rank, or equal shares."""
    count = ranks(options)
    values = options.get("shares")
    if values is None:
        return [1] * count
    if (
        not isinstance(values, list)
        or len(values) != count
        or any(type(value) is not int or value < 1 for value in values)
    ):
        raise ValueError("shares lists one positive integer per rank")
    return values


def rank_options(options, rank=0):
    """The options rank ``rank`` plans and trains with: its share of the
    step's tokens. ``max_tokens_per_step`` is the whole step's, over every
    rank, and every share must be whole sequences."""
    parts = shares(options)
    tokens = options["max_tokens_per_step"]
    if tokens % (sum(parts) * options["max_seq_len"]):
        raise ValueError(
            "max_tokens_per_step must divide into whole sequences for every "
            "rank's share"
        )
    return {**options, "max_tokens_per_step": tokens // sum(parts) * parts[rank]}


def rank_counts(options, microbatches, rank):
    """Every rank's microbatches per step when rank ``rank`` takes
    ``microbatches``: the shares scaled to it."""
    parts = shares(options)
    if microbatches % parts[rank]:
        raise ValueError("a rank's microbatches must be a multiple of its share")
    unit = microbatches // parts[rank]
    return [unit * part for part in parts]


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
        counts=rank_counts(options, choice["microbatches"], rank),
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
