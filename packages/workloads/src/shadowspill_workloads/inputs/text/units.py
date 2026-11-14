"""Text in units of the sequence cap, divided among ranks and packed per rank.

A unit is one packed sequence of ``max_seq_len`` tokens. A run's held-out
evaluation is ``eval_tokens`` tokens of whole units, whatever microbatch
geometry each rank chose: the ranks divide the units in proportion to their
shares, and each concatenates its units into microbatches of its own
capacity, padding the last when its units do not fill it. Every rank's
losses are normalized over all ranks' units, so they sum to the evaluation's
mean.
"""

from __future__ import annotations

import numpy as np
import torch

from .documents import IGNORE


def concatenated(units):
    """Several unit microbatches as one: tokens and targets side by side,
    their length entries after one another, the shape the planner assumed
    for a microbatch of that many sequences' capacity."""
    tokens, targets, lengths = zip(*units, strict=True)
    return [torch.cat(tokens, 1), torch.cat(targets, 1), torch.cat(lengths)]


def unit_ranges(shares, units):
    """Each rank's first unit and unit count when ``units`` divide among the
    ranks in proportion to ``shares``."""
    total = sum(shares)
    if units % total:
        raise ValueError("the units must divide by the shares' total")
    scale = units // total
    ranges, at = [], 0
    for share in shares:
        ranges.append((at, share * scale))
        at += share * scale
    return ranges


def held_out_units(tokens, max_seq_len, shares):
    """How many units ``eval_tokens`` holds, refused unless they are whole
    units that divide among the shares."""
    if tokens <= 0 or tokens % max_seq_len:
        raise ValueError("eval_tokens must be whole sequences of max_seq_len tokens")
    units = tokens // max_seq_len
    if units % sum(shares):
        raise ValueError(
            "eval_tokens must divide into whole sequences for every rank's share"
        )
    return units


def padding_unit(packer):
    """A unit that trains nothing: end-of-text tokens, ignored targets and
    one padding sequence, laid out as the packer lays out its units."""
    tokens, targets, lengths = packer.microbatch(0)
    padding = np.zeros(lengths.numel(), dtype=np.int32)
    padding[0] = tokens.shape[1]
    return [
        torch.full_like(tokens, packer.source.eos),
        torch.full_like(targets, IGNORE),
        torch.from_numpy(padding),
    ]


def held_out(data, *, name, max_seq_len, capacity, tokens, shares, rank):
    """This rank's held-out calls for one evaluation: its share of
    ``tokens`` tokens of units, packed into microbatches of ``capacity``,
    each with the weight that makes the ranks' losses sum to the mean over
    every rank's units."""
    if capacity % max_seq_len:
        raise ValueError("a microbatch holds whole units of max_seq_len tokens")
    units = held_out_units(tokens, max_seq_len, shares)
    packer = data.validation_packer(max_seq_len, max_seq_len)
    first, count = unit_ranges(shares, units)[rank]
    per_microbatch = capacity // max_seq_len
    mine = [packer.microbatch(unit) for unit in range(first, first + count)]
    mine += [padding_unit(packer)] * (-count % per_microbatch)
    normalizer = packer.trained_tokens(range(units))
    if normalizer <= 0:
        raise ValueError("the held-out units contain no valid targets")
    weight = torch.tensor(1.0 / normalizer, dtype=torch.float32)
    return [
        (concatenated(mine[at : at + per_microbatch]), weight)
        for at in range(0, len(mine), per_microbatch)
    ]
