"""A step shared in units of the sequence cap, each rank taking its share."""

from __future__ import annotations

import torch

from workloads.recipes.text.experiment.inputs import call_inputs


def concatenated(units):
    """Several unit microbatches as one: tokens and targets side by side,
    their length entries after one another, the shape the planner assumed
    for a microbatch of that many sequences' capacity."""
    tokens, targets, lengths = zip(*units, strict=True)
    return [torch.cat(tokens, 1), torch.cat(targets, 1), torch.cat(lengths)]


def unit_ranges(shares, step_units):
    """Each rank's first unit and unit count within a step of ``step_units``."""
    total = sum(shares)
    if step_units % total:
        raise ValueError("the step's units must divide by the shares' total")
    scale = step_units // total
    starts, at = [], 0
    for share in shares:
        starts.append((at, share * scale))
        at += share * scale
    return starts


class UnitUpdates:
    """One rank's microbatches of a step, concatenated from its units."""

    def __init__(self, data, *, name, max_seq_len, capacity, step_units, shares, rank):
        if capacity % max_seq_len:
            raise ValueError("a microbatch holds whole units of max_seq_len tokens")
        self.name = name
        self.unit = max_seq_len
        self.per_microbatch = capacity // max_seq_len
        self.step_units = step_units
        self.first, self.units = unit_ranges(shares, step_units)[rank]
        if self.units % self.per_microbatch:
            raise ValueError(
                "a rank's units must fill whole microbatches of its geometry"
            )
        self.rank = rank
        self.packer = data.train_packer(max_seq_len, max_seq_len)
        self.next_step = 0
        self.trained_tokens = 0
        self.last_stats = {}
        self.last_documents = []

    def __iter__(self):
        return self

    def __next__(self):
        base = self.next_step * self.step_units
        mine = range(base + self.first, base + self.first + self.units)
        parts = [
            concatenated(
                [
                    self.packer.microbatch(unit)
                    for unit in mine[i : i + self.per_microbatch]
                ]
            )
            for i in range(0, self.units, self.per_microbatch)
        ]
        normalizer = self.packer.trained_tokens(range(base, base + self.step_units))
        self.last_stats = self.packer.stats(mine)
        self.last_documents = [
            [
                entry
                for unit in mine[i : i + self.per_microbatch]
                for entry in self.packer.documents(unit)
            ]
            for i in range(0, self.units, self.per_microbatch)
        ]
        self.next_step += 1
        self.trained_tokens += normalizer
        return call_inputs(
            {"candidates": {self.name: parts}, "normalizer": normalizer}, self.name
        )

    def state_dict(self):
        return {
            "candidate": self.name,
            "step_units": self.step_units,
            "first": self.first,
            "units": self.units,
            "per_microbatch": self.per_microbatch,
            "next_step": self.next_step,
            "trained_tokens": self.trained_tokens,
            "packs": self.packer.packs,
            "pending": list(self.packer.pending),
            "next_document": self.packer.next_document,
        }

    def load_state_dict(self, state):
        keys = ("candidate", "step_units", "first", "units", "per_microbatch")
        if tuple(state[key] for key in keys) != (
            self.name,
            self.step_units,
            self.first,
            self.units,
            self.per_microbatch,
        ):
            raise ValueError("saved text source uses another share or geometry")
        from collections import deque

        self.next_step = state["next_step"]
        self.trained_tokens = state["trained_tokens"]
        self.packer.packs = [list(pack) for pack in state["packs"]]
        self.packer.pending = deque(state["pending"])
        self.packer.next_document = state["next_document"]


def validation(data, *, name, max_seq_len, capacity, step_units, shares, rank, steps):
    """This rank's held-out microbatches over ``steps`` steps' worth of units,
    its share of each, normalized over every rank's units."""
    packer = data.validation_packer(max_seq_len, max_seq_len)
    first, units = unit_ranges(shares, step_units)[rank]
    per_microbatch = capacity // max_seq_len
    parts = []
    for step in range(steps):
        base = step * step_units
        mine = range(base + first, base + first + units)
        parts.extend(
            concatenated(
                [packer.microbatch(unit) for unit in mine[i : i + per_microbatch]]
            )
            for i in range(0, units, per_microbatch)
        )
    normalizer = packer.trained_tokens(range(steps * step_units))
    return call_inputs({"candidates": {name: parts}, "normalizer": normalizer}, name)[
        "args"
    ]
