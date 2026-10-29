"""Packed text as an ordinary resumable iterable and microbatch functions."""

from __future__ import annotations

from collections import deque

import torch

from ..geometry import search_geometries
from .packing import seq_slots


def candidate_data(
    data,
    *,
    max_seq_len,
    max_tokens_per_step,
    min_tokens_per_microbatch=None,
    max_tokens_per_microbatch=None,
):
    if max_tokens_per_step % max_seq_len:
        raise ValueError("the text planning example needs whole fixed-length sequences")
    geometries, skipped = search_geometries(
        max_tokens_per_step // max_seq_len,
        sequence_length=max_seq_len,
        min_tokens_per_microbatch=min_tokens_per_microbatch,
        max_tokens_per_microbatch=max_tokens_per_microbatch,
    )
    if not geometries:
        raise ValueError("no text microbatch geometry satisfies the requested bounds")
    tokens = torch.randint(data.vocab_size, (1, max_tokens_per_step))
    targets = torch.roll(tokens, -1, 1)
    targets[:, max_seq_len - 1 :: max_seq_len] = -100
    normalizer = (max_tokens_per_step // max_seq_len) * (max_seq_len - 1)
    examples, functions, geometry = {}, {}, {}
    for sequences, accumulation in geometries:
        capacity = sequences * max_seq_len
        name = f"{capacity}_tokens"
        lengths = torch.zeros(
            seq_slots(capacity, data.min_tokens_per_seq), dtype=torch.int32
        )
        lengths[:sequences] = max_seq_len
        # A slice clone can retain the full-step stride of the singleton axis.
        # Match the packer's canonical [1, capacity] layout for exact guards.
        examples[name] = [
            [
                tokens[:, start : start + capacity].flatten().clone().unsqueeze(0),
                targets[:, start : start + capacity].flatten().clone().unsqueeze(0),
                lengths.clone(),
            ]
            for start in range(0, max_tokens_per_step, capacity)
        ]
        functions[name] = TextMicrobatches(name)
        geometry[name] = (capacity, accumulation)
    return (
        {"candidates": examples, "normalizer": normalizer},
        functions,
        geometry,
        skipped,
    )


class TextMicrobatches:
    def __init__(self, name):
        self.name = name

    def __call__(self, update):
        normalizer = update["normalizer"]
        if normalizer <= 0:
            raise ValueError("the packed update contains no valid targets")
        for values in update["candidates"][self.name]:
            yield values, 1.0 / normalizer


class PackedUpdates:
    """A step's microbatches for one rank. ``counts`` says how many of the
    step's microbatches each rank takes, in rank order and consecutively;
    every rank's normalizer is the step's trained-token count over all of
    them, so each rank's loss is its share of the step's mean (see the loss
    convention) whatever its share."""

    def __init__(self, data, *, name, tokens, counts, max_seq_len, rank=0):
        counts = [int(count) for count in counts]
        if not counts or any(count < 1 for count in counts):
            raise ValueError("every rank takes at least one microbatch of a step")
        if not 0 <= rank < len(counts):
            raise ValueError("rank must lie inside the ranks counted")
        self.name = name
        self.counts = counts
        self.accumulation = sum(counts)
        self.rank, self.world_size = rank, len(counts)
        self.packer = data.train_packer(tokens, max_seq_len)
        self.next_step = 0
        self.trained_tokens = 0
        self.last_stats = {}
        self.last_documents = []

    def __iter__(self):
        return self

    def __next__(self):
        start = self.next_step * self.accumulation
        step = range(start, start + self.accumulation)
        first = start + sum(self.counts[: self.rank])
        mine = range(first, first + self.counts[self.rank])
        parts = [self.packer.microbatch(index) for index in mine]
        normalizer = self.packer.trained_tokens(step)
        self.last_stats = self.packer.stats(mine)
        self.last_documents = [self.packer.documents(index) for index in mine]
        self.next_step += 1
        self.trained_tokens += normalizer
        return {"candidates": {self.name: parts}, "normalizer": normalizer}

    def state_dict(self):
        return {
            "candidate": self.name,
            "counts": self.counts,
            "rank": self.rank,
            "next_step": self.next_step,
            "trained_tokens": self.trained_tokens,
            "packs": self.packer.packs,
            "pending": list(self.packer.pending),
            "next_document": self.packer.next_document,
        }

    def load_state_dict(self, state):
        counts = state.get("counts", [state.get("accumulation")])
        if (state["candidate"], list(counts)) != (self.name, self.counts):
            raise ValueError("saved text source uses a different microbatch geometry")
        if state.get("rank", 0) != self.rank:
            raise ValueError("saved text source belongs to another rank")
        self.next_step = state["next_step"]
        self.trained_tokens = state["trained_tokens"]
        self.packer.packs = [list(pack) for pack in state["packs"]]
        self.packer.pending = deque(state["pending"])
        self.packer.next_document = state["next_document"]


def validation_update(
    data, *, name, tokens, max_seq_len, microbatches, rank=0, world_size=1
):
    """``microbatches`` held-out microbatches for this rank, distinct from
    every other rank's, normalized by the trained tokens of all ranks' so
    that the ranks' losses sum to the mean over the whole evaluation."""
    packer = data.validation_packer(tokens, max_seq_len)
    mine = range(rank * microbatches, (rank + 1) * microbatches)
    return {
        "candidates": {name: [packer.microbatch(i) for i in mine]},
        "normalizer": packer.trained_tokens(range(world_size * microbatches)),
    }
