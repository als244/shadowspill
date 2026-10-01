"""Packed text as an ordinary resumable iterable and microbatch functions."""

from __future__ import annotations

from collections import deque

import torch

from .geometry import search_geometries
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
        examples[name] = [
            [
                tokens[:, start : start + capacity].clone(),
                targets[:, start : start + capacity].clone(),
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
    def __init__(self, data, *, name, tokens, accumulation, max_seq_len):
        self.name = name
        self.accumulation = accumulation
        self.packer = data.train_packer(tokens, max_seq_len)
        self.next_step = 0
        self.trained_tokens = 0
        self.last_stats = {}
        self.last_documents = []

    def __iter__(self):
        return self

    def __next__(self):
        start = self.next_step * self.accumulation
        indices = range(start, start + self.accumulation)
        parts = [self.packer.microbatch(index) for index in indices]
        normalizer = self.packer.trained_tokens(indices)
        self.last_stats = self.packer.stats(indices)
        self.last_documents = [self.packer.documents(index) for index in indices]
        self.next_step += 1
        self.trained_tokens += normalizer
        return {"candidates": {self.name: parts}, "normalizer": normalizer}

    def state_dict(self):
        return {
            "candidate": self.name,
            "accumulation": self.accumulation,
            "next_step": self.next_step,
            "trained_tokens": self.trained_tokens,
            "packs": self.packer.packs,
            "pending": list(self.packer.pending),
            "next_document": self.packer.next_document,
        }

    def load_state_dict(self, state):
        if (state["candidate"], state["accumulation"]) != (
            self.name,
            self.accumulation,
        ):
            raise ValueError("saved text source uses a different microbatch geometry")
        self.next_step = state["next_step"]
        self.trained_tokens = state["trained_tokens"]
        self.packer.packs = [list(pack) for pack in state["packs"]]
        self.packer.pending = deque(state["pending"])
        self.packer.next_document = state["next_document"]


def validation_update(data, *, name, tokens, max_seq_len, microbatches):
    packer = data.validation_packer(tokens, max_seq_len)
    indices = range(microbatches)
    return {
        "candidates": {name: [packer.microbatch(i) for i in indices]},
        "normalizer": packer.trained_tokens(indices),
    }
