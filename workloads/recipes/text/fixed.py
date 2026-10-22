"""Fixed-length next-token updates from an ordinary tokenized text corpus."""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path

import numpy as np
import torch


class TokenUpdates:
    """An update iterable and its microbatch splitter, usable with ``Trainer``.

    Read ``train.bin`` or ``val.bin`` (uint32 token IDs) and ``meta.json`` from the text
    preparation recipe. Each update has exactly ``sequences_per_step`` windows
    of ``sequence_length`` tokens. Windows attend independently; targets are
    the next tokens in the corpus. No padding or synthetic targets are used.
    """

    def __init__(
        self,
        directory: str | Path,
        *,
        sequence_length: int,
        sequences_per_step: int,
        split: str = "train",
        rank: int = 0,
        world_size: int = 1,
        max_sequences_per_microbatch: int | None = None,
    ) -> None:
        if min(sequence_length, sequences_per_step, world_size) <= 0:
            raise ValueError("sequence and batch sizes must be positive")
        if not 0 <= rank < world_size:
            raise ValueError("rank must lie inside world_size")
        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'")
        self.directory = Path(directory).resolve()
        self.split = split
        meta = json.loads((self.directory / "meta.json").read_text())
        if meta.get("dtype", "uint32") != "uint32":
            raise ValueError("TokenUpdates expects uint32 token IDs")
        self.vocab_size = int(meta["vocab_size"])
        self.tokens = np.memmap(
            self.directory / f"{split}.bin", dtype=np.uint32, mode="r"
        )
        self.sequence_length = sequence_length
        self.sequences_per_step = sequences_per_step
        self.rank, self.world_size = rank, world_size
        self.max_sequences_per_microbatch = (
            sequences_per_step
            if max_sequences_per_microbatch is None
            else max_sequences_per_microbatch
        )
        if self.max_sequences_per_microbatch < 1:
            raise ValueError("max_sequences_per_microbatch must be positive")
        self.tokens_per_step = sequence_length * sequences_per_step
        self.next_step = 0
        if len(self) == 0:
            raise ValueError("the corpus is shorter than one complete update")

    def __len__(self):
        return (len(self.tokens) - 1) // (self.tokens_per_step * self.world_size)

    def __iter__(self):
        return self

    def _update(self, index):
        start = (index * self.world_size + self.rank) * self.tokens_per_step
        span = torch.from_numpy(
            np.array(
                self.tokens[start : start + self.tokens_per_step + 1], dtype=np.int64
            )
        )
        shape = (self.sequences_per_step, self.sequence_length)
        return span[:-1].reshape(shape).clone(), span[1:].reshape(shape).clone()

    def example(self):
        """A representative update without consuming training data."""
        return self._update(0)

    def __next__(self):
        if self.next_step >= len(self):
            raise StopIteration
        update = self._update(self.next_step)
        self.next_step += 1
        return update

    @property
    def candidates(self):
        """Every divisor of the local sequence count; Trainer searches orderings."""
        return {
            f"{size}_sequences": self.splitter(sequences_per_microbatch=size)
            for size in range(1, self.sequences_per_step + 1)
            if self.sequences_per_step % size == 0
            and size <= self.max_sequences_per_microbatch
        }

    def splitter(self, *, sequences_per_microbatch):
        """Fix a size and return the function Trainer applies to each update."""
        if (
            sequences_per_microbatch <= 0
            or self.sequences_per_step % sequences_per_microbatch
        ):
            raise ValueError("sequences_per_microbatch must divide sequences_per_step")
        return partial(
            self.microbatches, sequences_per_microbatch=sequences_per_microbatch
        )

    def microbatches(self, update, *, sequences_per_microbatch):
        tokens, targets = update
        size = sequences_per_microbatch
        if size <= 0 or self.sequences_per_step % size:
            raise ValueError("sequences_per_microbatch must divide sequences_per_step")
        for start in range(0, self.sequences_per_step, size):
            yield (
                (
                    tokens[start : start + size].reshape(1, -1).clone(),
                    targets[start : start + size].reshape(1, -1).clone(),
                    (self.sequence_length,) * size,
                ),
                1.0 / (self.tokens_per_step * self.world_size),
            )

    def state_dict(self):
        return {
            "directory": str(self.directory),
            "split": self.split,
            "next_step": self.next_step,
            "geometry": [
                self.sequence_length,
                self.sequences_per_step,
                self.rank,
                self.world_size,
                self.max_sequences_per_microbatch,
            ],
        }

    def load_state_dict(self, state):
        expected = self.state_dict()
        if any(
            state[key] != expected[key] for key in ("directory", "split", "geometry")
        ):
            raise ValueError("checkpoint corpus, split or geometry differs")
        if not 0 <= state["next_step"] <= len(self):
            raise ValueError("checkpoint data position is outside the corpus")
        self.next_step = state["next_step"]
