"""The tokens an engine trains on: synthetic by default, drawn per rank from
a seed, or windows of a corpus ``prepare_corpus`` tokenized once per
tokenizer family.

Synthetic tokens and targets are independent uniform draws over the
vocabulary, as the quickstart's text recipe draws them, so a synthetic run
measures the step and nothing about the data. A corpus window is a
contiguous span of the tokenized documents, one per rank per step, with the
targets one token on; the spans of all ranks and steps are disjoint until the
corpus wraps, which the window index reports.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

#: What ``prepare_corpus`` writes beside ``train.bin`` and ``val.bin``.
META_FILE = "meta.json"
TRAIN_FILE = "train.bin"


def synthetic_tokens(vocabulary: int, count: int, *, seed: int, rank: int) -> Any:
    """``(tokens, targets)``, each ``(1, count)`` int64, distinct per rank."""

    import torch

    if vocabulary < 1 or count < 1:
        raise ValueError("vocabulary and token count must be positive")
    generator = torch.Generator().manual_seed(seed * 1_000_003 + rank)
    shape = (1, count)
    return (
        torch.randint(vocabulary, shape, generator=generator, device="cpu"),
        torch.randint(vocabulary, shape, generator=generator, device="cpu"),
    )


class Corpus:
    """A pre-tokenized corpus directory: ``train.bin`` of uint32 token ids
    and ``meta.json`` naming the tokenizer and its vocabulary."""

    def __init__(self, directory: str | Path) -> None:
        import numpy

        self.directory = Path(directory)
        meta_path = self.directory / META_FILE
        if not meta_path.is_file():
            raise FileNotFoundError(f"{self.directory} has no {META_FILE}")
        self.meta = json.loads(meta_path.read_text())
        for key in ("tokenizer", "vocab_size", "dtype"):
            if key not in self.meta:
                raise ValueError(f"{meta_path} lacks {key!r}")
        if self.meta["dtype"] != "uint32":
            raise ValueError(f"{meta_path}: only uint32 corpora are read")
        self.tokens = numpy.memmap(
            self.directory / TRAIN_FILE, dtype=numpy.uint32, mode="r"
        )
        if len(self.tokens) < 2:
            raise ValueError(
                f"{self.directory}: the corpus holds fewer than two tokens"
            )

    @property
    def vocab_size(self) -> int:
        return int(self.meta["vocab_size"])

    def __len__(self) -> int:
        return len(self.tokens)

    def check_vocabulary(self, model_vocabulary: int) -> None:
        """A model cannot read ids its vocabulary does not have rows for."""
        if self.vocab_size > model_vocabulary:
            raise ValueError(
                f"the corpus was tokenized for a vocabulary of {self.vocab_size},"
                f" larger than the model's {model_vocabulary}"
            )

    def window(self, *, rank: int, world_size: int, step: int, tokens: int) -> Any:
        """``(tokens, targets, wrapped)``: the span this rank reads at this
        step, each ``(1, tokens)`` int64, and how many times the corpus has
        been exhausted before it (zero until the data repeats)."""

        import numpy
        import torch

        if tokens < 1 or world_size < 1 or not 0 <= rank < world_size or step < 0:
            raise ValueError("invalid window request")
        spans_per_pass = (len(self.tokens) - 1) // tokens
        if spans_per_pass < 1:
            raise ValueError(f"the corpus is shorter than one {tokens}-token window")
        index = step * world_size + rank
        wrapped, position = divmod(index, spans_per_pass)
        start = position * tokens
        span = numpy.asarray(self.tokens[start : start + tokens + 1], dtype=numpy.int64)
        values = torch.from_numpy(span)
        return values[:-1].unsqueeze(0), values[1:].unsqueeze(0), wrapped


__all__ = ["META_FILE", "TRAIN_FILE", "Corpus", "synthetic_tokens"]
