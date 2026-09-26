"""A small tokenized dataset and a tiny language model for the training tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EOS = 0
VOCAB = 64
#: Document lengths, the end-of-text token included; 700 exceeds the tests'
#: max_seq_len of 512 and is dropped.
LENGTHS = (5, 300, 120, 700, 250, 90, 400, 33, 512, 180, 60, 222, 310, 77, 150)


def write_dataset(directory: Path, repeats: int = 40) -> Path:
    """``train.bin``, ``val.bin`` and ``meta.json`` as ``prepare_data`` writes
    them, from ``LENGTHS`` repeated: token ids 1..VOCAB-1, each document ended
    by ``EOS``."""

    directory.mkdir(parents=True, exist_ok=True)
    generator = np.random.default_rng(0)
    for name, count in (("train.bin", repeats), ("val.bin", 4)):
        documents = [
            np.append(generator.integers(1, VOCAB, length - 1), EOS)
            for _ in range(count)
            for length in LENGTHS
        ]
        np.concatenate(documents).astype(np.uint32).tofile(directory / name)
    meta = {"vocab_size": VOCAB, "eos_id": EOS}
    (directory / "meta.json").write_text(json.dumps(meta))
    return directory


class TinyModel(nn.Module):
    """An embedding and a head: enough of a language model to train."""

    def __init__(self, vocab: int = VOCAB, width: int = 8) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, width)
        self.head = nn.Linear(width, vocab)

    def loss(
        self, tokens: torch.Tensor, targets: torch.Tensor, *, seq_lens: torch.Tensor
    ) -> torch.Tensor:
        logits = self.head(self.embed(tokens)).float()
        summed = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
        return summed / targets.numel()


#: What ``note`` was called with, in order: a stand-in for a process-wide setting.
NOTES: list[str] = []


def note(text: str) -> None:
    NOTES.append(text)
