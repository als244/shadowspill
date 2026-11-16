"""Caller-owned next-token batches; only NumPy and the lightweight client."""

import hashlib
import json
from pathlib import Path

import numpy as np

from shadowspill_client.text import text_batch, text_datum


class TokenBatches:
    def __init__(
        self,
        directory,
        sequence_length,
        sequences_per_step,
        *,
        split="train",
        normalize=True,
    ):
        if min(sequence_length, sequences_per_step) < 1 or split not in {
            "train",
            "val",
        }:
            raise ValueError(
                "positive sequence/batch sizes and train/val split required"
            )
        root = Path(directory).resolve()
        with (root / "meta.json").open("rb") as stream:
            raw = stream.read((64 << 10) + 1)
        if len(raw) > 64 << 10:
            raise ValueError("corpus metadata exceeds 64 KiB")
        metadata = json.loads(raw)
        if metadata.get("dtype", "uint32") != "uint32":
            raise ValueError("expected a uint32 token corpus")
        self.vocab_size = metadata["vocab_size"]
        if type(self.vocab_size) is not int or self.vocab_size < 1:
            raise ValueError("corpus vocabulary size must be a positive integer")
        path = root / f"{split}.bin"
        facts = path.stat()
        self.tokens = np.memmap(path, dtype=np.uint32, mode="r")
        self.length, self.count = sequence_length, sequences_per_step
        self.positions = sequence_length * sequences_per_step
        if type(normalize) is not bool:
            raise TypeError("normalize must be boolean")
        self.normalize = normalize
        self.identity = {
            "path": str(path),
            "size_bytes": facts.st_size,
            "mtime_ns": facts.st_mtime_ns,
            "metadata_sha256": hashlib.sha256(raw).hexdigest(),
            "sequence_length": self.length,
            "sequences_per_step": self.count,
        }
        mask_path = root / f"{split}.mask.bin"
        self.masks = None
        if mask_path.exists():
            mask_facts = mask_path.stat()
            if mask_facts.st_size != len(self.tokens):
                raise ValueError("corpus loss mask must contain one byte per token")
            self.masks = np.memmap(mask_path, dtype=np.uint8, mode="r")
            self.identity["mask"] = {
                "path": str(mask_path),
                "size_bytes": mask_facts.st_size,
                "mtime_ns": mask_facts.st_mtime_ns,
            }
        elif metadata.get("mask") is not None:
            raise ValueError("corpus metadata declares a missing loss mask")
        if not len(self):
            raise ValueError("corpus is shorter than one complete update")

    def __len__(self):
        return (len(self.tokens) - 1) // self.positions

    def __getitem__(self, index):
        if type(index) is not int or not 0 <= index < len(self):
            raise IndexError("update lies outside the corpus")
        start = index * self.positions
        offsets = range(start, start + self.positions, self.length)
        sequences = [
            self.tokens[offset : offset + self.length + 1] for offset in offsets
        ]
        masks = [
            None
            if self.masks is None
            else self.masks[offset : offset + self.length + 1]
            for offset in offsets
        ]
        if self.normalize:
            return text_batch(sequences, loss_masks=masks)
        # Each Datum attends independently. Composed objectives can use unit
        # weights and apply their common normalization at the optimizer instead.
        return [
            text_datum(
                tokens,
                loss_mask=mask,
                weights=1.0,
            )
            for tokens, mask in zip(sequences, masks, strict=True)
        ]


def validate_vocabulary(preview, *sources):
    """Check token corpora before preparation or worker acquisition."""
    vocabulary = preview.config.model.config["architecture"]["vocab_size"]
    if any(source is not None and source.vocab_size > vocabulary for source in sources):
        raise ValueError("corpus vocabulary exceeds model vocabulary")
