"""Caller-owned next-token batches; only NumPy and the lightweight client."""

import hashlib
import json
from pathlib import Path

import numpy as np

from shadowspill_client.text import text_batch, text_datum


class _TokenCorpus:
    def __init__(self, directory, split):
        if split not in {"train", "val"}:
            raise ValueError("train/val split required")
        root = Path(directory).resolve()
        with (root / "meta.json").open("rb") as stream:
            raw = stream.read((64 << 10) + 1)
        if len(raw) > 64 << 10:
            raise ValueError("corpus metadata exceeds 64 KiB")
        metadata = json.loads(raw)
        self.metadata = metadata
        if metadata.get("dtype", "uint32") != "uint32":
            raise ValueError("expected a uint32 token corpus")
        self.vocab_size = metadata["vocab_size"]
        if type(self.vocab_size) is not int or self.vocab_size < 1:
            raise ValueError("corpus vocabulary size must be a positive integer")
        path = root / f"{split}.bin"
        facts = path.stat()
        self.tokens = np.memmap(path, dtype=np.uint32, mode="r")
        self.identity = {
            "path": str(path),
            "size_bytes": facts.st_size,
            "mtime_ns": facts.st_mtime_ns,
            "metadata_sha256": hashlib.sha256(raw).hexdigest(),
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


class TokenBatches(_TokenCorpus):
    """Fixed contiguous windows for next-token pretraining."""

    def __init__(
        self,
        directory,
        sequence_length,
        sequences_per_step,
        *,
        split="train",
        normalize=True,
    ):
        if min(sequence_length, sequences_per_step) < 1:
            raise ValueError("positive sequence/batch sizes required")
        super().__init__(directory, split)
        self.length, self.count = sequence_length, sequences_per_step
        self.positions = sequence_length * sequences_per_step
        if type(normalize) is not bool:
            raise TypeError("normalize must be boolean")
        self.normalize = normalize
        self.identity.update(sequence_length=self.length, sequences_per_step=self.count)
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


class ConversationBatches(_TokenCorpus):
    """Complete variable-length conversations, grouped by a global token budget.

    Conversation separators come from prepare_chat's metadata and cannot occur
    inside a rendered conversation. Overlong and unsupervised conversations are
    skipped; no conversation is truncated or joined to another. This helper
    chooses logical batches, not physical packing or worker placement.
    """

    SCAN_TOKENS = 64 << 10

    def __init__(
        self,
        directory,
        *,
        max_sequence_length,
        tokens_per_step,
        max_sequences=1024,
        split="train",
    ):
        for value in (max_sequence_length, tokens_per_step, max_sequences):
            if type(value) is not int or value < 1:
                raise ValueError(
                    "conversation and batch limits must be positive integers"
                )
        if max_sequence_length > tokens_per_step:
            raise ValueError(
                "maximum conversation length exceeds the step token budget"
            )
        super().__init__(directory, split)
        self.separator = self.metadata.get("eos_id")
        if (
            self.metadata.get("rendered") != "chat_template"
            or self.masks is None
            or type(self.separator) is not int
            or not 0 <= self.separator < self.vocab_size
        ):
            raise ValueError(
                "expected a prepare_chat corpus with masks and conversation separators"
            )
        if self.tokens[-1] != self.separator:
            raise ValueError("chat corpus has an incomplete final conversation")
        self.max_length, self.capacity, self.max_sequences = (
            max_sequence_length,
            tokens_per_step,
            max_sequences,
        )
        self.identity.update(
            max_sequence_length=max_sequence_length,
            tokens_per_step=tokens_per_step,
            max_sequences=max_sequences,
        )

    def __iter__(self):
        sequences, masks = [], []
        positions = 0
        start = 0
        found = False
        for offset in range(0, len(self.tokens), self.SCAN_TOKENS):
            ends = np.flatnonzero(
                self.tokens[offset : offset + self.SCAN_TOKENS] == self.separator
            )
            for relative in ends:
                end = offset + int(relative)
                first, start = start, end + 1
                length = end - first - 1  # one extra original token for shifting
                if not 0 < length <= self.max_length:
                    continue
                tokens = self.tokens[first:end]
                mask = self.masks[first:end]
                if not np.any(mask[1:]):
                    continue
                if (
                    positions + length > self.capacity
                    or len(sequences) == self.max_sequences
                ):
                    yield text_batch(sequences, loss_masks=masks)
                    sequences, masks, positions = [], [], 0
                sequences.append(tokens)
                masks.append(mask)
                positions += length
                found = True
        if sequences:
            yield text_batch(sequences, loss_masks=masks)
        if not found:
            raise ValueError(
                "chat corpus contains no supervised conversations "
                "within the length limit"
            )


def validate_vocabulary(preview, *sources):
    """Check token corpora before preparation or worker acquisition."""
    vocabulary = preview.config.model.config["architecture"]["vocab_size"]
    if any(source is not None and source.vocab_size > vocabulary for source in sources):
        raise ValueError("corpus vocabulary exceeds model vocabulary")
