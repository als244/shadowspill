"""The data a run trains on: tokenized documents, packed whole."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from training.documents import TokenDocuments
from training.packing import Packer, seq_slots, uniform_microbatch


class PackedTokens:
    """The documents of a directory ``prepare_data`` wrote -- ``train.bin``,
    ``val.bin`` and ``meta.json`` -- packed into microbatches.

    The trainer says how long a sequence may be, ``max_seq_len``; a longer
    document is dropped, truncated to its first ``max_seq_len`` tokens, or
    spliced into pieces of ``max_seq_len`` tokens, as ``long_documents`` --
    ``"drop"``, ``"truncate"`` or ``"splice"`` -- says. A microbatch has room
    for one sequence per ``min_tokens_per_seq`` tokens -- its ``seq_slots``,
    the entries of its lengths tensor -- which bounds how many sequences it
    holds; shorter ones still pack, the microbatch simply fills its entries
    first. The best fit looks ``window`` sequences ahead.
    """

    def __init__(
        self,
        directory: str | Path,
        long_documents: str = "drop",
        min_tokens_per_seq: int = 128,
        window: int = 64,
    ) -> None:
        self.directory = Path(directory)
        self.long_documents = long_documents
        self.min_tokens_per_seq = min_tokens_per_seq
        self.window = window
        self.meta = json.loads((self.directory / "meta.json").read_text())

    @property
    def vocab_size(self) -> int:
        return int(self.meta["vocab_size"])

    def train_packer(self, tokens: int, max_seq_len: int) -> Packer:
        """The training documents of at most ``max_seq_len`` tokens, in
        microbatches of ``tokens`` tokens."""

        return self._packer("train.bin", tokens, max_seq_len)

    def validation_packer(self, tokens: int, max_seq_len: int) -> Packer:
        """The validation documents of at most ``max_seq_len`` tokens, in
        microbatches of ``tokens`` tokens."""

        return self._packer("val.bin", tokens, max_seq_len)

    def examples(
        self, documents: int, count: int, max_seq_len: int
    ) -> list[list[torch.Tensor]]:
        """What planning assumes: ``count`` microbatches of ``documents`` whole
        documents of ``max_seq_len`` tokens, every one with tensors of its own."""

        tokens = documents * max_seq_len
        slots = seq_slots(tokens, self.min_tokens_per_seq)
        return [
            uniform_microbatch(tokens, slots, max_seq_len, self.vocab_size)
            for _ in range(count)
        ]

    def _packer(self, name: str, tokens: int, max_seq_len: int) -> Packer:
        source = TokenDocuments(
            self.directory / name,
            int(self.meta["eos_id"]),
            max_seq_len,
            self.long_documents,
        )
        return Packer(
            source, tokens, seq_slots(tokens, self.min_tokens_per_seq), self.window
        )
