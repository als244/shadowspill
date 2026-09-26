"""Sequences packed into fixed-size microbatches, the same for every run.

A sequence is a whole document, or a piece of a long one (see
``training.documents``). Sequences are packed back to back into microbatches of
``tokens`` tokens by a best fit over a small look-ahead; the tail a microbatch
cannot fill is padding. Each microbatch carries its sequences' lengths as a
zero-padded int32 tensor, so attention stays inside each sequence and one
planned step serves every packing. Targets come from the data source, and the
padding takes ``IGNORE``.

Packing is deterministic: microbatch *i* always holds the same sequences, so
every backend trains on identical data and a resumed run continues exactly
where it stopped. ``documents(i)`` names them, each by its document's ordinal
among the file's documents, so a run can record what it trained on.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable

import numpy as np
import torch

from training.documents import IGNORE, TokenDocuments


def seq_slots(tokens: int, min_tokens_per_seq: int) -> int:
    """How many sequences a microbatch of ``tokens`` tokens has room for -- the
    entries of its lengths tensor -- when its sequences average at least
    ``min_tokens_per_seq`` tokens."""

    return tokens // min_tokens_per_seq


class Packer:
    """Packs a data source's documents into microbatches of ``tokens`` tokens.

    Each microbatch holds at most ``seq_slots`` sequences: its own, and the
    padded tail split into sequences of at most ``max_len``. The best fit looks
    ``window`` unpacked sequences ahead.
    """

    def __init__(
        self, source: TokenDocuments, tokens: int, seq_slots: int, window: int
    ) -> None:
        self.source = source
        self.tokens = tokens
        self.seq_slots = seq_slots
        self.window = window
        # The padded tail may need a few sequences of its own, none over max_len.
        self.max_sequences = seq_slots - -(-tokens // source.max_len)
        self.packs: list[list[int]] = []
        self.pending: deque[int] = deque()
        self.next_document = 0

    def documents(self, index: int) -> list[list[int]]:
        """What microbatch ``index`` holds, one entry per sequence: its
        document's ordinal, its length, and its offset in the document when it
        is a later piece of a spliced one."""

        return [self.source.record(sequence) for sequence in self._pack(index)]

    def microbatch(self, index: int) -> list[torch.Tensor]:
        """[tokens, targets, seq_lens]: tokens and targets [1, T], seq_lens int32."""

        tokens = np.full(self.tokens, self.source.eos, dtype=np.int64)
        targets = np.full(self.tokens, IGNORE, dtype=np.int64)
        seq_lens = np.zeros(self.seq_slots, dtype=np.int32)
        pack = self._pack(index)
        at = 0
        for entry, document in enumerate(pack):
            length = int(self.source.lengths[document])
            tokens[at : at + length] = self.source.tokens(document)
            targets[at : at + length] = self.source.targets(document)
            seq_lens[entry] = length
            at += length
        entry = len(pack)
        while at < self.tokens:  # the padded tail, as sequences of its own
            seq_lens[entry] = min(self.source.max_len, self.tokens - at)
            at += int(seq_lens[entry])
            entry += 1
        return [
            torch.from_numpy(tokens).view(1, -1),
            torch.from_numpy(targets).view(1, -1),
            torch.from_numpy(seq_lens),
        ]

    def trained_tokens(self, indices: Iterable[int]) -> int:
        """How many targets the loss reads over these microbatches."""

        return sum(
            self.source.trained_tokens(document)
            for index in indices
            for document in self._pack(index)
        )

    def stats(self, indices: Iterable[int]) -> dict[str, float]:
        """What these microbatches hold: their sequences, how long those are,
        the share of the microbatches' tokens they fill, and how many targets
        the loss reads."""

        indices = list(indices)
        packs = [self._pack(index) for index in indices]
        counts = [len(pack) for pack in packs]
        lengths = np.array([self.source.lengths[d] for pack in packs for d in pack])
        return {
            "sequences": sum(counts),
            "sequences_per_microbatch_min": min(counts),
            "sequences_per_microbatch_max": max(counts),
            "sequence_length_mean": float(lengths.mean()),
            "sequence_length_median": float(np.median(lengths)),
            "fill": float(lengths.sum()) / (self.tokens * len(packs)),
            "trained_tokens": self.trained_tokens(indices),
        }

    def _pack(self, index: int) -> list[int]:
        while len(self.packs) <= index:
            self.packs.append(self._pack_next())
        return self.packs[index]

    def _pack_next(self) -> list[int]:
        """Best fit over a small look-ahead: while any fits, the microbatch takes
        the longest of the next ``window`` unpacked sequences."""

        lengths = self.source.lengths
        while len(self.pending) < self.window and self.next_document < len(lengths):
            self.pending.append(self.next_document)
            self.next_document += 1
        if not self.pending:
            raise IndexError("ran out of documents; prepare more data")
        room, chosen = self.tokens, []
        while len(chosen) < self.max_sequences:
            fitting = [d for d in self.pending if lengths[d] <= room]
            if not fitting:
                break
            best = max(fitting, key=lambda d: lengths[d])
            self.pending.remove(best)
            chosen.append(best)
            room -= int(lengths[best])
        return chosen


def uniform_microbatch(
    tokens: int, seq_slots: int, length: int, vocab: int
) -> list[torch.Tensor]:
    """A microbatch of sequences of ``length`` tokens: the shape planning
    assumes, with random token ids."""

    seq_lens = torch.zeros(seq_slots, dtype=torch.int32)
    seq_lens[: tokens // length] = length
    ids = torch.randint(vocab, (1, tokens))
    return [ids, torch.roll(ids, -1, 1), seq_lens]
