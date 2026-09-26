"""Documents from a tokenized file: the data source pretraining packs.

``prepare_data`` writes every document's token ids followed by the tokenizer's
end-of-text id, back to back, as uint32. A document is what lies between two
end-of-text ids, the second one included.

The source hands the packer sequences of at most ``max_len`` tokens. A document
that fits is one sequence. A longer one, as ``long_documents`` says, is
dropped; truncated to its first ``max_len`` tokens; or spliced into consecutive
pieces of ``max_len`` tokens, the last piece taking the rest, each a sequence
of its own. A sequence's targets are its tokens' next tokens; at its last
position that is the document's next token when the document goes on past it,
and nothing (``IGNORE``) at the document's end, whose next token is another
document's.

Another kind of training packs another source -- supervised fine-tuning, say,
whose targets ignore the prompt -- with the same packer.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

IGNORE = -100  # a target that neither the loss nor its gradient reads

LONG_DOCUMENTS = ("drop", "truncate", "splice")


class TokenDocuments:
    """The sequences of one token file, in file order: its documents, each of
    at most ``max_len`` tokens, a longer one handled as ``long_documents`` says.

    For every sequence: ``ordinals``, its document's place among the file's
    documents; ``offsets``, where in that document it starts; ``starts`` and
    ``lengths`` in the file; and ``continues``, whether the document goes on
    past its last token.
    """

    def __init__(
        self, path: Path, eos: int, max_len: int, long_documents: str = "drop"
    ) -> None:
        if long_documents not in LONG_DOCUMENTS:
            raise ValueError(
                f"long_documents must be one of {', '.join(LONG_DOCUMENTS)}"
            )
        self.stream = np.memmap(path, dtype=np.uint32, mode="r")
        ends = np.flatnonzero(self.stream == eos) + 1
        starts = np.concatenate(([0], ends[:-1]))
        lengths = ends - starts
        if long_documents == "drop":
            documents = np.flatnonzero(lengths <= max_len)
        else:
            documents = np.arange(len(lengths))
        if long_documents == "splice":
            pieces = -(-lengths[documents] // max_len)
        else:
            pieces = np.ones(len(documents), dtype=np.int64)
        self.ordinals = np.repeat(documents, pieces)
        first = np.repeat(np.cumsum(pieces) - pieces, pieces)
        self.offsets = (np.arange(len(self.ordinals)) - first) * max_len
        remaining = lengths[self.ordinals] - self.offsets
        self.starts = starts[self.ordinals] + self.offsets
        self.lengths = np.minimum(remaining, max_len)
        self.continues = remaining > max_len
        self.eos = eos
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.lengths)

    def tokens(self, sequence: int) -> np.ndarray:
        start = int(self.starts[sequence])
        return self.stream[start : start + int(self.lengths[sequence])]

    def targets(self, sequence: int) -> np.ndarray:
        """Each position's next token: at the last position, the document's
        next token if it continues, and nothing at the document's end."""

        start, length = int(self.starts[sequence]), int(self.lengths[sequence])
        if self.continues[sequence]:
            return self.stream[start + 1 : start + length + 1].astype(np.int64)
        return np.append(
            self.stream[start + 1 : start + length].astype(np.int64), IGNORE
        )

    def trained_tokens(self, sequence: int) -> int:
        """How many of the sequence's targets the loss reads."""

        length = int(self.lengths[sequence])
        return length if self.continues[sequence] else length - 1

    def record(self, sequence: int) -> list[int]:
        """``[ordinal, length]``, and the offset in the document for a piece
        that does not start it: enough to rebuild the sequence from the file."""

        entry = [int(self.ordinals[sequence]), int(self.lengths[sequence])]
        offset = int(self.offsets[sequence])
        return entry if offset == 0 else [*entry, offset]
