"""Shared accounting for the binary Datum manifest and owned tensor storage."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .._records import integer

TREE_NODE_BYTES = 64
MAX_TREE_DEPTH = 60
# Batch manifests describe thousands of sequences, unlike control messages.
# Actual sizes still reserve the client's/server's existing buffer budgets.
MAX_INPUT_METADATA_BYTES = 64 << 20
MAX_INPUT_METADATA_NODES = 1 << 20


def metadata_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def manifest_json(datums: Any, tensors: Any) -> bytes:
    return metadata_json(
        {"version": 1, "kind": "datum.tensors", "datums": datums, "tensors": tensors}
    )


@dataclass(frozen=True)
class PayloadSize:
    """Actual or conservative byte counts under the same ownership contract.

    The allowance for Python containers is accounting, not a prediction of RSS.
    Normalization/decoding retains at most one additional tensor at a time.
    """

    tensors: int
    header: int
    tree: int
    largest_tensor: int

    def __post_init__(self) -> None:
        for name in ("tensors", "header", "tree", "largest_tensor"):
            integer(getattr(self, name), name)
        if (
            self.header > MAX_INPUT_METADATA_BYTES
            or self.tree > MAX_INPUT_METADATA_BYTES
        ):
            raise ValueError("input metadata exceeds its bounded byte limit")

    @property
    def retained(self) -> int:
        return self.tensors + 8 * self.header + self.tree

    @property
    def staging(self) -> int:
        return self.retained + self.largest_tensor

    @property
    def wire(self) -> int:
        return 16 + self.header + self.tensors
