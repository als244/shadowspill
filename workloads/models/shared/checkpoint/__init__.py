"""Local checkpoint readers and tensor layouts shared by model families."""

from .copy import copy_into
from .reader import Checkpoint
from .sources import (
    Source,
    bank,
    concatenated,
    deinterleave_heads,
    deinterleaved,
    permuted,
    tensor,
)


def layer_name(name: str) -> tuple[int, str] | None:
    """``(layer, rest)`` for a block parameter ``blocks.N.rest``."""
    parts = name.split(".", 2)
    if len(parts) == 3 and parts[0] == "blocks" and parts[1].isdigit():
        return int(parts[1]), parts[2]
    return None


__all__ = [
    "Checkpoint",
    "Source",
    "bank",
    "concatenated",
    "copy_into",
    "deinterleave_heads",
    "deinterleaved",
    "layer_name",
    "permuted",
    "tensor",
]
