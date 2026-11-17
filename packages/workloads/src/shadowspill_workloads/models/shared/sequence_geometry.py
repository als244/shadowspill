"""Static preparation bounds, separate from runtime packed sequence lengths."""

from contextlib import contextmanager
from contextvars import ContextVar

import torch

_sequence_length_bound: ContextVar[int | None] = ContextVar(
    "sequence_length_bound", default=None
)


@contextmanager
def sequence_length_bound(maximum: int | None):
    """Use the declared per-sequence bound while capturing a prepared graph.

    This affects kernel launch geometry only. Actual lengths and offsets remain
    input tensors. The build config records this bound in artifact identity.
    """
    token = _sequence_length_bound.set(maximum)
    try:
        yield
    finally:
        _sequence_length_bound.reset(token)


@torch.compiler.assume_constant_result
def bounded_sequence_length(capacity: int) -> int:
    maximum = _sequence_length_bound.get()
    return capacity if maximum is None else min(capacity, maximum)
