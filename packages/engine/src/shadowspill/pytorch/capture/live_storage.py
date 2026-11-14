"""Narrow frontend boundary for live PyTorch storage identity.

Only user-owned inputs, parameters, buffers, and their registered views may
use these helpers. Captured task-output semantics must use TaskStorageContract.
"""

from __future__ import annotations

import torch


def live_storage_identity(tensor: torch.Tensor) -> int:
    """Return process-local identity for a live frontend-owned storage."""

    return int(tensor.untyped_storage()._cdata)


def live_storage_bytes(tensor: torch.Tensor) -> int:
    """Return the physical byte extent of a live frontend-owned storage."""

    return int(tensor.untyped_storage().nbytes())


def live_view_key(tensor: torch.Tensor) -> tuple[int, int]:
    """Identify one registered view by storage and element offset."""

    return live_storage_identity(tensor), int(tensor.storage_offset())


def unique_live_tensors(tensors: list[torch.Tensor]) -> tuple[torch.Tensor, ...]:
    """Deduplicate frontend tensor objects without invoking tensor equality."""

    seen: set[int] = set()
    result: list[torch.Tensor] = []
    for tensor in tensors:
        identity = id(tensor)
        if identity in seen:
            continue
        seen.add(identity)
        result.append(tensor)
    return tuple(result)


__all__ = [
    "live_storage_bytes",
    "live_storage_identity",
    "live_view_key",
    "unique_live_tensors",
]
