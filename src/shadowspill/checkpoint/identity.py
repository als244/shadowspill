"""What a checkpoint binds to: the training state's tensors, not the Program."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

from shadowspill.artifacts._values import digest

from .tensors import geometry

if TYPE_CHECKING:
    from shadowspill.execution import Executor

#: What the manifest records about a tensor beyond its geometry.
ARCHIVE_KEYS = frozenset({"member", "bytes", "sha256"})


def inventory(executor: Executor) -> dict[str, dict[str, Any]]:
    """Every persistent tensor the executor owns, by name: its geometry and the
    region this rank owns, as the manifest records a tensor."""
    result: dict[str, dict[str, Any]] = {}
    for chunk in executor.tensor_chunks():
        if chunk.name in result:
            raise ValueError(f"duplicate tensor: {chunk.name}")
        result[chunk.name] = geometry(chunk)
    return result


def state_identity(executor: Executor) -> str:
    """The digest of the inventory: the same for every Program that trains the
    same state, whatever its kernels, microbatch geometry or budgets."""
    return digest(inventory(executor))


def describe_difference(
    saved: Iterable[Mapping[str, Any]],
    current: Mapping[str, Mapping[str, Any]],
    *,
    omitted: Iterable[str] = (),
) -> str:
    """Which tensors a checkpoint's manifest and this executor disagree on."""
    recorded = {
        entry["name"]: {k: v for k, v in entry.items() if k not in ARCHIVE_KEYS}
        for entry in saved
    }
    names = set(recorded) | set(omitted)
    parts = []
    for label, missing in (
        ("only in the checkpoint", sorted(names - set(current))),
        ("only in this executor", sorted(set(current) - names)),
        (
            "geometry differs",
            sorted(
                name
                for name in recorded.keys() & current.keys()
                if recorded[name] != current[name]
            ),
        ),
    ):
        if missing:
            shown = ", ".join(missing[:5]) + (" ..." if len(missing) > 5 else "")
            parts.append(f"{label}: {shown}")
    return "; ".join(parts) or "the same tensors in another order"
