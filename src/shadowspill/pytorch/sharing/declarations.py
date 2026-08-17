"""Explicit cross-callable input and output declarations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .paths import PathComponent, PytreePath, normalize_path
from .references import StateRef, TensorRef


class SharedConsistency(StrEnum):
    """Cross-callable value-ordering policy for one plan binding."""

    CAUSAL = "causal"
    UNORDERED = "unordered"


def _normalize_pools(value: str | tuple[str, ...]) -> tuple[str, ...]:
    candidates = (value,) if isinstance(value, str) else tuple(value)
    if not candidates or any(
        not isinstance(pool, str) or not pool for pool in candidates
    ):
        raise ValueError("shared residency requires non-empty pool names")
    if len(candidates) != len(set(candidates)):
        raise ValueError("shared residency pool names must be unique")
    return candidates


@dataclass(frozen=True, slots=True)
class SharedOutput:
    """Retain a declared output pytree leaf in one or more runtime pools."""

    path: PytreePath
    retain_in: tuple[str, ...] | str

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", normalize_path(self.path))
        object.__setattr__(self, "retain_in", _normalize_pools(self.retain_in))


@dataclass(frozen=True, slots=True)
class SharedInput:
    """Bind a runtime-backed reference at one callable input position."""

    reference: TensorRef | StateRef
    require_in: str
    consistency: SharedConsistency = SharedConsistency.CAUSAL

    def __post_init__(self) -> None:
        if not isinstance(self.reference, (TensorRef, StateRef)):
            raise TypeError("shared input reference must be TensorRef or StateRef")
        if not self.require_in:
            raise ValueError("shared input required pool must be non-empty")
        if not isinstance(self.consistency, SharedConsistency):
            raise TypeError("shared input consistency is invalid")


def shared_output(
    *path: PathComponent,
    retain_in: str | tuple[str, ...],
) -> SharedOutput:
    """Construct a shared-output declaration with readable path syntax."""

    return SharedOutput(path=path, retain_in=retain_in)


def shared_input(
    reference: TensorRef | StateRef,
    *,
    require_in: str,
    consistency: SharedConsistency = SharedConsistency.CAUSAL,
) -> SharedInput:
    """Construct a shared-input declaration."""

    return SharedInput(
        reference=reference,
        require_in=require_in,
        consistency=consistency,
    )


__all__ = [
    "SharedConsistency",
    "SharedInput",
    "SharedOutput",
    "shared_input",
    "shared_output",
]
