"""Framework-neutral records consumed by the private runtime adapter."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

INITIAL_PLACEMENT_TASK_ID = 1 << 60


class RuntimePlacementKind(IntEnum):
    """Physical policy for one admitted runtime allocation identity."""

    INITIAL_OBJECT = 0
    TASK_ALLOCATION = 1
    ACTION_DESTINATION = 2
    DYNAMIC_TASK_ALLOCATION = 3
    DYNAMIC_ACTION_DESTINATION = 4


@dataclass(frozen=True, slots=True)
class RuntimeFixedPlacement:
    """One indexed runtime identity and its fixed or dynamic allocation policy."""

    task_id: int
    ordinal: int
    object_id: int
    offset: int
    bytes: int
    alignment: int
    kind: RuntimePlacementKind


@dataclass(frozen=True, slots=True)
class RuntimeFixedDependency:
    """One eviction-completion proof required by a shared fixed range."""

    predecessor_task_id: int
    predecessor_action_ordinal: int
    successor_task_id: int
    successor_ordinal: int
    successor_kind: RuntimePlacementKind


@dataclass(frozen=True, slots=True)
class RuntimeFixedLayout:
    """Indexed, pointer-free fixed-layout certificate passed to the C runtime."""

    slice_bytes: int
    placements: tuple[RuntimeFixedPlacement, ...]
    dependencies: tuple[RuntimeFixedDependency, ...]
    initial_task_id: int


__all__ = [
    "INITIAL_PLACEMENT_TASK_ID",
    "RuntimeFixedDependency",
    "RuntimeFixedLayout",
    "RuntimeFixedPlacement",
    "RuntimePlacementKind",
]
