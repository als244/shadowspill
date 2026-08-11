"""Bounded allocator traces used by isolated PyTorch task profiling."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

from ._abi import AllocationEvent as CAllocationEvent

NO_ID = (1 << 64) - 1


class AllocationTelemetryError(RuntimeError):
    """Raised when the runtime cannot provide a complete allocation trace."""


class AllocationEventKind(IntEnum):
    """Physical allocator transition recorded by the neutral runtime."""

    CREATED = 0
    RELEASED = 1
    PROMOTED = 2


class AllocationCategory(IntEnum):
    """Planner ownership known at an allocation event."""

    ANONYMOUS = 0
    PLANNED_OBJECT = 1
    CALLER_OWNED = 2


@dataclass(frozen=True, slots=True)
class CapturedAllocationEvent:
    """Framework-free value copied from the runtime's bounded event buffer."""

    sequence: int
    task_id: int | None
    allocation_id: int
    generation: int
    requested_bytes: int
    charged_bytes: int
    slab_offset: int
    kind: AllocationEventKind
    category: AllocationCategory


@dataclass(frozen=True, slots=True)
class TaskWorkspaceProfile:
    """Exact anonymous live-set peak for one isolated task invocation."""

    task_id: int
    peak_requested_bytes: int
    peak_charged_bytes: int
    peak_extent_bytes: tuple[int, ...]
    promoted_allocation_ids: tuple[int, ...]
    output_allocation_ids: tuple[int, ...]
    events: tuple[CapturedAllocationEvent, ...]


def start_allocation_telemetry(library: Any, *, capacity: int) -> None:
    """Start a fixed-capacity capture without allocating in callbacks."""

    if capacity <= 0:
        raise ValueError("telemetry capacity must be positive")
    status = int(library.shadowspill_pytorch_allocation_telemetry_start(capacity))
    if status != 0:
        raise AllocationTelemetryError(
            f"allocation telemetry start failed with status {status}"
        )


def stop_allocation_telemetry(library: Any) -> None:
    """Stop capture, preserving its complete event buffer for a later read."""

    status = int(library.shadowspill_pytorch_allocation_telemetry_stop())
    if status != 0:
        raise AllocationTelemetryError(
            f"allocation telemetry stop failed with status {status}"
        )


def read_allocation_telemetry(library: Any) -> tuple[CapturedAllocationEvent, ...]:
    """Copy and validate the complete ordered capture from the C runtime."""

    count = ctypes.c_uint64()
    status = int(
        library.shadowspill_pytorch_allocation_telemetry_read(
            None, 0, ctypes.byref(count)
        )
    )
    if status != 0:
        raise AllocationTelemetryError(
            f"allocation telemetry size query failed with status {status}"
        )
    if count.value == 0:
        return ()
    buffer = (CAllocationEvent * count.value)()
    copied = ctypes.c_uint64()
    status = int(
        library.shadowspill_pytorch_allocation_telemetry_read(
            buffer, count.value, ctypes.byref(copied)
        )
    )
    if status != 0 or copied.value != count.value:
        raise AllocationTelemetryError(
            "allocation telemetry changed or failed during its bounded copy"
        )
    decoded: list[CapturedAllocationEvent] = []
    for expected_sequence, event in enumerate(buffer):
        if event.sequence != expected_sequence:
            raise AllocationTelemetryError(
                "allocation telemetry sequence is not contiguous"
            )
        try:
            kind = AllocationEventKind(event.kind)
            category = AllocationCategory(event.category)
        except ValueError as exc:
            raise AllocationTelemetryError(
                "allocation telemetry contains an unknown enum value"
            ) from exc
        decoded.append(
            CapturedAllocationEvent(
                sequence=event.sequence,
                task_id=None if event.task_id == NO_ID else event.task_id,
                allocation_id=event.allocation_id,
                generation=event.generation,
                requested_bytes=event.requested_bytes,
                charged_bytes=event.charged_bytes,
                slab_offset=event.slab_offset,
                kind=kind,
                category=category,
            )
        )
    return tuple(decoded)


def summarize_task_workspace(
    events: tuple[CapturedAllocationEvent, ...],
    *,
    task_id: int,
    output_allocation_ids: tuple[int, ...] = (),
) -> TaskWorkspaceProfile:
    """Replay task-local anonymous lifetimes; sequential buffers do not add."""

    if task_id < 0:
        raise ValueError("task ID must be non-negative")
    selected = tuple(event for event in events if event.task_id == task_id)
    promoted = {
        event.allocation_id
        for event in selected
        if event.kind is AllocationEventKind.PROMOTED
    }
    outputs = set(output_allocation_ids)
    live: dict[int, tuple[int, int]] = {}
    peak_requested = 0
    peak_charged = 0
    peak_extents: tuple[int, ...] = ()
    for event in selected:
        if event.kind is AllocationEventKind.CREATED:
            if event.allocation_id in promoted or event.allocation_id in outputs:
                continue
            if event.allocation_id in live:
                raise AllocationTelemetryError(
                    f"allocation {event.allocation_id} is created twice"
                )
            live[event.allocation_id] = (
                event.requested_bytes,
                event.charged_bytes,
            )
        elif event.kind is AllocationEventKind.RELEASED:
            live.pop(event.allocation_id, None)
        requested = sum(item[0] for item in live.values())
        charged = sum(item[1] for item in live.values())
        if charged > peak_charged:
            peak_requested = requested
            peak_charged = charged
            peak_extents = tuple(sorted(item[1] for item in live.values()))
    return TaskWorkspaceProfile(
        task_id=task_id,
        peak_requested_bytes=peak_requested,
        peak_charged_bytes=peak_charged,
        peak_extent_bytes=peak_extents,
        promoted_allocation_ids=tuple(sorted(promoted)),
        output_allocation_ids=tuple(sorted(outputs)),
        events=selected,
    )
