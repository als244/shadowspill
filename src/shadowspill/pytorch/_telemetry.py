"""Bounded allocator traces used by isolated PyTorch task profiling."""

from __future__ import annotations

import ctypes
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

from ._abi import AllocationEvent as CAllocationEvent
from .profiling import (
    TaskAllocationEvent,
    TaskAllocationOperation,
    TaskOutputInputBinding,
)

NO_ID = (1 << 64) - 1


class AllocationTelemetryError(RuntimeError):
    """Raised when the runtime cannot provide a complete allocation trace."""


class AllocationEventKind(IntEnum):
    """Physical allocator transition recorded by the neutral runtime."""

    CREATED = 0
    RELEASED = 1
    PROMOTED = 2
    LOGICAL_FREED = 3


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
    allocation_trace: tuple[TaskAllocationEvent, ...]
    output_input_bindings: tuple[TaskOutputInputBinding, ...] = ()
    persistent_allocation_ids: tuple[int, ...] = ()
    persistent_extent_bytes: tuple[int, ...] = ()


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
    return decode_allocation_events(buffer)


def decode_allocation_events(
    events: Iterable[CAllocationEvent],
) -> tuple[CapturedAllocationEvent, ...]:
    """Decode one caller-owned sequence of native allocation records."""

    decoded: list[CapturedAllocationEvent] = []
    for expected_sequence, event in enumerate(events):
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
    output_allocation_views: Mapping[int, tuple[tuple[int, int], ...]] | None = None,
    output_input_bindings: tuple[TaskOutputInputBinding, ...] = (),
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
    output_views = dict(output_allocation_views or {})
    outputs = set(output_views)
    allocation_trace, persistent_ids = _normalize_task_allocation_trace(
        selected,
        output_views=output_views,
        retained_allocation_ids=promoted,
    )
    persistent = set(persistent_ids)
    persistent_extents = tuple(
        event.charged_bytes
        for event in selected
        if event.kind is AllocationEventKind.CREATED
        and event.allocation_id in persistent
    )
    live: dict[int, tuple[int, int]] = {}
    peak_requested = 0
    peak_charged = 0
    peak_extents: tuple[int, ...] = ()
    for event in selected:
        if event.kind is AllocationEventKind.CREATED:
            if event.allocation_id in promoted or event.allocation_id in outputs:
                continue
            if event.allocation_id in persistent:
                continue
            if event.allocation_id in live:
                raise AllocationTelemetryError(
                    f"allocation {event.allocation_id} is created twice"
                )
            live[event.allocation_id] = (
                event.requested_bytes,
                event.charged_bytes,
            )
        elif event.kind is AllocationEventKind.LOGICAL_FREED:
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
        allocation_trace=allocation_trace,
        output_input_bindings=output_input_bindings,
        persistent_allocation_ids=persistent_ids,
        persistent_extent_bytes=persistent_extents,
    )


def _normalize_task_allocation_trace(
    events: tuple[CapturedAllocationEvent, ...],
    *,
    output_views: Mapping[int, tuple[tuple[int, int], ...]],
    retained_allocation_ids: set[int],
) -> tuple[tuple[TaskAllocationEvent, ...], tuple[int, ...]]:
    """Replace process allocation IDs with stable task-local ordinals."""

    ordinal_by_id: dict[int, int] = {}
    sizes_by_id: dict[int, tuple[int, int]] = {}
    pending_by_span: dict[tuple[int, int], tuple[int, int]] = {}
    pending_span_by_id: dict[int, tuple[int, int]] = {}
    trace: list[TaskAllocationEvent] = []
    for event in events:
        if event.kind is AllocationEventKind.CREATED:
            if event.allocation_id in ordinal_by_id:
                raise AllocationTelemetryError(
                    f"allocation {event.allocation_id} is created twice"
                )
            ordinal = len(ordinal_by_id)
            pending = pending_by_span.pop(
                (event.slab_offset, event.charged_bytes), None
            )
            if pending is not None:
                pending_span_by_id.pop(pending[0], None)
            ordinal_by_id[event.allocation_id] = ordinal
            sizes_by_id[event.allocation_id] = (
                event.requested_bytes,
                event.charged_bytes,
            )
            trace.append(
                TaskAllocationEvent(
                    ordinal,
                    TaskAllocationOperation.ALLOCATE,
                    event.requested_bytes,
                    event.charged_bytes,
                    tuple(
                        leaf_index
                        for leaf_index, _offset in output_views.get(
                            event.allocation_id, ()
                        )
                    ),
                    tuple(
                        offset
                        for _leaf_index, offset in output_views.get(
                            event.allocation_id, ()
                        )
                    ),
                    None if pending is None else pending[1],
                )
            )
        elif event.kind is AllocationEventKind.LOGICAL_FREED:
            freed_ordinal = ordinal_by_id.get(event.allocation_id)
            if freed_ordinal is None:
                continue
            if event.allocation_id in output_views:
                continue
            requested, charged = sizes_by_id[event.allocation_id]
            trace.append(
                TaskAllocationEvent(
                    freed_ordinal,
                    TaskAllocationOperation.FREE,
                    requested,
                    charged,
                )
            )
            pending_by_span[(event.slab_offset, event.charged_bytes)] = (
                event.allocation_id,
                freed_ordinal,
            )
            pending_span_by_id[event.allocation_id] = (
                event.slab_offset,
                event.charged_bytes,
            )
        elif event.kind is AllocationEventKind.RELEASED:
            original_span = pending_span_by_id.pop(event.allocation_id, None)
            if original_span is not None:
                pending = pending_by_span.get(original_span)
                if pending is not None and pending[0] == event.allocation_id:
                    del pending_by_span[original_span]
    live = {
        event.allocation_ordinal
        for event in trace
        if event.operation is TaskAllocationOperation.ALLOCATE
    }
    live.difference_update(
        event.allocation_ordinal
        for event in trace
        if event.operation is TaskAllocationOperation.FREE
    )
    output_ordinals = {
        event.allocation_ordinal for event in trace if event.output_leaf_indices
    }
    retained_ordinals = {
        ordinal_by_id[allocation_id]
        for allocation_id in retained_allocation_ids
        if allocation_id in ordinal_by_id
    }
    unexpected = live - output_ordinals - retained_ordinals
    allocation_id_by_ordinal = {
        ordinal: allocation_id for allocation_id, ordinal in ordinal_by_id.items()
    }
    persistent_ids = tuple(
        allocation_id_by_ordinal[ordinal] for ordinal in sorted(unexpected)
    )
    filtered = tuple(
        event for event in trace if event.allocation_ordinal not in unexpected
    )
    return filtered, persistent_ids
