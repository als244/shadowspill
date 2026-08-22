"""Typed Python projection of the neutral runtime's bounded trace session."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

from shadowspill.pytorch.runtime_adapter.abi import (
    TRACE_ABI_VERSION,
    AllocationEvent,
    TraceConfig,
    TraceEvent,
    TraceSummary,
)
from shadowspill.pytorch.runtime_adapter.telemetry import (
    NO_ID,
    CapturedAllocationEvent,
    decode_allocation_events,
)


class RuntimeTraceEventKind(IntEnum):
    SESSION_BEGIN = 0
    SESSION_END = 1
    BEFORE_TASK = 2
    AFTER_TASK = 3
    READINESS_WAIT = 4
    ACTION_QUEUED = 5
    DESTINATION_RESERVED = 6
    TRANSFER_DISPATCHED = 7
    TRANSFER_COMPLETED = 8
    ALLOCATION_WAIT_BEGIN = 9
    ALLOCATION_WAIT_END = 10
    RETIREMENT_COMPLETED = 11
    FAILURE_LATCHED = 12


_ACTION_NAMES = {0: "release", 1: "offload", 2: "prefetch"}
_DIRECTION_NAMES = {0: "fetch", 1: "evict"}


@dataclass(frozen=True, slots=True)
class RuntimeTraceEvent:
    """One host-clock event emitted by the framework-neutral runtime."""

    sequence: int
    timestamp_ns: int
    step_id: int
    task_id: int | None
    object_id: int | None
    allocation_id: int | None
    bytes: int
    kind: RuntimeTraceEventKind
    detail_0: int
    detail_1: int

    def details(self) -> dict[str, object]:
        if self.kind is RuntimeTraceEventKind.BEFORE_TASK:
            return {"input_count": self.detail_0, "queued_actions": self.detail_1}
        if self.kind is RuntimeTraceEventKind.AFTER_TASK:
            return {"status": self.detail_0, "action_count": self.detail_1}
        if self.kind is RuntimeTraceEventKind.READINESS_WAIT:
            return {
                "wait_type": "stream_event" if self.detail_0 else "thread_condition",
                "queue_or_wait_count": self.detail_1,
            }
        if self.kind in {
            RuntimeTraceEventKind.ACTION_QUEUED,
            RuntimeTraceEventKind.DESTINATION_RESERVED,
        }:
            result: dict[str, object] = {
                "action": _ACTION_NAMES.get(self.detail_0, f"unknown_{self.detail_0}")
            }
            result[
                "slab_offset"
                if self.kind is RuntimeTraceEventKind.DESTINATION_RESERVED
                else "queued_actions"
            ] = self.detail_1
            return result
        if self.kind in {
            RuntimeTraceEventKind.TRANSFER_DISPATCHED,
            RuntimeTraceEventKind.TRANSFER_COMPLETED,
        }:
            return {
                "direction": _DIRECTION_NAMES.get(
                    self.detail_0, f"unknown_{self.detail_0}"
                ),
                "queued_actions": self.detail_1,
            }
        if self.kind in {
            RuntimeTraceEventKind.ALLOCATION_WAIT_BEGIN,
            RuntimeTraceEventKind.ALLOCATION_WAIT_END,
        }:
            return {
                "free_bytes": self.detail_0,
                "largest_free_range_bytes": self.detail_1,
            }
        if self.kind is RuntimeTraceEventKind.RETIREMENT_COMPLETED:
            return {"slab_offset": self.detail_0, "charged_bytes": self.detail_1}
        if self.kind is RuntimeTraceEventKind.FAILURE_LATCHED:
            return {"status": self.detail_0, "free_bytes": self.detail_1}
        return {}

    def as_dict(self) -> dict[str, object]:
        return {
            "sequence": self.sequence,
            "timestamp_ns": self.timestamp_ns,
            "step_id": self.step_id,
            "task_id": self.task_id,
            "object_id": self.object_id,
            "allocation_id": self.allocation_id,
            "bytes": self.bytes,
            "kind": self.kind.name.lower(),
            "details": self.details(),
        }


@dataclass(frozen=True, slots=True)
class CapturedRuntimeTrace:
    """Complete bounded runtime trace copied out after a step becomes idle."""

    step_id: int
    begin_timestamp_ns: int
    end_timestamp_ns: int
    event_capacity: int
    allocation_event_capacity: int
    event_overflow: bool
    allocation_event_overflow: bool
    events: tuple[RuntimeTraceEvent, ...]
    allocation_events: tuple[CapturedAllocationEvent, ...]


def prepare_runtime_trace(
    library: Any, *, event_capacity: int, allocation_event_capacity: int
) -> None:
    if event_capacity <= 0 or allocation_event_capacity <= 0:
        raise ValueError("trace capacities must be positive")
    config = TraceConfig(
        abi_version=TRACE_ABI_VERSION,
        event_capacity=event_capacity,
        allocation_event_capacity=allocation_event_capacity,
    )
    status = int(library.shadowspill_pytorch_trace_prepare(ctypes.byref(config)))
    if status != 0:
        raise RuntimeError(f"runtime trace preparation failed with status {status}")


def begin_runtime_trace(library: Any, *, step_id: int) -> None:
    status = int(library.shadowspill_pytorch_trace_begin(step_id))
    if status != 0:
        raise RuntimeError(f"runtime trace begin failed with status {status}")


def end_runtime_trace(library: Any) -> None:
    status = int(library.shadowspill_pytorch_trace_end())
    if status != 0:
        raise RuntimeError(f"runtime trace end failed with status {status}")


def read_runtime_trace(library: Any) -> CapturedRuntimeTrace:
    summary = TraceSummary()
    status = int(
        library.shadowspill_pytorch_trace_read(ctypes.byref(summary), None, 0, None, 0)
    )
    if status != 0 or int(summary.abi_version) != TRACE_ABI_VERSION:
        raise RuntimeError(f"runtime trace query failed with status {status}")
    event_buffer = (TraceEvent * int(summary.event_count))()
    allocation_buffer = (AllocationEvent * int(summary.allocation_event_count))()
    copied = TraceSummary()
    status = int(
        library.shadowspill_pytorch_trace_read(
            ctypes.byref(copied),
            event_buffer if summary.event_count else None,
            int(summary.event_count),
            allocation_buffer if summary.allocation_event_count else None,
            int(summary.allocation_event_count),
        )
    )
    if status != 0 or (
        copied.event_count != summary.event_count
        or copied.allocation_event_count != summary.allocation_event_count
    ):
        raise RuntimeError("runtime trace changed or failed during bounded copy")
    decoded: list[RuntimeTraceEvent] = []
    for expected_sequence, event in enumerate(event_buffer):
        if int(event.sequence) != expected_sequence:
            raise RuntimeError("runtime trace sequence is not contiguous")
        try:
            kind = RuntimeTraceEventKind(int(event.kind))
        except ValueError as exc:
            raise RuntimeError("runtime trace contains an unknown event kind") from exc
        decoded.append(
            RuntimeTraceEvent(
                sequence=int(event.sequence),
                timestamp_ns=int(event.timestamp_ns),
                step_id=int(event.step_id),
                task_id=None if int(event.task_id) == NO_ID else int(event.task_id),
                object_id=(
                    None if int(event.object_id) == NO_ID else int(event.object_id)
                ),
                allocation_id=(
                    None
                    if int(event.allocation_id) == NO_ID
                    else int(event.allocation_id)
                ),
                bytes=int(event.bytes),
                kind=kind,
                detail_0=int(event.detail_0),
                detail_1=int(event.detail_1),
            )
        )
    return CapturedRuntimeTrace(
        step_id=int(copied.step_id),
        begin_timestamp_ns=int(copied.begin_timestamp_ns),
        end_timestamp_ns=int(copied.end_timestamp_ns),
        event_capacity=int(copied.event_capacity),
        allocation_event_capacity=int(copied.allocation_event_capacity),
        event_overflow=bool(copied.event_overflow),
        allocation_event_overflow=bool(copied.allocation_event_overflow),
        events=tuple(decoded),
        allocation_events=decode_allocation_events(allocation_buffer),
    )
