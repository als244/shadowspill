"""Size and collect the optional execution trace without changing its schedule."""

from __future__ import annotations

import ctypes
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from shadowspill.ir import MemoryAction, MemoryActionKind, TaskSpec
from shadowspill.runtime import trace
from shadowspill.runtime.abi import AdapterStatistics
from shadowspill.runtime.timing import Marker
from shadowspill.runtime.trace import CapturedRuntimeTrace
from shadowspill.task.allocations.events import TaskAllocationOperation

from .common import TaskMemoryEnvelope

if TYPE_CHECKING:
    from .bridge import RuntimeBridge

# Leave room for records outside a task's profiled allocation contract.
_TRACE_RECORD_FLOOR = 1_000_000


@dataclass(frozen=True, slots=True)
class TraceCapacities:
    """How many records a runtime trace of one invocation is prepared for."""

    events: int
    allocation_events: int


def trace_capacities(
    tasks: Iterable[tuple[TaskSpec, TaskMemoryEnvelope]],
    actions: Iterable[MemoryAction],
    *,
    floor: int = _TRACE_RECORD_FLOOR,
) -> TraceCapacities:
    """Room for every record one invocation of these tasks and actions makes.

    Most of the count is exact: the session opens and closes, each task is
    entered and left, each action is queued, and each copy reserves its
    destination, is dispatched and completes. The rest is bounded from the
    plan: a task waits at most once for each distinct object it reads or
    updates; each allocation in its profiled contract may wait for a range,
    which is two records, and retires once; a release or an eviction retires
    the lease it frees. In the allocator's ledger a lease changes state at
    most four times. A trace that fills anyway says so, in its overflow flags,
    rather than failing the step. Neither capacity falls below `floor`.
    """

    # Session begin and end, and a latched failure.
    events = 3
    leases = 0
    for task, envelope in tasks:
        contract = envelope.allocation_contract
        allocations = (
            0
            if contract is None
            else sum(
                step.operation is TaskAllocationOperation.ALLOCATE
                for step in contract.steps
            )
        )
        objects = set(task.inputs) | {item.object_id for item in task.mutations}
        events += 2 + len(objects) + 3 * allocations
        leases += allocations
    for action in actions:
        events += 1
        if action.kind is not MemoryActionKind.RELEASE:
            events += 3
        if action.kind in (MemoryActionKind.RELEASE, MemoryActionKind.EVICT):
            events += 1
        leases += 1
    return TraceCapacities(
        events=max(floor, events), allocation_events=max(floor, 4 * leases)
    )


def statistics(bridge: RuntimeBridge) -> AdapterStatistics:
    result = AdapterStatistics()
    bridge.require(
        bridge.library.shadowspill_pytorch_allocator_statistics(ctypes.byref(result)),
        "read runtime statistics",
    )
    return result


def prepare_runtime_trace(
    bridge: RuntimeBridge, *, event_capacity: int, allocation_event_capacity: int
) -> None:
    """Allocate reusable bounded CPU trace buffers without enabling trace."""

    trace.prepare_runtime_trace(
        bridge.runtime._runtime_handle,
        event_capacity=event_capacity,
        allocation_event_capacity=allocation_event_capacity,
    )


def begin_runtime_trace(
    bridge: RuntimeBridge, *, step_id: int, origin: Marker | None = None
) -> None:
    trace.begin_runtime_trace(
        bridge.runtime._runtime_handle, step_id=step_id, origin=origin
    )


def end_and_read_runtime_trace(bridge: RuntimeBridge) -> CapturedRuntimeTrace:
    trace.end_runtime_trace(bridge.runtime._runtime_handle)
    return trace.read_runtime_trace(bridge.runtime._runtime_handle)
