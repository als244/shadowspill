"""What the bridge can say about a plan: statistics, object states, occupancy,
profiler ranges and the runtime trace.

Nothing here changes runtime state. The occupancy description is what a
refused fixed layout carries, so a reader sees which ranges stood in the way
and what holds them; the object states are what a failed task boundary names.
"""

from __future__ import annotations

import ctypes
import dataclasses
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

from shadowspill.runtime import failures, trace
from shadowspill.runtime.abi import AdapterStatistics, ObjectSnapshot
from shadowspill.runtime.occupancy import allocation_id_for_address, live_allocations
from shadowspill.runtime.timing import Marker
from shadowspill.runtime.trace import CapturedRuntimeTrace

if TYPE_CHECKING:
    from . import RuntimeBridge

#: How many allocations a layout refusal lists before it says how many more.
#: Enough to show a seam, short enough that the message stays readable.
_REPORTED_ALLOCATIONS = 24

#: The runtime's object residency, named. Mirrors ShadowSpillObjectResidency.
_OBJECT_RESIDENCY_NAMES: dict[int, str] = {
    0: "spill_only",
    1: "execution_ready",
    2: "fetching",
    3: "evicting",
    4: "released",
}


def _describe_occupants(
    holders: tuple[Any, ...], holding: Mapping[int, tuple[str, ...]]
) -> str:
    """Name what holds a range, and what in turn keeps that holder alive.

    The holder says what the range is in the framework's terms; the retainer
    says where the reference lives, which is the only place it can be dropped.
    """

    if not holders:
        return ", held by no frontend object"
    described = []
    for item in holders[:3]:
        kept = holding.get(id(item), ())
        where = f" via {', '.join(kept[:2])}" if kept else " retained by a library"
        described.append(f"{type(item).__name__}{tuple(item.shape)}{where}")
    more = f" +{len(holders) - 3}" if len(holders) > 3 else ""
    return f", held by {'; '.join(described)}{more}"


def describe_pool_occupants(bridge: RuntimeBridge) -> str:
    """What is occupying the pool, for a layout that could not be placed.

    A refusal for want of a contiguous range is not explained by how many
    allocations are live: a small one in the wrong place costs the largest
    free range and leaves the free total nearly untouched. The offsets are
    the diagnosis, so they are listed in pool order with the scope that made
    each one.
    """

    try:
        held = live_allocations(bridge.runtime)
    except Exception:  # a failure report must not fail
        return ""
    if not held:
        return ""
    shown = held[:_REPORTED_ALLOCATIONS]
    try:
        # Everything in this pool arrived through the framework's allocator,
        # so a range with no frontend object is held by the framework's own
        # internals rather than by anything a caller can drop. Saying which
        # is what separates a reference to release from one to relocate.
        frontend = bridge.runtime.frontend
        holders = frontend.occupants(
            shown,
            lambda address: allocation_id_for_address(bridge.runtime, address),
        )
        occupying = [item for objects in holders.values() for item in objects]
        holding = frontend.retainers(
            occupying, ignore=(holders, occupying, *holders.values())
        )
    except Exception:
        holders = {}
        holding = {}
    lines = "".join(
        f"\n  offset={item.offset} bytes={item.charged_bytes}"
        f" {bridge.objects.role_of(item)} from {item.origin}"
        f"{' scratch' if item.scratch else ''}"
        f"{' planned' if item.plan_owned else ''}"
        f"{' freed-pending-retirement' if item.logical_freed else ''}"
        f"{_describe_occupants(holders.get(item.allocation_id, ()), holding)}"
        for item in shown
    )
    omitted = len(held) - len(shown)
    tail = f"\n  ... {omitted} more" if omitted > 0 else ""
    return f"; the pool is held by:{lines}{tail}"


def statistics(bridge: RuntimeBridge) -> AdapterStatistics:
    result = AdapterStatistics()
    bridge.require(
        bridge.library.shadowspill_pytorch_allocator_statistics(ctypes.byref(result)),
        "read runtime statistics",
    )
    return result


def describe_object_state(bridge: RuntimeBridge, object_id: int) -> str | None:
    """Where a value is and whether each copy is current, as the runtime sees it.

    A refusal names an object; what decides the refusal is the object's state, so
    a report that gives one without the other cannot be read.
    """

    snapshot = ObjectSnapshot()
    status = int(
        bridge.runtime_library.shadowspill_object_snapshot(
            bridge.runtime._runtime_handle, object_id, ctypes.byref(snapshot)
        )
    )
    if status != 0:
        return None
    residency = _OBJECT_RESIDENCY_NAMES.get(
        int(snapshot.residency), f"unknown_{int(snapshot.residency)}"
    )
    return (
        f"{residency}, spill_current={int(snapshot.spill_current)}, "
        f"generation={int(snapshot.generation)}, "
        f"execution_pointer={int(snapshot.execution_pointer or 0)}"
    )


def describe_refused_action(
    bridge: RuntimeBridge,
    diagnostics: failures.RuntimeFailureDiagnostics,
    actions: Iterable[Any],
) -> failures.RuntimeFailureDiagnostics:
    """Name the action a refusal was about, and the state that refused it.

    The runtime refuses an action because of the state of the object it names,
    and it reports that object as a runtime identifier. On its own that has to
    be decoded by hand against the plan, so this resolves the identifier to its
    alias, finds the action among `actions` that named it, and reads back the
    residency the refusal turned on.

    `actions` is whatever the caller was submitting when it was refused. An
    object that no action among them names is reported as such, because that is
    itself the answer: the refusal was not about the batch it arrived with.
    """

    if diagnostics.object_id is None:
        return dataclasses.replace(diagnostics)
    alias_id = bridge.objects.alias_for_runtime_object(diagnostics.object_id)
    refused: str | None = None
    if alias_id is not None:
        for action in actions:
            if action.alias_group_id == alias_id:
                refused = (
                    f"{action.kind.name} {alias_id} "
                    f"(trigger {action.trigger_task_id})"
                )
                break
        if refused is None:
            refused = f"no action on {alias_id} in this batch"
    return dataclasses.replace(
        diagnostics,
        refused_action=refused,
        object_state=describe_object_state(bridge, diagnostics.object_id),
    )


def input_failure_states(
    bridge: RuntimeBridge, alias_ids: Iterable[str]
) -> tuple[str, ...]:
    """Describe unavailable inputs without changing runtime state."""

    result: list[str] = []
    for alias_id in dict.fromkeys(alias_ids):
        snapshot = ObjectSnapshot()
        status = int(
            bridge.runtime_library.shadowspill_object_snapshot(
                bridge.runtime._runtime_handle,
                bridge.objects.runtime_object_id(alias_id),
                ctypes.byref(snapshot),
            )
        )
        if status != 0:
            result.append(f"{alias_id}:snapshot_status={status}")
            continue
        allocation_id = int(snapshot.allocation_id)
        pointer = int(snapshot.execution_pointer or 0)
        if int(snapshot.residency) in {1, 2} and pointer != 0:
            continue
        result.append(
            f"{alias_id}:residency={int(snapshot.residency)},"
            f"allocation={allocation_id},generation={int(snapshot.generation)},"
            f"pointer={pointer},spill_current={int(snapshot.spill_current)}"
        )
    return tuple(result)


def profile_range_begin(bridge: RuntimeBridge, name: str) -> int:
    """Open one optional backend-backed profiling range."""

    return int(
        bridge.runtime_library.shadowspill_profiler_range_begin(
            bridge.runtime._runtime_handle, name.encode("utf-8")
        )
    )


def profile_range_end(bridge: RuntimeBridge, range_id: int) -> None:
    """Close a range returned by :meth:`profile_range_begin`."""

    bridge.runtime_library.shadowspill_profiler_range_end(
        bridge.runtime._runtime_handle, range_id
    )


def set_profiler_annotations(bridge: RuntimeBridge, enabled: bool) -> None:
    """Toggle backend annotations without changing runtime tracing."""

    bridge.require(
        bridge.runtime_library.shadowspill_profiler_annotations_set(
            bridge.runtime._runtime_handle, enabled
        ),
        f"{'enable' if enabled else 'disable'} profiler annotations",
    )


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


def raise_if_allocator_failed(bridge: RuntimeBridge, operation: str) -> None:
    """Raise the first callback failure without touching the device timeline."""

    failures.raise_if_allocator_failed(bridge.library, operation)


__all__ = [
    "begin_runtime_trace",
    "describe_object_state",
    "describe_pool_occupants",
    "describe_refused_action",
    "end_and_read_runtime_trace",
    "input_failure_states",
    "prepare_runtime_trace",
    "profile_range_begin",
    "profile_range_end",
    "raise_if_allocator_failed",
    "set_profiler_annotations",
    "statistics",
]
