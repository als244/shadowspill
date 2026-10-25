"""Admission facts in indexed form, and a schedule encoded against them.

The pool topology a plan must fit is indexed once and reused across
every schedule measured against it, because it does not change when a
schedule does.
"""

from __future__ import annotations

import ctypes
from array import array
from dataclasses import dataclass
from itertools import repeat
from typing import cast

import numpy as np

from shadowspill.ir import (
    IndexedMemorySchedule,
    MemoryActionKind,
    MemoryLocation,
    MemorySchedule,
)
from shadowspill.simulator import (
    ActionPhysicalDelta,
    MemoryReuseDependency,
    SimulationAdmission,
    TaskPhysicalDelta,
)
from shadowspill.simulator.capi import NO_INDEX
from shadowspill.simulator.indexing import IndexedSimulationTemplate
from shadowspill.status import ABI_VERSION, Status

from ..capi import (
    CAdmissionFacts,
    CIndexedSchedule,
    CScheduleAdmissionResult,
    planner_api,
)
from . import AdmissionFacts
from .task_columns import ALLOCATE, TaskColumns


def _u32(values: tuple[int, ...]) -> ctypes.Array[ctypes.c_uint32]:
    result_type = ctypes.c_uint32 * max(1, len(values))
    return result_type.from_buffer_copy(array("I", values)) if values else result_type()


def _u8(values: tuple[int, ...]) -> ctypes.Array[ctypes.c_uint8]:
    result_type = ctypes.c_uint8 * max(1, len(values))
    return result_type.from_buffer_copy(bytes(values)) if values else result_type()


@dataclass(frozen=True, slots=True)
class IndexedAdmissionFacts:
    """Borrowed C facts plus Python owners for all indexed arrays."""

    value: CAdmissionFacts
    buffers: tuple[object, ...]
    digest: str


@dataclass(frozen=True, slots=True)
class CompiledScheduleAdmission:
    """Exact physical projection for one selected indexed schedule."""

    simulation_admission: SimulationAdmission
    decision_digest: int
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    peak_fragmentation_bytes: int


_ACTION_KIND = {
    MemoryActionKind.RELEASE: 0,
    MemoryActionKind.EVICT: 1,
    MemoryActionKind.FETCH: 2,
    MemoryActionKind.WRITE_BACK: 3,
}
_LOCATION = {MemoryLocation.DEVICE: 0, MemoryLocation.SPILL: 1}


def encode_schedule(
    schedule: MemorySchedule,
    simulation: IndexedSimulationTemplate,
) -> IndexedMemorySchedule:
    """Encode one public schedule against an immutable facts."""

    return IndexedMemorySchedule(
        action_trigger_tasks=tuple(
            simulation.task_index[item.trigger_task_id] for item in schedule.actions
        ),
        action_aliases=tuple(
            simulation.alias_index[item.alias_group_id] for item in schedule.actions
        ),
        action_kinds=tuple(_ACTION_KIND[item.kind] for item in schedule.actions),
        initial_aliases=tuple(
            simulation.alias_index[item.alias_group_id]
            for item in schedule.initial_residency
        ),
        initial_locations=tuple(
            _LOCATION[item.location] for item in schedule.initial_residency
        ),
        final_aliases=tuple(
            simulation.alias_index[item.alias_group_id]
            for item in schedule.final_residency
        ),
        final_locations=tuple(
            _LOCATION[item.location] for item in schedule.final_residency
        ),
    )


def index_admission_facts(
    facts: AdmissionFacts,
    simulation: IndexedSimulationTemplate,
) -> IndexedAdmissionFacts:
    """Project only tasks selected by one resolved program."""

    if facts.device_id not in simulation.device_ids:
        raise ValueError(
            f"admission device {facts.device_id!r} is absent from simulation"
        )
    buffers, allocation_slot_count, task_count = _compiled_tasks(facts, simulation)
    value = CAdmissionFacts(
        abi_version=ABI_VERSION,
        task_count=task_count,
        alias_count=len(simulation.alias_ids),
        pool_capacity_bytes=facts.pool_capacity_bytes,
        object_capacity_bytes=facts.object_capacity_bytes,
        minimum_alignment=facts.minimum_alignment,
        task_workspace_offsets=buffers[0],
        task_workspace_extent_bytes=buffers[1],
        fresh_output_offsets=buffers[2],
        fresh_output_aliases=buffers[3],
        replacement_offsets=buffers[4],
        replacement_aliases=buffers[5],
        handoff_offsets=buffers[6],
        handoff_source_aliases=buffers[7],
        handoff_destination_aliases=buffers[8],
        allocation_slot_count=allocation_slot_count,
        task_allocation_offsets=buffers[9],
        task_allocation_slots=buffers[10],
        task_allocation_bytes=buffers[11],
        task_allocation_aliases=buffers[12],
        task_allocation_kinds=buffers[13],
    )
    return IndexedAdmissionFacts(value, buffers, facts.digest)


def _compiled_tasks(
    facts: AdmissionFacts, simulation: IndexedSimulationTemplate
) -> tuple[tuple[object, ...], int, int]:
    """The per-task arrays of `facts.tasks` in this template's index space.

    They depend on the tasks alone, which every capacity variant of a set of
    facts shares, so they are compiled once per template and the variants
    differ only in the scalars their own structure carries. They are gathered
    from the facts' columns in the template's task order.
    """

    cached = cast(
        tuple[object, tuple[tuple[object, ...], int, int]] | None,
        simulation.derived.get("admission_tasks"),
    )
    if cached is not None and cached[0] is facts.tasks:
        return cached[1]
    columns = facts.task_columns
    position = dict(zip(columns.task_ids, range(len(columns.task_ids)), strict=True))
    try:
        order = np.fromiter(
            map(position.__getitem__, simulation.task_ids),
            dtype=np.int64,
            count=len(simulation.task_ids),
        )
    except KeyError as exc:
        raise ValueError(
            f"admission facts lacks selected task {exc.args[0]!r}"
        ) from exc
    alias_index = simulation.alias_index
    # Where each alias the facts name sits in the template; an alias only
    # unselected tasks name may be absent, and only a selected one is read.
    aliases = np.fromiter(
        map(alias_index.get, columns.alias_ids, repeat(-1)),
        dtype=np.int64,
        count=len(columns.alias_ids),
    )

    def indexed(positions: np.ndarray) -> np.ndarray:
        found = aliases[positions]
        if (found < 0).any():
            raise KeyError(columns.alias_ids[int(positions[np.argmax(found < 0)])])
        return cast(np.ndarray, found)

    workspace_offsets, workspace = _gather(columns.workspace_offsets, order)
    fresh_offsets, fresh = _gather(columns.fresh_offsets, order)
    replacement_offsets, replacements = _gather(columns.replacement_offsets, order)
    handoff_offsets, handoffs = _gather(columns.handoff_offsets, order)
    allocation_offsets, steps = _gather(columns.step_offsets, order)
    slots, slot_count = _allocation_slots(columns, allocation_offsets, steps)
    outputs = columns.step_outputs[steps]
    present = outputs >= 0
    allocation_aliases = np.full(len(steps), NO_INDEX, dtype=np.int64)
    allocation_aliases[present] = indexed(outputs[present])
    buffers = (
        _u32_array(workspace_offsets),
        _u64_array(columns.workspace_extents[workspace]),
        _u32_array(fresh_offsets),
        _u32_array(indexed(columns.fresh_aliases[fresh])),
        _u32_array(replacement_offsets),
        _u32_array(indexed(columns.replacement_aliases[replacements])),
        _u32_array(handoff_offsets),
        _u32_array(indexed(columns.handoff_sources[handoffs])),
        _u32_array(indexed(columns.handoff_destinations[handoffs])),
        _u32_array(allocation_offsets),
        _u32_array(slots),
        _u64_array(columns.step_bytes[steps]),
        _u32_array(allocation_aliases),
        _u8_array(columns.step_kinds[steps]),
    )
    compiled = (buffers, slot_count, len(order))
    simulation.derived["admission_tasks"] = (facts.tasks, compiled)
    return compiled


def _gather(offsets: np.ndarray, order: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A per-task list's offsets in `order`, and where each of its values
    sits in the facts' own order."""

    starts = offsets[order]
    lengths = offsets[order + 1] - starts
    gathered = np.zeros(len(order) + 1, dtype=np.int64)
    np.cumsum(lengths, out=gathered[1:])
    return gathered, np.repeat(starts - gathered[:-1], lengths) + np.arange(
        gathered[-1], dtype=np.int64
    )


def _allocation_slots(
    columns: TaskColumns, offsets: np.ndarray, steps: np.ndarray
) -> tuple[np.ndarray, int]:
    """The lease slot of every allocation step, in compiled order.

    An allocation that reuses no earlier one takes the next new slot, in step
    order; an allocation that reuses an ordinal takes the slot that ordinal's
    allocation took, and a release the slot of the allocation it ends. A
    task's allocations name distinct ordinals, so each step names one
    allocation of its own task, and following those names reaches a new slot.
    """

    count = len(steps)
    if count == 0:
        return np.zeros(0, dtype=np.int64), 0
    task = np.repeat(np.arange(len(offsets) - 1, dtype=np.int64), np.diff(offsets))
    ordinals = columns.step_ordinals[steps]
    reuses = columns.step_reuses[steps]
    allocates = columns.step_kinds[steps] == ALLOCATE
    fresh = allocates & (reuses < 0)
    slot = np.full(count, -1, dtype=np.int64)
    slot_count = int(fresh.sum())
    slot[fresh] = np.arange(slot_count, dtype=np.int64)

    # Each allocation keyed by its task and ordinal, to find the allocation a
    # step names.
    span = int(max(ordinals.max(), reuses.max())) + 1
    allocations = np.flatnonzero(allocates)
    keys = task[allocations] * span + ordinals[allocations]
    by_key = np.argsort(keys, kind="stable")
    sorted_keys = keys[by_key]

    def allocation_of(where: np.ndarray, named: np.ndarray) -> np.ndarray:
        wanted = task[where] * span + named
        at = np.minimum(np.searchsorted(sorted_keys, wanted), len(sorted_keys) - 1)
        if len(at) and not np.array_equal(sorted_keys[at], wanted):
            raise ValueError(
                "an allocation step names an ordinal its task never allocated"
            )
        return cast(np.ndarray, allocations[by_key[at]])

    source = np.arange(count, dtype=np.int64)
    reusing = np.flatnonzero(allocates & (reuses >= 0))
    source[reusing] = allocation_of(reusing, reuses[reusing])
    releasing = np.flatnonzero(~allocates)
    source[releasing] = allocation_of(releasing, ordinals[releasing])
    while True:
        followed = source[source]
        if np.array_equal(followed, source):
            break
        source = followed
    return slot[source], slot_count


def _u32_array(values: np.ndarray) -> ctypes.Array[ctypes.c_uint32]:
    result_type = ctypes.c_uint32 * max(1, len(values))
    if not len(values):
        return result_type()
    return result_type.from_buffer_copy(np.ascontiguousarray(values, dtype=np.uint32))


def _u64_array(values: np.ndarray) -> ctypes.Array[ctypes.c_uint64]:
    result_type = ctypes.c_uint64 * max(1, len(values))
    if not len(values):
        return result_type()
    return result_type.from_buffer_copy(np.ascontiguousarray(values, dtype=np.uint64))


def _u8_array(values: np.ndarray) -> ctypes.Array[ctypes.c_uint8]:
    result_type = ctypes.c_uint8 * max(1, len(values))
    if not len(values):
        return result_type()
    return result_type.from_buffer_copy(np.ascontiguousarray(values, dtype=np.uint8))


def evaluate_schedule_admission(
    simulation: IndexedSimulationTemplate,
    admission: IndexedAdmissionFacts,
    schedule: IndexedMemorySchedule,
) -> CompiledScheduleAdmission:
    """Evaluate one selected schedule through compiled production-pool policy."""

    action_count = len(schedule.action_kinds)
    if not (
        len(schedule.action_trigger_tasks)
        == len(schedule.action_aliases)
        == action_count
    ):
        raise ValueError("indexed admission schedule action arrays disagree")
    action_tasks = _u32(schedule.action_trigger_tasks)
    action_aliases = _u32(schedule.action_aliases)
    action_kinds = _u8(schedule.action_kinds)
    initial_aliases = _u32(schedule.initial_aliases)
    initial_locations = _u8(schedule.initial_locations)
    final_aliases = _u32(schedule.final_aliases)
    final_locations = _u8(schedule.final_locations)
    indexed = CIndexedSchedule(
        action_count=action_count,
        action_trigger_tasks=action_tasks,
        action_aliases=action_aliases,
        action_kinds=action_kinds,
        initial_count=len(schedule.initial_aliases),
        initial_aliases=initial_aliases,
        initial_locations=initial_locations,
        final_count=len(schedule.final_aliases),
        final_aliases=final_aliases,
        final_locations=final_locations,
    )
    task_count = len(simulation.task_ids)
    task_start = (ctypes.c_int64 * max(1, task_count))()
    task_completion = (ctypes.c_int64 * max(1, task_count))()
    action_trigger = (ctypes.c_int64 * max(1, action_count))()
    action_completion = (ctypes.c_int64 * max(1, action_count))()
    reuse_capacity = action_count
    reuse_predecessors = (ctypes.c_uint32 * max(1, reuse_capacity))()
    reuse_predecessor_tasks = (ctypes.c_uint32 * max(1, reuse_capacity))()
    reuse_tasks = (ctypes.c_uint32 * max(1, reuse_capacity))()
    reuse_actions = (ctypes.c_uint32 * max(1, reuse_capacity))()
    result = CScheduleAdmissionResult(
        task_start_deltas=task_start,
        task_completion_deltas=task_completion,
        task_capacity=task_count,
        action_trigger_deltas=action_trigger,
        action_completion_deltas=action_completion,
        action_capacity=action_count,
        reuse_predecessor_actions=reuse_predecessors,
        reuse_predecessor_tasks=reuse_predecessor_tasks,
        reuse_successor_tasks=reuse_tasks,
        reuse_successor_actions=reuse_actions,
        reuse_capacity=reuse_capacity,
    )
    library = planner_api()
    while True:
        status = int(
            library.shadowspill_evaluate_schedule_admission(
                ctypes.byref(simulation.program),
                ctypes.byref(admission.value),
                ctypes.byref(indexed),
                ctypes.byref(result),
            )
        )
        if (
            status != Status.INVALID_ARGUMENT
            or result.reuse_count <= result.reuse_capacity
        ):
            break
        result.reuse_capacity = result.reuse_count
        arrays = [(ctypes.c_uint32 * result.reuse_capacity)() for _ in range(4)]
        reuse_predecessors, reuse_predecessor_tasks, reuse_tasks, reuse_actions = arrays
        result.reuse_predecessor_actions = reuse_predecessors
        result.reuse_predecessor_tasks = reuse_predecessor_tasks
        result.reuse_successor_tasks = reuse_tasks
        result.reuse_successor_actions = reuse_actions
    if status != 0:
        if status == Status.NO_FEASIBLE_CANDIDATE:
            raise ValueError(
                "selected schedule failed dynamic MemoryPool admission: "
                f"operation={int(result.error_operation_index)}, "
                f"request={int(result.error_requested_bytes)}, "
                f"free={int(result.error_free_bytes)}, "
                "largest_free_range="
                f"{int(result.error_largest_free_range_bytes)}"
            )
        encoded = library.shadowspill_status_string(status)
        detail = encoded.decode("utf-8") if encoded else f"planner status {status}"
        raise RuntimeError(f"schedule admission failed: {detail}")
    dependencies: list[MemoryReuseDependency] = []
    for index in range(int(result.reuse_count)):
        predecessor = int(reuse_predecessors[index])
        predecessor_task = int(reuse_predecessor_tasks[index])
        successor_task = int(reuse_tasks[index])
        successor_action = int(reuse_actions[index])
        if successor_task != NO_INDEX:
            dependencies.append(
                MemoryReuseDependency(
                    None if predecessor == NO_INDEX else predecessor,
                    predecessor_task_id=None
                    if predecessor_task == NO_INDEX
                    else simulation.task_ids[predecessor_task],
                    successor_task_id=simulation.task_ids[successor_task],
                )
            )
        else:
            dependencies.append(
                MemoryReuseDependency(
                    None if predecessor == NO_INDEX else predecessor,
                    predecessor_task_id=None
                    if predecessor_task == NO_INDEX
                    else simulation.task_ids[predecessor_task],
                    successor_action_index=successor_action,
                )
            )
    facts = admission.value
    simulation_admission = SimulationAdmission(
        initial_physical_bytes=(
            (simulation.device_ids[0], int(result.initial_physical_bytes)),
        ),
        device_capacity_bytes=(
            (simulation.device_ids[0], int(facts.pool_capacity_bytes)),
        ),
        task_deltas=tuple(
            TaskPhysicalDelta(
                task_id,
                int(task_start[index]),
                int(task_completion[index]),
            )
            for index, task_id in enumerate(simulation.task_ids)
        ),
        action_deltas=tuple(
            ActionPhysicalDelta(
                index,
                int(action_trigger[index]),
                int(action_completion[index]),
            )
            for index in range(action_count)
        ),
        reuse_dependencies=tuple(dependencies),
    )
    return CompiledScheduleAdmission(
        simulation_admission=simulation_admission,
        decision_digest=int(result.decision_digest),
        peak_allocated_bytes=int(result.peak_allocated_bytes),
        peak_reserved_bytes=int(result.peak_reserved_bytes),
        peak_fragmentation_bytes=int(result.peak_fragmentation_bytes),
    )


__all__ = [
    "CompiledScheduleAdmission",
    "IndexedAdmissionFacts",
    "IndexedMemorySchedule",
    "encode_schedule",
    "evaluate_schedule_admission",
    "index_admission_facts",
]
