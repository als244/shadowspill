"""Indexed one-time projection of admission facts for the C planner."""

from __future__ import annotations

import ctypes
from array import array
from dataclasses import dataclass
from typing import Protocol

from shadowspill.ir import MemoryActionKind, MemoryLocation, MemorySchedule
from shadowspill.simulator import (
    ActionPhysicalDelta,
    MemoryReuseDependency,
    SimulationAdmission,
    TaskPhysicalDelta,
)
from shadowspill.simulator.capi import NO_INDEX
from shadowspill.simulator.indexed import IndexedSimulationTemplate
from shadowspill.status import ABI_VERSION, Status

from ..capi import (
    CAdmissionFacts,
    CIndexedSchedule,
    CScheduleAdmissionResult,
    planner_api,
)
from . import (
    AdmissionFacts,
    TaskAdmissionSpec,
    TaskAllocationStepKind,
)


def _u32(values: tuple[int, ...]) -> ctypes.Array[ctypes.c_uint32]:
    result_type = ctypes.c_uint32 * max(1, len(values))
    return result_type.from_buffer_copy(array("I", values)) if values else result_type()


def _u64(values: tuple[int, ...]) -> ctypes.Array[ctypes.c_uint64]:
    result_type = ctypes.c_uint64 * max(1, len(values))
    return result_type.from_buffer_copy(array("Q", values)) if values else result_type()


def _u8(values: tuple[int, ...]) -> ctypes.Array[ctypes.c_uint8]:
    result_type = ctypes.c_uint8 * max(1, len(values))
    return result_type.from_buffer_copy(bytes(values)) if values else result_type()


def _flatten_rows(
    rows: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    offsets = [0]
    flattened: list[int] = []
    for row in rows:
        flattened.extend(row)
        offsets.append(len(flattened))
    return tuple(offsets), tuple(flattened)


@dataclass(frozen=True, slots=True)
class IndexedAdmissionFacts:
    """Borrowed C facts plus Python owners for all indexed arrays."""

    value: CAdmissionFacts
    buffers: tuple[object, ...]
    digest: str


class IndexedSchedule(Protocol):
    """Structural interface shared with the PressureFit winner."""

    @property
    def action_trigger_tasks(self) -> tuple[int, ...]: ...

    @property
    def action_aliases(self) -> tuple[int, ...]: ...

    @property
    def action_kinds(self) -> tuple[int, ...]: ...

    @property
    def initial_aliases(self) -> tuple[int, ...]: ...

    @property
    def initial_locations(self) -> tuple[int, ...]: ...

    @property
    def final_aliases(self) -> tuple[int, ...]: ...

    @property
    def final_locations(self) -> tuple[int, ...]: ...


@dataclass(frozen=True, slots=True)
class CompiledScheduleAdmission:
    """Exact physical projection for one selected indexed schedule."""

    simulation_admission: SimulationAdmission
    decision_digest: int
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    peak_fragmentation_bytes: int


@dataclass(frozen=True, slots=True)
class EncodedIndexedSchedule:
    """Indexed schedule projection accepted by planner helpers."""

    action_trigger_tasks: tuple[int, ...]
    action_aliases: tuple[int, ...]
    action_kinds: tuple[int, ...]
    initial_aliases: tuple[int, ...]
    initial_locations: tuple[int, ...]
    final_aliases: tuple[int, ...]
    final_locations: tuple[int, ...]


_ACTION_KIND = {
    MemoryActionKind.RELEASE: 0,
    MemoryActionKind.EVICT: 1,
    MemoryActionKind.FETCH: 2,
}
_LOCATION = {MemoryLocation.DEVICE: 0, MemoryLocation.SPILL: 1}
_ALLOCATION_KIND = {
    TaskAllocationStepKind.ALLOCATE: 0,
    TaskAllocationStepKind.RELEASE: 1,
}
_CompiledAllocationRow = tuple[tuple[int, int, int, int], ...]


def encode_schedule(
    schedule: MemorySchedule,
    simulation: IndexedSimulationTemplate,
) -> EncodedIndexedSchedule:
    """Encode one public schedule against an immutable facts."""

    return EncodedIndexedSchedule(
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
    """Project only tasks selected by one recomputation problem."""

    if facts.device_id not in simulation.device_ids:
        raise ValueError(
            f"admission device {facts.device_id!r} is absent from simulation"
        )
    by_task = {item.task_id: item for item in facts.tasks}
    try:
        tasks = tuple(by_task[item] for item in simulation.task_ids)
    except KeyError as exc:
        raise ValueError(
            f"admission facts lacks selected task {exc.args[0]!r}"
        ) from exc
    alias_index = simulation.alias_index
    fresh_rows = tuple(
        tuple(alias_index[item] for item in task.fresh_output_aliases) for task in tasks
    )
    replacement_rows = tuple(
        tuple(alias_index[item] for item in task.replacement_aliases) for task in tasks
    )
    handoff_source_rows = tuple(
        tuple(alias_index[item.source_alias_group_id] for item in task.storage_handoffs)
        for task in tasks
    )
    handoff_destination_rows = tuple(
        tuple(
            alias_index[item.destination_alias_group_id]
            for item in task.storage_handoffs
        )
        for task in tasks
    )
    fresh_offsets, fresh = _flatten_rows(fresh_rows)
    replacement_offsets, replacements = _flatten_rows(replacement_rows)
    handoff_offsets, handoff_sources = _flatten_rows(handoff_source_rows)
    _, handoff_destinations = _flatten_rows(handoff_destination_rows)
    workspace_offsets, workspace_extents = _flatten_rows(
        tuple(task.workspace_extents for task in tasks)
    )
    allocation_rows, allocation_slot_count = _compile_allocation_rows(
        tasks,
        alias_index=alias_index,
    )
    allocation_offsets, allocation_kinds = _flatten_rows(
        tuple(tuple(item[0] for item in row) for row in allocation_rows)
    )
    _, allocation_slots = _flatten_rows(
        tuple(tuple(item[1] for item in row) for row in allocation_rows)
    )
    _, allocation_bytes = _flatten_rows(
        tuple(tuple(item[2] for item in row) for row in allocation_rows)
    )
    _, allocation_aliases = _flatten_rows(
        tuple(tuple(item[3] for item in row) for row in allocation_rows)
    )
    buffers = (
        _u32(workspace_offsets),
        _u64(workspace_extents),
        _u32(fresh_offsets),
        _u32(fresh),
        _u32(replacement_offsets),
        _u32(replacements),
        _u32(handoff_offsets),
        _u32(handoff_sources),
        _u32(handoff_destinations),
        _u32(allocation_offsets),
        _u32(allocation_slots),
        _u64(allocation_bytes),
        _u32(allocation_aliases),
        _u8(allocation_kinds),
    )
    value = CAdmissionFacts(
        abi_version=ABI_VERSION,
        task_count=len(tasks),
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


def _compile_allocation_rows(
    tasks: tuple[TaskAdmissionSpec, ...],
    *,
    alias_index: dict[str, int],
) -> tuple[tuple[_CompiledAllocationRow, ...], int]:
    """Assign indexed lease slots while preserving each profiled task order."""

    rows: list[tuple[tuple[int, int, int, int], ...]] = []
    next_slot = 0
    for task in tasks:
        steps = task.allocation_steps
        slot_by_ordinal: dict[int, int] = {}
        row: list[tuple[int, int, int, int]] = []
        for step in steps:
            if step.kind is TaskAllocationStepKind.ALLOCATE:
                if step.reuses_allocation_ordinal is None:
                    slot = next_slot
                    next_slot += 1
                else:
                    slot = slot_by_ordinal[step.reuses_allocation_ordinal]
                slot_by_ordinal[step.allocation_ordinal] = slot
            else:
                slot = slot_by_ordinal[step.allocation_ordinal]
            row.append(
                (
                    _ALLOCATION_KIND[step.kind],
                    slot,
                    step.charged_bytes,
                    (
                        NO_INDEX
                        if step.output_alias_group_id is None
                        else alias_index[step.output_alias_group_id]
                    ),
                )
            )
        rows.append(tuple(row))
    return tuple(rows), next_slot


def evaluate_schedule_admission(
    simulation: IndexedSimulationTemplate,
    admission: IndexedAdmissionFacts,
    schedule: IndexedSchedule,
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
        reuse_successor_tasks=reuse_tasks,
        reuse_successor_actions=reuse_actions,
        reuse_capacity=reuse_capacity,
    )
    library = planner_api()
    status = int(
        library.shadowspill_evaluate_schedule_admission(
            ctypes.byref(simulation.program),
            ctypes.byref(admission.value),
            ctypes.byref(indexed),
            ctypes.byref(result),
        )
    )
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
        successor_task = int(reuse_tasks[index])
        successor_action = int(reuse_actions[index])
        if successor_task != NO_INDEX:
            dependencies.append(
                MemoryReuseDependency(
                    predecessor,
                    successor_task_id=simulation.task_ids[successor_task],
                )
            )
        else:
            dependencies.append(
                MemoryReuseDependency(
                    predecessor,
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
    "EncodedIndexedSchedule",
    "IndexedAdmissionFacts",
    "encode_schedule",
    "evaluate_schedule_admission",
    "index_admission_facts",
]
