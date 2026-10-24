"""One candidate schedule bound onto a template, ready to simulate."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass

import numpy as np

from shadowspill.ir import (
    MemorySchedule,
    ResourceKind,
    ShadowSpillProgram,
    TaskAlternativeChoice,
)
from shadowspill.ir.indexing import (
    MEMORY_ACTION_CODE as _ACTION_CODE,
)
from shadowspill.ir.indexing import (
    MEMORY_LOCATION_CODE as _LOCATION_CODE,
)

from ..capi import (
    CDevice,
    CProgram,
)
from ..model import (
    SimulationAdmission,
    SimulationConfig,
)
from .arrays import _i64_array, _u8_array, _u32_array, _u64_array
from .reuse import reuse_arrays
from .template import IndexedSimulationTemplate, index_simulation_template

#: The binding's marker for a delta the admission leaves to the simulator.
_DEFAULT_PHYSICAL_DELTA = -(1 << 63)


@dataclass(slots=True)
class _Projection:
    program: CProgram
    buffers: tuple[object, ...]
    task_ids: tuple[str, ...]
    alias_ids: tuple[str, ...]
    device_ids: tuple[str, ...]
    task_resources: tuple[tuple[ResourceKind, int], ...]
    shared_device_bytes: tuple[int, ...]
    shared_spill_bytes: int
    template: IndexedSimulationTemplate | None = None


@dataclass(frozen=True, slots=True)
class _ScheduleArrays:
    """One candidate schedule, indexed onto a template's alias and task order."""

    action_tasks: ctypes.Array[ctypes.c_uint32]
    action_aliases: ctypes.Array[ctypes.c_uint32]
    action_kinds: ctypes.Array[ctypes.c_uint8]
    initial_aliases: ctypes.Array[ctypes.c_uint32]
    initial_locations: ctypes.Array[ctypes.c_uint8]
    final_aliases: ctypes.Array[ctypes.c_uint32]
    final_locations: ctypes.Array[ctypes.c_uint8]


@dataclass(frozen=True, slots=True)
class _AdmissionArrays:
    """The physical accounting a simulation is given, or empty arrays for none."""

    initial_physical: ctypes.Array[ctypes.c_uint64]
    task_start_deltas: ctypes.Array[ctypes.c_int64]
    task_completion_deltas: ctypes.Array[ctypes.c_int64]
    action_trigger_deltas: ctypes.Array[ctypes.c_int64]
    action_completion_deltas: ctypes.Array[ctypes.c_int64]
    reuse_predecessors: ctypes.Array[ctypes.c_uint32]
    reuse_predecessor_tasks: ctypes.Array[ctypes.c_uint32]
    reuse_successor_tasks: ctypes.Array[ctypes.c_uint32]
    reuse_successor_actions: ctypes.Array[ctypes.c_uint32]
    devices: ctypes.Array[CDevice] | None


def _schedule_arrays(
    template: IndexedSimulationTemplate, schedule: MemorySchedule
) -> _ScheduleArrays:
    """Index the schedule's actions and its two residency boundaries."""

    return _ScheduleArrays(
        action_tasks=_u32_array(
            tuple(
                template.task_index[item.trigger_task_id] for item in schedule.actions
            )
        ),
        action_aliases=_u32_array(
            tuple(
                template.alias_index[item.alias_group_id] for item in schedule.actions
            )
        ),
        action_kinds=_u8_array(
            tuple(_ACTION_CODE[item.kind] for item in schedule.actions)
        ),
        initial_aliases=_u32_array(
            tuple(
                template.alias_index[item.alias_group_id]
                for item in schedule.initial_residency
            )
        ),
        initial_locations=_u8_array(
            tuple(_LOCATION_CODE[item.location] for item in schedule.initial_residency)
        ),
        final_aliases=_u32_array(
            tuple(
                template.alias_index[item.alias_group_id]
                for item in schedule.final_residency
            )
        ),
        final_locations=_u8_array(
            tuple(_LOCATION_CODE[item.location] for item in schedule.final_residency)
        ),
    )


def _no_admission() -> _AdmissionArrays:
    """Empty arrays: the simulation prices no physical accounting."""

    return _AdmissionArrays(
        initial_physical=_u64_array(()),
        task_start_deltas=_i64_array(()),
        task_completion_deltas=_i64_array(()),
        action_trigger_deltas=_i64_array(()),
        action_completion_deltas=_i64_array(()),
        reuse_predecessors=_u32_array(()),
        reuse_predecessor_tasks=_u32_array(()),
        reuse_successor_tasks=_u32_array(()),
        reuse_successor_actions=_u32_array(()),
        devices=None,
    )


def _admission_arrays(
    template: IndexedSimulationTemplate,
    schedule: MemorySchedule,
    admission: SimulationAdmission,
) -> _AdmissionArrays:
    """Index the admission, refusing anything it names that the program lacks."""

    initial_by_device = dict(admission.initial_physical_bytes)
    if set(initial_by_device) != set(template.device_ids):
        raise ValueError(
            "simulation admission devices must exactly match program devices; "
            f"expected {sorted(template.device_ids)}, "
            f"got {sorted(initial_by_device)}"
        )
    task_deltas = {item.task_id: item for item in admission.task_deltas}
    unknown_tasks = set(task_deltas) - set(template.task_ids)
    if unknown_tasks:
        raise ValueError(
            f"simulation admission contains unknown task IDs: {sorted(unknown_tasks)}"
        )
    action_deltas = {item.action_index: item for item in admission.action_deltas}
    unknown_actions = set(action_deltas) - set(range(len(schedule.actions)))
    if unknown_actions:
        raise ValueError(
            "simulation admission contains unknown action indices: "
            f"{sorted(unknown_actions)}"
        )
    physical_capacity = dict(admission.device_capacity_bytes)
    if physical_capacity and set(physical_capacity) != set(template.device_ids):
        raise ValueError(
            "simulation admission capacities must exactly match ShadowSpillProgram "
            f"devices; expected {sorted(template.device_ids)}, "
            f"got {sorted(physical_capacity)}"
        )
    (
        reuse_predecessors,
        reuse_predecessor_tasks,
        reuse_successor_tasks,
        reuse_successor_actions,
    ) = reuse_arrays(template, schedule, admission)
    # Every entry starts at the default for what the admission leaves out, and
    # the entries it lists are written over it.
    unlisted = (
        _DEFAULT_PHYSICAL_DELTA
        if admission.unlisted_delta_bytes is None
        else admission.unlisted_delta_bytes
    )
    task_start = np.full(len(template.task_ids), unlisted, dtype=np.int64)
    task_completion = task_start.copy()
    for item in task_deltas.values():
        index = template.task_index[item.task_id]
        task_start[index] = item.start_bytes
        task_completion[index] = item.completion_bytes
    action_trigger = np.full(len(schedule.actions), unlisted, dtype=np.int64)
    action_completion = action_trigger.copy()
    for index, delta in action_deltas.items():
        action_trigger[index] = delta.trigger_bytes
        action_completion[index] = delta.completion_bytes
    return _AdmissionArrays(
        initial_physical=_u64_array(
            tuple(initial_by_device[item] for item in template.device_ids)
        ),
        task_start_deltas=_i64_array(task_start),
        task_completion_deltas=_i64_array(task_completion),
        action_trigger_deltas=_i64_array(action_trigger),
        action_completion_deltas=_i64_array(action_completion),
        reuse_predecessors=reuse_predecessors,
        reuse_predecessor_tasks=reuse_predecessor_tasks,
        reuse_successor_tasks=reuse_successor_tasks,
        reuse_successor_actions=reuse_successor_actions,
        devices=(CDevice * len(template.device_ids))(
            *(
                CDevice(
                    physical_capacity.get(
                        device_id, int(template.program.devices[index].capacity_bytes)
                    ),
                    int(
                        template.program.devices[
                            index
                        ].fetch_solo_bandwidth_bytes_per_second
                    ),
                    int(
                        template.program.devices[
                            index
                        ].fetch_concurrent_bandwidth_bytes_per_second
                    ),
                    int(
                        template.program.devices[
                            index
                        ].evict_solo_bandwidth_bytes_per_second
                    ),
                    int(
                        template.program.devices[
                            index
                        ].evict_concurrent_bandwidth_bytes_per_second
                    ),
                    int(template.program.devices[index].fetch_latency_ns),
                    int(template.program.devices[index].evict_latency_ns),
                )
                for index, device_id in enumerate(template.device_ids)
            )
        ),
    )


def _bind_schedule(
    template: IndexedSimulationTemplate,
    schedule: MemorySchedule,
    admission: SimulationAdmission | None = None,
) -> _Projection:
    """Bind candidate-only arrays to one immutable topology."""

    bound = _schedule_arrays(template, schedule)
    physical = (
        _no_admission()
        if admission is None
        else _admission_arrays(template, schedule, admission)
    )
    c_program = CProgram.from_buffer_copy(template.program)
    if physical.devices is not None:
        c_program.devices = physical.devices
    c_program.action_count = len(schedule.actions)
    c_program.initial_count = len(schedule.initial_residency)
    c_program.final_count = len(schedule.final_residency)
    c_program.reuse_dependency_count = (
        0 if admission is None else len(admission.reuse_dependencies)
    )
    c_program.use_admission_accounting = int(admission is not None)
    c_program.action_trigger_tasks = bound.action_tasks
    c_program.action_aliases = bound.action_aliases
    c_program.action_kinds = bound.action_kinds
    c_program.action_trigger_physical_deltas = physical.action_trigger_deltas
    c_program.action_completion_physical_deltas = physical.action_completion_deltas
    c_program.initial_aliases = bound.initial_aliases
    c_program.initial_locations = bound.initial_locations
    c_program.initial_physical_bytes = physical.initial_physical
    c_program.final_aliases = bound.final_aliases
    c_program.final_locations = bound.final_locations
    c_program.task_start_physical_deltas = physical.task_start_deltas
    c_program.task_completion_physical_deltas = physical.task_completion_deltas
    c_program.reuse_predecessor_actions = physical.reuse_predecessors
    c_program.reuse_predecessor_tasks = physical.reuse_predecessor_tasks
    c_program.reuse_successor_tasks = physical.reuse_successor_tasks
    c_program.reuse_successor_actions = physical.reuse_successor_actions
    return _Projection(
        c_program,
        (template, bound, physical),
        template.task_ids,
        template.alias_ids,
        template.device_ids,
        template.task_resources,
        template.shared_device_bytes,
        template.shared_spill_bytes,
        template=template,
    )


def _project(
    program: ShadowSpillProgram,
    schedule: MemorySchedule,
    selections: tuple[TaskAlternativeChoice, ...],
    config: SimulationConfig,
    admission: SimulationAdmission | None,
) -> _Projection:
    schedule.validate(program, selections)
    return _bind_schedule(
        index_simulation_template(program, selections, config),
        schedule,
        admission,
    )
