"""The part of the simulator's input that one program fixes."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass

from shadowspill.ir import (
    MemoryActionKind,
    MemoryLocation,
    ResidencySpec,
    ResourceKind,
    ShadowSpillProgram,
    TaskAlternativeChoice,
    TaskSpec,
    shared_residency_footprint,
)
from shadowspill.ir.indexing import flatten_rows
from shadowspill.ir.sharing import SharedResidencyFootprint
from shadowspill.status import ABI_VERSION

from ..capi import (
    CDevice,
    CProgram,
)
from ..model import (
    SimulationConfig,
)
from .arrays import _Arena

_RESOURCE_CODE = {
    ResourceKind.COMPUTE: 0,
    ResourceKind.COMMUNICATION: 1,
    ResourceKind.CONTROL: 2,
}
_LOCATION_CODE = {
    MemoryLocation.DEVICE: 0,
    MemoryLocation.SPILL: 1,
}
_ACTION_CODE = {
    MemoryActionKind.RELEASE: 0,
    MemoryActionKind.EVICT: 1,
    MemoryActionKind.FETCH: 2,
    MemoryActionKind.WRITE_BACK: 3,
}
_STALL_REASONS = (
    (1 << 0, "input-residency"),
    (1 << 1, "device-capacity"),
    (1 << 2, "source-readiness"),
    (1 << 3, "host-capacity"),
    (1 << 4, "memory-reuse"),
)
_VIOLATION_REASONS = (
    "initial-device-capacity",
    "initial-spill-capacity",
    "fetch-device-capacity",
    "evict-spill-capacity",
    "task-device-capacity",
)
_VIOLATION_LOCATIONS = ("device", "spill")
_DEFAULT_PHYSICAL_DELTA = -(1 << 63)


@dataclass(frozen=True, slots=True)
class IndexedSimulationTemplate:
    """Immutable indexed topology reused across schedule candidates."""

    program: CProgram
    buffers: tuple[object, ...]
    task_ids: tuple[str, ...]
    task_index: dict[str, int]
    alias_ids: tuple[str, ...]
    alias_index: dict[str, int]
    device_ids: tuple[str, ...]
    task_resources: tuple[tuple[ResourceKind, int], ...]
    shared_device_bytes: tuple[int, ...]
    shared_spill_bytes: int


def _shared_footprint(
    program: ShadowSpillProgram, config: SimulationConfig
) -> tuple[SharedResidencyFootprint, tuple[int, ...]]:
    """Check the configured devices against the program, and price shared leases.

    What the runtime holds resident for every step is not the schedule's to
    move, so it is taken off each capacity before anything else is indexed.
    """

    configured = {item.device_id: item for item in config.devices}
    device_ids = tuple(item.device_id for item in program.devices)
    if set(configured) != set(device_ids):
        raise ValueError(
            "simulation devices must exactly match ShadowSpillProgram devices; "
            f"expected {sorted(device_ids)}, got {sorted(configured)}"
        )
    shared = shared_residency_footprint(program)
    shared_device_bytes = tuple(shared.for_device(item) for item in device_ids)
    for device_id, shared_bytes in zip(device_ids, shared_device_bytes, strict=True):
        capacity = configured[device_id].capacity_bytes
        if shared_bytes > capacity:
            raise ValueError(
                f"shared residency requires {shared_bytes} bytes on "
                f"{device_id!r}, exceeding capacity {capacity}"
            )
    if shared.spill_bytes > config.spill_capacity_bytes:
        raise ValueError(
            "shared spill residency exceeds host capacity: "
            f"shared={shared.spill_bytes}, capacity={config.spill_capacity_bytes}"
        )
    return shared, shared_device_bytes


def _task_rows(
    tasks: tuple[TaskSpec, ...],
    object_alias: dict[str, int],
    task_index: dict[str, int],
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    """Flatten each task's dependencies, inputs, outputs and mutations.

    Each row becomes one (offsets, values) pair, which is how the C program
    reads a ragged table.
    """

    dependencies = tuple(
        tuple(task_index[value] for value in task.dependencies) for task in tasks
    )
    inputs = tuple(
        tuple(dict.fromkeys(object_alias[value] for value in task.inputs))
        for task in tasks
    )
    outputs = tuple(
        tuple(dict.fromkeys(object_alias[value] for value in task.outputs))
        for task in tasks
    )
    mutations = tuple(
        tuple(object_alias[item.object_id] for item in task.mutations) for task in tasks
    )
    return tuple(
        flatten_rows(rows) for rows in (dependencies, inputs, outputs, mutations)
    )


def _boundary_residency(
    values: tuple[ResidencySpec, ...],
    alias_index: dict[str, int],
    arena: _Arena,
    *,
    field_name: str,
) -> tuple[ctypes.Array[ctypes.c_uint32], ctypes.Array[ctypes.c_uint8]]:
    """Index one residency declaration, refusing an unknown or repeated group."""

    seen: set[str] = set()
    aliases: list[int] = []
    locations: list[int] = []
    for index, value in enumerate(values):
        if value.alias_group_id not in alias_index:
            raise ValueError(
                f"{field_name}[{index}] contains unknown alias group "
                f"{value.alias_group_id!r}"
            )
        if value.alias_group_id in seen:
            raise ValueError(
                f"{field_name}[{index}] duplicates alias group {value.alias_group_id!r}"
            )
        seen.add(value.alias_group_id)
        aliases.append(alias_index[value.alias_group_id])
        locations.append(_LOCATION_CODE[value.location])
    return arena.u32(tuple(aliases)), arena.u8(tuple(locations))


def index_simulation_template(
    program: ShadowSpillProgram,
    selections: tuple[TaskAlternativeChoice, ...],
    config: SimulationConfig,
    *,
    selected_tasks: tuple[TaskSpec, ...] | None = None,
    initial_residency: tuple[ResidencySpec, ...] = (),
    final_residency: tuple[ResidencySpec, ...] = (),
) -> IndexedSimulationTemplate:
    """Project schedule-invariant program geometry exactly once.

    The optional residency declarations describe the planning boundary, not a
    candidate schedule.  They let the planner derive indexed planning
    facts directly from this immutable topology.  Candidate binding replaces
    these arrays with the selected schedule before simulation.
    """

    configured = {item.device_id: item for item in config.devices}
    device_ids = tuple(item.device_id for item in program.devices)
    shared, shared_device_bytes = _shared_footprint(program, config)
    alias_ids = tuple(item.alias_group_id for item in program.alias_groups)
    alias_index = {value: index for index, value in enumerate(alias_ids)}
    device_index = {value: index for index, value in enumerate(device_ids)}
    object_alias = {
        item.object_id: alias_index[item.alias_group_id] for item in program.objects
    }
    profiles = {item.profile_id: item for item in program.profiles}
    tasks = (
        program.selected_tasks(selections) if selected_tasks is None else selected_tasks
    )
    task_ids = tuple(item.task_id for item in tasks)
    task_index = {value: index for index, value in enumerate(task_ids)}
    rows = _task_rows(tasks, object_alias, task_index)
    (
        (dependency_offsets, dependency_values),
        (input_offsets, input_values),
        (output_offsets, output_values),
        (mutation_offsets, mutation_values),
    ) = rows

    arena = _Arena()
    c_devices = arena.keep(
        (CDevice * len(device_ids))(
            *(
                CDevice(
                    configured[device_id].capacity_bytes - shared.for_device(device_id),
                    configured[device_id].fetch_bandwidth_bytes_per_second,
                    configured[device_id].evict_bandwidth_bytes_per_second,
                    configured[device_id].fetch_latency_ns,
                    configured[device_id].evict_latency_ns,
                )
                for device_id in device_ids
            )
        )
    )
    initial_aliases, initial_locations = _boundary_residency(
        initial_residency, alias_index, arena, field_name="initial_residency"
    )
    final_aliases, final_locations = _boundary_residency(
        final_residency, alias_index, arena, field_name="final_residency"
    )
    empty_u32 = arena.u32(())
    empty_u8 = arena.u8(())
    empty_u64 = arena.u64(())
    empty_i64 = arena.i64(())
    c_program = CProgram(
        abi_version=ABI_VERSION,
        device_count=len(device_ids),
        alias_count=len(alias_ids),
        task_count=len(tasks),
        action_count=0,
        initial_count=len(initial_residency),
        final_count=len(final_residency),
        dependency_count=len(dependency_values),
        input_count=len(input_values),
        output_count=len(output_values),
        mutation_count=len(mutation_values),
        reuse_dependency_count=0,
        use_admission_accounting=0,
        spill_capacity_bytes=config.spill_capacity_bytes - shared.spill_bytes,
        devices=c_devices,
        alias_device=arena.u32(
            tuple(device_index[item.device_id] for item in program.alias_groups)
        ),
        alias_size_bytes=arena.u64(
            tuple(
                0 if item.shared_residency is not None else item.size_bytes
                for item in program.alias_groups
            )
        ),
        alias_initial_version=arena.u64(
            tuple(item.initial_version for item in program.alias_groups)
        ),
        alias_retain_spill_copy=arena.u8(
            tuple(
                int(item.retain_spill_copy and item.shared_residency is None)
                for item in program.alias_groups
            )
        ),
        task_device=arena.u32(
            tuple(device_index[item.resource.device_id] for item in tasks)
        ),
        task_resource_kind=arena.u8(
            tuple(_RESOURCE_CODE[item.resource.kind] for item in tasks)
        ),
        task_resource_lane=arena.u32(tuple(item.resource.lane for item in tasks)),
        task_runtime_ns=arena.u64(
            tuple(profiles[item.profile_id].runtime_ns for item in tasks)
        ),
        task_workspace_bytes=arena.u64(
            tuple(profiles[item.profile_id].workspace_bytes for item in tasks)
        ),
        task_start_physical_deltas=empty_i64,
        task_completion_physical_deltas=empty_i64,
        dependency_offsets=arena.u32(dependency_offsets),
        dependencies=arena.u32(dependency_values),
        input_offsets=arena.u32(input_offsets),
        input_aliases=arena.u32(input_values),
        output_offsets=arena.u32(output_offsets),
        output_aliases=arena.u32(output_values),
        mutation_offsets=arena.u32(mutation_offsets),
        mutation_aliases=arena.u32(mutation_values),
        mutation_version_deltas=arena.u64(
            tuple(item.version_delta for task in tasks for item in task.mutations)
        ),
        action_trigger_tasks=empty_u32,
        action_aliases=empty_u32,
        action_kinds=empty_u8,
        action_trigger_physical_deltas=empty_i64,
        action_completion_physical_deltas=empty_i64,
        initial_aliases=initial_aliases,
        initial_locations=initial_locations,
        initial_physical_bytes=empty_u64,
        final_aliases=final_aliases,
        final_locations=final_locations,
        reuse_predecessor_actions=empty_u32,
        reuse_successor_tasks=empty_u32,
        reuse_successor_actions=empty_u32,
    )
    return IndexedSimulationTemplate(
        c_program,
        tuple(arena.buffers),
        task_ids,
        task_index,
        alias_ids,
        alias_index,
        device_ids,
        tuple((item.resource.kind, item.resource.lane) for item in tasks),
        shared_device_bytes,
        shared.spill_bytes,
    )
