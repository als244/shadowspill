"""Canonical IR projection and result decoding for the simulator."""

from __future__ import annotations

import ctypes
from array import array
from dataclasses import dataclass

from shadowspill._status import ABI_VERSION, Status
from shadowspill.ir import (
    MemoryActionKind,
    MemoryLocation,
    MemorySchedule,
    Program,
    RecomputationSelection,
    ResidencySpec,
    ResourceKind,
    TaskSpec,
    shared_residency_footprint,
)

from ._capi import (
    NO_INDEX,
    CDevice,
    CDevicePeak,
    CProgram,
    CResult,
    CTaskInterval,
    CTransferInterval,
    simulator_api,
)
from ._diagnostics import simulation_failure_detail, simulation_status_kind
from .model import (
    DeviceMemoryPeak,
    SimulationAdmission,
    SimulationConfig,
    SimulationInfeasibleError,
    SimulationResult,
    TaskInterval,
    TransferDirection,
    TransferInterval,
)

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
    MemoryActionKind.OFFLOAD: 1,
    MemoryActionKind.PREFETCH: 2,
}
_STALL_REASONS = (
    (1 << 0, "input-residency"),
    (1 << 1, "device-capacity"),
    (1 << 2, "source-readiness"),
    (1 << 3, "host-capacity"),
    (1 << 4, "memory-reuse"),
)
_DEFAULT_PHYSICAL_DELTA = -(1 << 63)


def _u32_array(values: tuple[int, ...]) -> ctypes.Array[ctypes.c_uint32]:
    array_type = ctypes.c_uint32 * max(1, len(values))
    return array_type.from_buffer_copy(array("I", values)) if values else array_type()


def _u64_array(values: tuple[int, ...]) -> ctypes.Array[ctypes.c_uint64]:
    array_type = ctypes.c_uint64 * max(1, len(values))
    return array_type.from_buffer_copy(array("Q", values)) if values else array_type()


def _i64_array(values: tuple[int, ...]) -> ctypes.Array[ctypes.c_int64]:
    array_type = ctypes.c_int64 * max(1, len(values))
    return array_type(*values) if values else array_type()


def _u8_array(values: tuple[int, ...]) -> ctypes.Array[ctypes.c_uint8]:
    array_type = ctypes.c_uint8 * max(1, len(values))
    return array_type.from_buffer_copy(bytes(values)) if values else array_type()


def _offsets(
    rows: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    offsets = [0]
    values: list[int] = []
    for row in rows:
        values.extend(row)
        offsets.append(len(values))
    return tuple(offsets), tuple(values)


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


@dataclass(frozen=True, slots=True)
class IndexedSimulationSummary:
    """Selection-only result that avoids materializing interval records."""

    makespan_ns: int


def index_simulation_template(
    program: Program,
    selections: tuple[RecomputationSelection, ...],
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
    if set(configured) != set(device_ids):
        raise ValueError(
            "simulation devices must exactly match Program devices; "
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
    mutation_deltas = tuple(
        item.version_delta for task in tasks for item in task.mutations
    )
    dependency_offsets, dependency_values = _offsets(dependencies)
    input_offsets, input_values = _offsets(inputs)
    output_offsets, output_values = _offsets(outputs)
    mutation_offsets, mutation_values = _offsets(mutations)

    buffers: list[object] = []

    def keep(value: object) -> object:
        buffers.append(value)
        return value

    c_devices = (CDevice * len(device_ids))(
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
    keep(c_devices)

    def u32(values: tuple[int, ...]) -> ctypes.Array[ctypes.c_uint32]:
        result = _u32_array(values)
        keep(result)
        return result

    def u64(values: tuple[int, ...]) -> ctypes.Array[ctypes.c_uint64]:
        result = _u64_array(values)
        keep(result)
        return result

    def u8(values: tuple[int, ...]) -> ctypes.Array[ctypes.c_uint8]:
        result = _u8_array(values)
        keep(result)
        return result

    alias_device = u32(
        tuple(device_index[item.device_id] for item in program.alias_groups)
    )
    alias_size = u64(
        tuple(
            0 if item.shared_residency is not None else item.size_bytes
            for item in program.alias_groups
        )
    )
    alias_version = u64(tuple(item.initial_version for item in program.alias_groups))
    alias_spill = u8(
        tuple(
            int(item.retain_spill_copy and item.shared_residency is None)
            for item in program.alias_groups
        )
    )
    task_device = u32(tuple(device_index[item.resource.device_id] for item in tasks))
    task_kind = u8(tuple(_RESOURCE_CODE[item.resource.kind] for item in tasks))
    task_lane = u32(tuple(item.resource.lane for item in tasks))
    task_runtime = u64(tuple(profiles[item.profile_id].runtime_ns for item in tasks))
    task_workspace = u64(
        tuple(profiles[item.profile_id].workspace_bytes for item in tasks)
    )
    dependency_offsets_buffer = u32(dependency_offsets)
    dependency_buffer = u32(dependency_values)
    input_offsets_buffer = u32(input_offsets)
    input_buffer = u32(input_values)
    output_offsets_buffer = u32(output_offsets)
    output_buffer = u32(output_values)
    mutation_offsets_buffer = u32(mutation_offsets)
    mutation_buffer = u32(mutation_values)
    mutation_delta_buffer = u64(mutation_deltas)

    def residency_arrays(
        values: tuple[ResidencySpec, ...],
        *,
        field: str,
    ) -> tuple[ctypes.Array[ctypes.c_uint32], ctypes.Array[ctypes.c_uint8]]:
        seen: set[str] = set()
        aliases: list[int] = []
        locations: list[int] = []
        for index, value in enumerate(values):
            if value.alias_group_id not in alias_index:
                raise ValueError(
                    f"{field}[{index}] contains unknown alias group "
                    f"{value.alias_group_id!r}"
                )
            if value.alias_group_id in seen:
                raise ValueError(
                    f"{field}[{index}] duplicates alias group {value.alias_group_id!r}"
                )
            seen.add(value.alias_group_id)
            aliases.append(alias_index[value.alias_group_id])
            locations.append(_LOCATION_CODE[value.location])
        return u32(tuple(aliases)), u8(tuple(locations))

    initial_aliases, initial_locations = residency_arrays(
        initial_residency,
        field="initial_residency",
    )
    final_aliases, final_locations = residency_arrays(
        final_residency,
        field="final_residency",
    )
    empty_u32 = u32(())
    empty_u8 = u8(())
    empty_u64 = u64(())
    empty_i64 = _i64_array(())
    keep(empty_i64)
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
        alias_device=alias_device,
        alias_size_bytes=alias_size,
        alias_initial_version=alias_version,
        alias_retain_spill_copy=alias_spill,
        task_device=task_device,
        task_resource_kind=task_kind,
        task_resource_lane=task_lane,
        task_runtime_ns=task_runtime,
        task_workspace_bytes=task_workspace,
        task_start_physical_deltas=empty_i64,
        task_completion_physical_deltas=empty_i64,
        dependency_offsets=dependency_offsets_buffer,
        dependencies=dependency_buffer,
        input_offsets=input_offsets_buffer,
        input_aliases=input_buffer,
        output_offsets=output_offsets_buffer,
        output_aliases=output_buffer,
        mutation_offsets=mutation_offsets_buffer,
        mutation_aliases=mutation_buffer,
        mutation_version_deltas=mutation_delta_buffer,
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
        tuple(buffers),
        task_ids,
        task_index,
        alias_ids,
        alias_index,
        device_ids,
        tuple((item.resource.kind, item.resource.lane) for item in tasks),
        shared_device_bytes,
        shared.spill_bytes,
    )


def _bind_schedule(
    template: IndexedSimulationTemplate,
    schedule: MemorySchedule,
    admission: SimulationAdmission | None = None,
) -> _Projection:
    """Bind candidate-only arrays to one immutable topology."""

    physical_devices: ctypes.Array[CDevice] | None = None
    action_tasks = _u32_array(
        tuple(template.task_index[item.trigger_task_id] for item in schedule.actions)
    )
    action_aliases = _u32_array(
        tuple(template.alias_index[item.alias_group_id] for item in schedule.actions)
    )
    action_kinds = _u8_array(
        tuple(_ACTION_CODE[item.kind] for item in schedule.actions)
    )
    initial_aliases = _u32_array(
        tuple(
            template.alias_index[item.alias_group_id]
            for item in schedule.initial_residency
        )
    )
    initial_locations = _u8_array(
        tuple(_LOCATION_CODE[item.location] for item in schedule.initial_residency)
    )
    final_aliases = _u32_array(
        tuple(
            template.alias_index[item.alias_group_id]
            for item in schedule.final_residency
        )
    )
    final_locations = _u8_array(
        tuple(_LOCATION_CODE[item.location] for item in schedule.final_residency)
    )
    if admission is None:
        initial_physical = _u64_array(())
        task_start_deltas = _i64_array(())
        task_completion_deltas = _i64_array(())
        action_trigger_deltas = _i64_array(())
        action_completion_deltas = _i64_array(())
        reuse_predecessors = _u32_array(())
        reuse_successor_tasks = _u32_array(())
        reuse_successor_actions = _u32_array(())
    else:
        initial_by_device = dict(admission.initial_physical_bytes)
        if set(initial_by_device) != set(template.device_ids):
            raise ValueError(
                "simulation admission devices must exactly match Program devices; "
                f"expected {sorted(template.device_ids)}, "
                f"got {sorted(initial_by_device)}"
            )
        task_deltas = {item.task_id: item for item in admission.task_deltas}
        unknown_tasks = set(task_deltas) - set(template.task_ids)
        if unknown_tasks:
            raise ValueError(
                "simulation admission contains unknown task IDs: "
                f"{sorted(unknown_tasks)}"
            )
        action_deltas = {item.action_index: item for item in admission.action_deltas}
        unknown_actions = set(action_deltas) - set(range(len(schedule.actions)))
        if unknown_actions:
            raise ValueError(
                "simulation admission contains unknown action indices: "
                f"{sorted(unknown_actions)}"
            )
        initial_physical = _u64_array(
            tuple(initial_by_device[item] for item in template.device_ids)
        )
        physical_capacity = dict(admission.device_capacity_bytes)
        if physical_capacity and set(physical_capacity) != set(template.device_ids):
            raise ValueError(
                "simulation admission capacities must exactly match Program "
                f"devices; expected {sorted(template.device_ids)}, "
                f"got {sorted(physical_capacity)}"
            )
        physical_devices = (CDevice * len(template.device_ids))(
            *(
                CDevice(
                    physical_capacity.get(
                        device_id,
                        int(template.program.devices[index].capacity_bytes),
                    ),
                    int(
                        template.program.devices[index].fetch_bandwidth_bytes_per_second
                    ),
                    int(
                        template.program.devices[index].evict_bandwidth_bytes_per_second
                    ),
                    int(template.program.devices[index].fetch_latency_ns),
                    int(template.program.devices[index].evict_latency_ns),
                )
                for index, device_id in enumerate(template.device_ids)
            )
        )
        task_start_deltas = _i64_array(
            tuple(
                task_deltas[item].start_bytes
                if item in task_deltas
                else _DEFAULT_PHYSICAL_DELTA
                for item in template.task_ids
            )
        )
        task_completion_deltas = _i64_array(
            tuple(
                task_deltas[item].completion_bytes
                if item in task_deltas
                else _DEFAULT_PHYSICAL_DELTA
                for item in template.task_ids
            )
        )
        action_trigger_deltas = _i64_array(
            tuple(
                action_deltas[index].trigger_bytes
                if index in action_deltas
                else _DEFAULT_PHYSICAL_DELTA
                for index in range(len(schedule.actions))
            )
        )
        action_completion_deltas = _i64_array(
            tuple(
                action_deltas[index].completion_bytes
                if index in action_deltas
                else _DEFAULT_PHYSICAL_DELTA
                for index in range(len(schedule.actions))
            )
        )
        predecessor_actions: list[int] = []
        successor_tasks: list[int] = []
        successor_actions: list[int] = []
        for dependency in admission.reuse_dependencies:
            predecessor = dependency.predecessor_action_index
            if predecessor >= len(schedule.actions):
                raise ValueError(
                    f"memory-reuse predecessor action is unknown: {predecessor}"
                )
            if schedule.actions[predecessor].kind is not MemoryActionKind.OFFLOAD:
                raise ValueError(
                    f"memory-reuse predecessor must be an OFFLOAD action: {predecessor}"
                )
            predecessor_actions.append(predecessor)
            if dependency.successor_task_id is None:
                assert dependency.successor_action_index is not None
                if dependency.successor_action_index >= len(schedule.actions):
                    raise ValueError(
                        "memory-reuse successor action is unknown: "
                        f"{dependency.successor_action_index}"
                    )
                successor_tasks.append(NO_INDEX)
                successor_actions.append(dependency.successor_action_index)
            else:
                try:
                    successor_tasks.append(
                        template.task_index[dependency.successor_task_id]
                    )
                except KeyError as exc:
                    raise ValueError(
                        "memory-reuse successor task is unknown: "
                        f"{dependency.successor_task_id!r}"
                    ) from exc
                successor_actions.append(NO_INDEX)
        reuse_predecessors = _u32_array(tuple(predecessor_actions))
        reuse_successor_tasks = _u32_array(tuple(successor_tasks))
        reuse_successor_actions = _u32_array(tuple(successor_actions))
    c_program = CProgram.from_buffer_copy(template.program)
    if physical_devices is not None:
        c_program.devices = physical_devices
    c_program.action_count = len(schedule.actions)
    c_program.initial_count = len(schedule.initial_residency)
    c_program.final_count = len(schedule.final_residency)
    c_program.reuse_dependency_count = (
        0 if admission is None else len(admission.reuse_dependencies)
    )
    c_program.use_admission_accounting = int(admission is not None)
    c_program.action_trigger_tasks = action_tasks
    c_program.action_aliases = action_aliases
    c_program.action_kinds = action_kinds
    c_program.action_trigger_physical_deltas = action_trigger_deltas
    c_program.action_completion_physical_deltas = action_completion_deltas
    c_program.initial_aliases = initial_aliases
    c_program.initial_locations = initial_locations
    c_program.initial_physical_bytes = initial_physical
    c_program.final_aliases = final_aliases
    c_program.final_locations = final_locations
    c_program.task_start_physical_deltas = task_start_deltas
    c_program.task_completion_physical_deltas = task_completion_deltas
    c_program.reuse_predecessor_actions = reuse_predecessors
    c_program.reuse_successor_tasks = reuse_successor_tasks
    c_program.reuse_successor_actions = reuse_successor_actions
    return _Projection(
        c_program,
        (
            template,
            action_tasks,
            action_aliases,
            action_kinds,
            initial_aliases,
            initial_locations,
            final_aliases,
            final_locations,
            initial_physical,
            task_start_deltas,
            task_completion_deltas,
            action_trigger_deltas,
            action_completion_deltas,
            reuse_predecessors,
            reuse_successor_tasks,
            reuse_successor_actions,
            physical_devices,
        ),
        template.task_ids,
        template.alias_ids,
        template.device_ids,
        template.task_resources,
        template.shared_device_bytes,
        template.shared_spill_bytes,
    )


def _project(
    program: Program,
    schedule: MemorySchedule,
    selections: tuple[RecomputationSelection, ...],
    config: SimulationConfig,
    admission: SimulationAdmission | None,
) -> _Projection:
    schedule.validate(program, selections)
    return _bind_schedule(
        index_simulation_template(program, selections, config),
        schedule,
        admission,
    )


def _stall_reasons(mask: int) -> tuple[str, ...]:
    return tuple(name for bit, name in _STALL_REASONS if mask & bit)


def _optional_name(names: tuple[str, ...], index: int) -> str | None:
    return None if index == NO_INDEX else names[index]


def _raise_error(
    status: int,
    result: CResult,
    projection: _Projection,
) -> None:
    alias = _optional_name(projection.alias_ids, int(result.error_alias))
    message = simulation_failure_detail(
        status,
        time_ns=int(result.error_time_ns),
        error_device=int(result.error_device),
        error_location=int(result.error_location),
        capacity_bytes=int(result.error_capacity_bytes),
        used_bytes=int(result.error_used_bytes),
        requested_bytes=int(result.error_requested_bytes),
        device_ids=projection.device_ids,
    )
    raise SimulationInfeasibleError(
        message,
        kind=simulation_status_kind(status),
        time_ns=int(result.error_time_ns),
        task_id=_optional_name(projection.task_ids, int(result.error_task)),
        alias_group_ids=(() if alias is None else (alias,)),
        location=(
            None
            if int(result.error_device) == NO_INDEX and status
            not in (
                Status.INITIAL_SPILL_CAPACITY,
                Status.OFFLOAD_SPILL_CAPACITY,
                Status.FINAL_RESIDENCY,
            )
            else (
                "host"
                if int(result.error_location) == 1
                else f"device:{projection.device_ids[int(result.error_device)]}"
            )
        ),
        capacity_bytes=int(result.error_capacity_bytes),
        used_bytes=int(result.error_used_bytes),
        requested_bytes=int(result.error_requested_bytes),
    )


def simulate_program(
    program: Program,
    schedule: MemorySchedule,
    *,
    selections: tuple[RecomputationSelection, ...] = (),
    config: SimulationConfig,
    admission: SimulationAdmission | None = None,
) -> SimulationResult:
    """Replay through `libshadowspill.so`."""

    projection = _project(program, schedule, selections, config, admission)
    return _simulate_projection(projection, schedule)


def simulate_template(
    template: IndexedSimulationTemplate,
    schedule: MemorySchedule,
    *,
    admission: SimulationAdmission | None = None,
) -> SimulationResult:
    """Replay a validated schedule using cached indexed program geometry."""

    return _simulate_projection(_bind_schedule(template, schedule, admission), schedule)


def simulate_template_summary(
    template: IndexedSimulationTemplate,
    schedule: MemorySchedule,
) -> IndexedSimulationSummary:
    """Replay a candidate without decoding its detailed interval report."""

    projection = _bind_schedule(template, schedule)
    _task_buffer, _transfer_buffer, _peak_buffer, result = _run_projection(
        projection,
        schedule,
    )
    return IndexedSimulationSummary(int(result.makespan_ns))


@dataclass(frozen=True, slots=True)
class IntervalArrays:
    """The simulator's own interval arrays, borrowed by compiled consumers.

    Handing these on costs nothing and saves a consumer working in index space
    from re-encoding the decoded intervals it would otherwise be given.
    """

    task_intervals: ctypes.Array[CTaskInterval]
    task_interval_count: int
    transfer_intervals: ctypes.Array[CTransferInterval]
    transfer_interval_count: int


def _run_projection(
    projection: _Projection,
    schedule: MemorySchedule,
) -> tuple[
    ctypes.Array[CTaskInterval],
    ctypes.Array[CTransferInterval],
    ctypes.Array[CDevicePeak],
    CResult,
]:
    task_buffer = (CTaskInterval * max(1, len(projection.task_ids)))()
    transfer_buffer = (CTransferInterval * max(1, len(schedule.actions)))()
    peak_buffer = (CDevicePeak * len(projection.device_ids))()
    result = CResult(
        task_intervals=task_buffer,
        task_interval_capacity=len(task_buffer),
        transfer_intervals=transfer_buffer,
        transfer_interval_capacity=len(transfer_buffer),
        device_peaks=peak_buffer,
        device_peak_capacity=len(peak_buffer),
    )
    library = simulator_api()
    status = int(
        library.shadowspill_simulate(
            ctypes.byref(projection.program),
            ctypes.byref(result),
        )
    )
    if status != 0:
        _raise_error(status, result, projection)
    return task_buffer, transfer_buffer, peak_buffer, result


def _simulate_projection(
    projection: _Projection,
    schedule: MemorySchedule,
) -> SimulationResult:
    task_buffer, transfer_buffer, peak_buffer, result = _run_projection(
        projection,
        schedule,
    )
    task_intervals = tuple(
        TaskInterval(
            task_id=projection.task_ids[item.task],
            device_id=projection.device_ids[projection.program.task_device[item.task]],
            resource_kind=projection.task_resources[item.task][0],
            resource_lane=projection.task_resources[item.task][1],
            ready_ns=int(item.ready_ns),
            start_ns=int(item.start_ns),
            end_ns=int(item.end_ns),
            workspace_bytes=int(item.workspace_bytes),
            stall_reasons=_stall_reasons(int(item.stall_mask)),
        )
        for item in sorted(
            task_buffer[: result.task_interval_count],
            key=lambda value: value.task,
        )
    )
    directions = {
        0: TransferDirection.FETCH,
        1: TransferDirection.EVICT,
    }
    transfer_intervals = tuple(
        TransferInterval(
            alias_group_id=projection.alias_ids[item.alias],
            trigger_task_id=projection.task_ids[item.trigger_task],
            device_id=projection.device_ids[item.device],
            direction=directions[item.direction],
            sequence=int(item.sequence),
            ready_ns=int(item.ready_ns),
            start_ns=int(item.start_ns),
            end_ns=int(item.end_ns),
            bytes=int(item.bytes),
            stall_reasons=_stall_reasons(int(item.stall_mask)),
        )
        for item in sorted(
            transfer_buffer[: result.transfer_interval_count],
            key=lambda value: (
                value.start_ns,
                directions[value.direction].value,
                projection.device_ids[value.device],
                value.sequence,
            ),
        )
    )
    device_peaks = tuple(
        DeviceMemoryPeak(
            device_id=device_id,
            object_bytes=(
                int(peak_buffer[index].object_bytes)
                + projection.shared_device_bytes[index]
            ),
            workspace_bytes=int(peak_buffer[index].workspace_bytes),
            total_bytes=(
                int(peak_buffer[index].total_bytes)
                + projection.shared_device_bytes[index]
            ),
        )
        for index, device_id in enumerate(projection.device_ids)
    )
    simulated = SimulationResult(
        makespan_ns=int(result.makespan_ns),
        task_intervals=task_intervals,
        transfer_intervals=transfer_intervals,
        device_peaks=device_peaks,
        spill_peak_bytes=int(result.spill_peak_bytes) + projection.shared_spill_bytes,
    )
    simulated.attach_interval_arrays(
        IntervalArrays(
            task_intervals=task_buffer,
            task_interval_count=int(result.task_interval_count),
            transfer_intervals=transfer_buffer,
            transfer_interval_count=int(result.transfer_interval_count),
        )
    )
    return simulated


__all__ = [
    "IndexedSimulationSummary",
    "IndexedSimulationTemplate",
    "IntervalArrays",
    "index_simulation_template",
    "simulate_program",
    "simulate_template",
    "simulate_template_summary",
]
