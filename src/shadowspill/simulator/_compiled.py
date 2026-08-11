"""Canonical IR projection and result decoding for the compiled simulator."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass

from shadowspill.ir import (
    MemoryActionKind,
    MemoryLocation,
    MemorySchedule,
    Program,
    RecomputationSelection,
    ResourceKind,
)

from ._capi import (
    ABI_VERSION,
    NO_INDEX,
    CDevice,
    CDevicePeak,
    CProgram,
    CResult,
    CTaskInterval,
    CTransferInterval,
    load_simulator_library,
)
from .model import (
    DeviceMemoryPeak,
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
    MemoryLocation.HOST: 1,
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
)
_STATUS_KIND = {
    1: "invalid-argument",
    2: "allocation-failure",
    3: "initial-device-capacity",
    4: "initial-host-capacity",
    5: "task-input-deadlock",
    6: "task-device-capacity",
    7: "prefetch-device-capacity",
    8: "offload-host-capacity",
    9: "transfer-deadlock",
    10: "invalid-release",
    11: "release-transfer-conflict",
    12: "invalid-offload",
    13: "invalid-prefetch",
    14: "final-residency",
    15: "internal-error",
}


def _u32_array(values: tuple[int, ...]) -> ctypes.Array[ctypes.c_uint32]:
    array_type = ctypes.c_uint32 * max(1, len(values))
    return array_type(*values) if values else array_type()


def _u64_array(values: tuple[int, ...]) -> ctypes.Array[ctypes.c_uint64]:
    array_type = ctypes.c_uint64 * max(1, len(values))
    return array_type(*values) if values else array_type()


def _u8_array(values: tuple[int, ...]) -> ctypes.Array[ctypes.c_uint8]:
    array_type = ctypes.c_uint8 * max(1, len(values))
    return array_type(*values) if values else array_type()


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


def _project(
    program: Program,
    schedule: MemorySchedule,
    selections: tuple[RecomputationSelection, ...],
    config: SimulationConfig,
) -> _Projection:
    schedule.validate(program, selections)
    configured = {item.device_id: item for item in config.devices}
    device_ids = tuple(item.device_id for item in program.devices)
    if set(configured) != set(device_ids):
        raise ValueError(
            "simulation devices must exactly match Program devices; "
            f"expected {sorted(device_ids)}, got {sorted(configured)}"
        )
    alias_ids = tuple(item.alias_group_id for item in program.alias_groups)
    alias_index = {value: index for index, value in enumerate(alias_ids)}
    device_index = {value: index for index, value in enumerate(device_ids)}
    object_alias = {
        item.object_id: alias_index[item.alias_group_id] for item in program.objects
    }
    profiles = {item.profile_id: item for item in program.profiles}
    tasks = program.selected_tasks(selections)
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
                configured[device_id].capacity_bytes,
                configured[device_id].h2d_bandwidth_bytes_per_second,
                configured[device_id].d2h_bandwidth_bytes_per_second,
                configured[device_id].h2d_latency_ns,
                configured[device_id].d2h_latency_ns,
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
    alias_size = u64(tuple(item.size_bytes for item in program.alias_groups))
    alias_version = u64(tuple(item.initial_version for item in program.alias_groups))
    alias_host = u8(
        tuple(int(item.retain_host_backing) for item in program.alias_groups)
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
    action_tasks = u32(
        tuple(task_index[item.trigger_task_id] for item in schedule.actions)
    )
    action_aliases = u32(
        tuple(alias_index[item.alias_group_id] for item in schedule.actions)
    )
    action_kinds = u8(tuple(_ACTION_CODE[item.kind] for item in schedule.actions))
    initial_aliases = u32(
        tuple(alias_index[item.alias_group_id] for item in schedule.initial_residency)
    )
    initial_locations = u8(
        tuple(_LOCATION_CODE[item.location] for item in schedule.initial_residency)
    )
    final_aliases = u32(
        tuple(alias_index[item.alias_group_id] for item in schedule.final_residency)
    )
    final_locations = u8(
        tuple(_LOCATION_CODE[item.location] for item in schedule.final_residency)
    )
    c_program = CProgram(
        abi_version=ABI_VERSION,
        device_count=len(device_ids),
        alias_count=len(alias_ids),
        task_count=len(tasks),
        action_count=len(schedule.actions),
        initial_count=len(schedule.initial_residency),
        final_count=len(schedule.final_residency),
        dependency_count=len(dependency_values),
        input_count=len(input_values),
        output_count=len(output_values),
        mutation_count=len(mutation_values),
        host_capacity_bytes=config.host_capacity_bytes,
        devices=c_devices,
        alias_device=alias_device,
        alias_size_bytes=alias_size,
        alias_initial_version=alias_version,
        alias_retain_host_backing=alias_host,
        task_device=task_device,
        task_resource_kind=task_kind,
        task_resource_lane=task_lane,
        task_runtime_ns=task_runtime,
        task_workspace_bytes=task_workspace,
        dependency_offsets=dependency_offsets_buffer,
        dependencies=dependency_buffer,
        input_offsets=input_offsets_buffer,
        input_aliases=input_buffer,
        output_offsets=output_offsets_buffer,
        output_aliases=output_buffer,
        mutation_offsets=mutation_offsets_buffer,
        mutation_aliases=mutation_buffer,
        mutation_version_deltas=mutation_delta_buffer,
        action_trigger_tasks=action_tasks,
        action_aliases=action_aliases,
        action_kinds=action_kinds,
        initial_aliases=initial_aliases,
        initial_locations=initial_locations,
        final_aliases=final_aliases,
        final_locations=final_locations,
    )
    return _Projection(
        c_program,
        tuple(buffers),
        task_ids,
        alias_ids,
        device_ids,
        tuple((item.resource.kind, item.resource.lane) for item in tasks),
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
    library = load_simulator_library()
    encoded = library.shadowspill_simulation_status_string(status)
    message = encoded.decode("utf-8") if encoded else f"simulator status {status}"
    alias = _optional_name(projection.alias_ids, int(result.error_alias))
    raise SimulationInfeasibleError(
        message,
        kind=_STATUS_KIND.get(status, "unknown"),
        time_ns=int(result.error_time_ns),
        task_id=_optional_name(projection.task_ids, int(result.error_task)),
        alias_group_ids=(() if alias is None else (alias,)),
        location=(
            None
            if int(result.error_device) == NO_INDEX and status not in (4, 8, 14)
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


def simulate_compiled(
    program: Program,
    schedule: MemorySchedule,
    *,
    selections: tuple[RecomputationSelection, ...] = (),
    config: SimulationConfig,
) -> SimulationResult:
    """Replay through `libshadowspill_simulator.so`."""

    projection = _project(program, schedule, selections, config)
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
    library = load_simulator_library()
    status = int(
        library.shadowspill_simulate(
            ctypes.byref(projection.program),
            ctypes.byref(result),
        )
    )
    if status != 0:
        _raise_error(status, result, projection)
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
        0: TransferDirection.HOST_TO_DEVICE,
        1: TransferDirection.DEVICE_TO_HOST,
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
            object_bytes=int(peak_buffer[index].object_bytes),
            workspace_bytes=int(peak_buffer[index].workspace_bytes),
            total_bytes=int(peak_buffer[index].total_bytes),
        )
        for index, device_id in enumerate(projection.device_ids)
    )
    return SimulationResult(
        makespan_ns=int(result.makespan_ns),
        task_intervals=task_intervals,
        transfer_intervals=transfer_intervals,
        device_peaks=device_peaks,
        host_peak_bytes=int(result.host_peak_bytes),
    )


__all__ = ["simulate_compiled"]
