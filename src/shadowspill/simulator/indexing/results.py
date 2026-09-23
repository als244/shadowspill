"""The simulator's answer, decoded back into the caller's names."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass

from shadowspill.ir import (
    MemoryActionKind,
    MemoryLocation,
    MemorySchedule,
    ResourceKind,
)
from shadowspill.status import Status

from ..capi import (
    NO_INDEX,
    CCapacityViolation,
    CDevicePeak,
    CResult,
    CTaskInterval,
    CTransferInterval,
    simulator_api,
)
from ..diagnostics import simulation_failure_detail, simulation_status_kind
from ..model import (
    CapacityViolation,
    DeviceMemoryPeak,
    SimulationInfeasibleError,
    SimulationResult,
    TaskInterval,
    TransferDirection,
    TransferInterval,
)
from .binding import _Projection
from .template import IndexedSimulationTemplate

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
    (1 << 5, "lane-busy"),
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
class IntervalArrays:
    """The simulator's own interval arrays, borrowed by compiled consumers.

    Handing these on costs nothing and saves a consumer working in index space
    from re-encoding the decoded intervals it would otherwise be given.
    """

    task_intervals: ctypes.Array[CTaskInterval]
    task_interval_count: int
    transfer_intervals: ctypes.Array[CTransferInterval]
    transfer_interval_count: int


def interval_arrays_from_result(
    template: IndexedSimulationTemplate, result: SimulationResult
) -> IntervalArrays:
    """Re-encode a result's intervals in the simulator's own index space.

    A result the simulator produced carries its arrays already; one read back
    from a store does not, and its consumers index by task and by (direction,
    sequence) rather than by position, so the order here is immaterial.
    """

    stall_bits = {name: bit for bit, name in _STALL_REASONS}
    directions = {TransferDirection.FETCH: 0, TransferDirection.EVICT: 1}
    tasks = (CTaskInterval * max(1, len(result.task_intervals)))()
    for index, task in enumerate(result.task_intervals):
        tasks[index] = CTaskInterval(
            task=template.task_index[task.task_id],
            ready_ns=task.ready_ns,
            start_ns=task.start_ns,
            end_ns=task.end_ns,
            workspace_bytes=task.workspace_bytes,
            stall_mask=sum(stall_bits[name] for name in task.stall_reasons),
        )
    transfers = (CTransferInterval * max(1, len(result.transfer_intervals)))()
    for index, transfer in enumerate(result.transfer_intervals):
        transfers[index] = CTransferInterval(
            alias=template.alias_index[transfer.alias_group_id],
            trigger_task=template.task_index[transfer.trigger_task_id],
            device=template.device_ids.index(transfer.device_id),
            direction=directions[transfer.direction],
            kind=_ACTION_CODE[transfer.kind],
            sequence=transfer.sequence,
            ready_ns=transfer.ready_ns,
            start_ns=transfer.start_ns,
            end_ns=transfer.end_ns,
            bytes=transfer.bytes,
            stall_mask=sum(stall_bits[name] for name in transfer.stall_reasons),
        )
    return IntervalArrays(
        task_intervals=tasks,
        task_interval_count=len(result.task_intervals),
        transfer_intervals=transfers,
        transfer_interval_count=len(result.transfer_intervals),
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
            if int(result.error_device) == NO_INDEX
            and status
            not in (
                Status.INITIAL_SPILL_CAPACITY,
                Status.EVICT_SPILL_CAPACITY,
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


def _run_projection(
    projection: _Projection,
    schedule: MemorySchedule,
) -> tuple[
    ctypes.Array[CTaskInterval],
    ctypes.Array[CTransferInterval],
    ctypes.Array[CDevicePeak],
    ctypes.Array[CCapacityViolation],
    CResult,
]:
    task_buffer = (CTaskInterval * max(1, len(projection.task_ids)))()
    transfer_buffer = (CTransferInterval * max(1, len(schedule.actions)))()
    peak_buffer = (CDevicePeak * len(projection.device_ids))()
    # Recorded once per task launch and once per action that came up short,
    # so the plan's own size bounds how many there can be.
    violations = len(projection.task_ids) + len(schedule.actions)
    violation_buffer = (CCapacityViolation * max(1, violations))()
    result = CResult(
        task_intervals=task_buffer,
        task_interval_capacity=len(task_buffer),
        transfer_intervals=transfer_buffer,
        transfer_interval_capacity=len(transfer_buffer),
        device_peaks=peak_buffer,
        device_peak_capacity=len(peak_buffer),
        capacity_violations=violation_buffer,
        capacity_violation_capacity=violations,
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
    return task_buffer, transfer_buffer, peak_buffer, violation_buffer, result


def _simulate_projection(
    projection: _Projection,
    schedule: MemorySchedule,
) -> SimulationResult:
    task_buffer, transfer_buffer, peak_buffer, violation_buffer, result = (
        _run_projection(projection, schedule)
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
    kinds = {code: kind for kind, code in _ACTION_CODE.items()}
    transfer_intervals = tuple(
        TransferInterval(
            alias_group_id=projection.alias_ids[item.alias],
            trigger_task_id=projection.task_ids[item.trigger_task],
            device_id=projection.device_ids[item.device],
            direction=directions[item.direction],
            kind=kinds[item.kind],
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
    reported = int(result.capacity_violation_count)
    violations = tuple(
        CapacityViolation(
            reason=_VIOLATION_REASONS[int(item.reason)],
            location=_VIOLATION_LOCATIONS[int(item.location)],
            time_ns=int(item.time_ns),
            capacity_bytes=int(item.capacity_bytes),
            used_bytes=int(item.used_bytes),
            requested_bytes=int(item.requested_bytes),
            device_id=_optional_name(projection.device_ids, int(item.device))
            or "unknown",
            task_id=_optional_name(projection.task_ids, int(item.task)),
            alias_group_id=_optional_name(projection.alias_ids, int(item.alias)),
        )
        for item in violation_buffer[: min(reported, len(violation_buffer))]
    )
    simulated = SimulationResult(
        makespan_ns=int(result.makespan_ns),
        task_intervals=task_intervals,
        transfer_intervals=transfer_intervals,
        device_peaks=device_peaks,
        spill_peak_bytes=int(result.spill_peak_bytes) + projection.shared_spill_bytes,
        capacity_violations=violations,
        capacity_violation_count=reported,
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
