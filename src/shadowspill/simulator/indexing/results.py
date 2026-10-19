"""The simulator's answer, decoded back into the caller's names."""

from __future__ import annotations

import ctypes
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from shadowspill.ir import (
    MemorySchedule,
)
from shadowspill.ir.indexing import (
    MEMORY_ACTION_CODE as _ACTION_CODE,
)
from shadowspill.ir.indexing import RESOURCE_KIND_CODE
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

_STALL_REASONS = (
    (1 << 0, "input-residency"),
    (1 << 1, "device-capacity"),
    (1 << 2, "source-readiness"),
    (1 << 3, "host-capacity"),
    (1 << 4, "memory-reuse"),
    (1 << 5, "lane-busy"),
    (1 << 6, "transfer-resource"),
)
_VIOLATION_REASONS = (
    "initial-device-capacity",
    "initial-spill-capacity",
    "fetch-device-capacity",
    "evict-spill-capacity",
    "task-device-capacity",
)
_VIOLATION_LOCATIONS = ("device", "spill")


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
    #: The index space the arrays are in: the template the schedule was bound
    #: to. A consumer that reads the arrays reads them through this one.
    template: IndexedSimulationTemplate | None = None
    #: The decoded intervals as columns, in their decoded order, and the names
    #: the index columns point into: what a writer stores without building a
    #: record per interval. Arrays have no equality to compare by.
    columns: Mapping[str, np.ndarray] | None = field(default=None, compare=False)
    names: IntervalNames | None = field(default=None, compare=False)


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
        template=template,
    )


def _stall_reasons(mask: int) -> tuple[str, ...]:
    return tuple(name for bit, name in _STALL_REASONS if mask & bit)


@dataclass(frozen=True, slots=True)
class IntervalNames:
    """The names interval columns index: tasks, alias groups and devices."""

    task_ids: tuple[str, ...]
    alias_ids: tuple[str, ...]
    device_ids: tuple[str, ...]


#: Interval columns, by record and field, with their element types. Index
#: columns point into `IntervalNames`; `stall_mask` is the simulator's bits.
TASK_COLUMNS: tuple[tuple[str, type[np.generic]], ...] = (
    ("task", np.uint32),
    ("device", np.uint32),
    ("resource_kind", np.uint8),
    ("resource_lane", np.uint32),
    ("ready_ns", np.uint64),
    ("start_ns", np.uint64),
    ("end_ns", np.uint64),
    ("workspace_bytes", np.uint64),
    ("stall_mask", np.uint32),
)
TRANSFER_COLUMNS: tuple[tuple[str, type[np.generic]], ...] = (
    ("alias", np.uint32),
    ("trigger_task", np.uint32),
    ("device", np.uint32),
    ("direction", np.uint8),
    ("kind", np.uint8),
    ("sequence", np.uint32),
    ("ready_ns", np.uint64),
    ("start_ns", np.uint64),
    ("end_ns", np.uint64),
    ("bytes", np.uint64),
    ("stall_mask", np.uint32),
)
_DIRECTIONS = {0: TransferDirection.FETCH, 1: TransferDirection.EVICT}
_DIRECTION_CODE = {direction: code for code, direction in _DIRECTIONS.items()}
_KINDS = {code: kind for kind, code in _ACTION_CODE.items()}
_RESOURCES = {code: kind for kind, code in RESOURCE_KIND_CODE.items()}


def _interval_columns(
    task_buffer: ctypes.Array[CTaskInterval],
    task_count: int,
    transfer_buffer: ctypes.Array[CTransferInterval],
    transfer_count: int,
    projection: _Projection,
) -> dict[str, np.ndarray]:
    """The library's interval records as columns, in decoded order: tasks by
    task, and transfers by start, then the direction's spelling, then the
    device's identifier, then sequence."""

    tasks = np.ctypeslib.as_array(task_buffer)[:task_count]
    tasks = tasks[np.argsort(tasks["task"], kind="stable")]
    task_index = tasks["task"]
    task_device = (
        np.ctypeslib.as_array(
            projection.program.task_device, shape=(len(projection.task_ids),)
        )
        if projection.task_ids
        else np.zeros(0, dtype=np.uint32)
    )
    kinds = np.array(
        [RESOURCE_KIND_CODE[kind] for kind, _ in projection.task_resources],
        dtype=np.uint8,
    )
    lanes = np.array([lane for _, lane in projection.task_resources], dtype=np.uint32)
    columns = {
        "tasks.task": task_index,
        "tasks.device": task_device[task_index],
        "tasks.resource_kind": kinds[task_index],
        "tasks.resource_lane": lanes[task_index],
        **{
            f"tasks.{name}": tasks[name]
            for name in (
                "ready_ns",
                "start_ns",
                "end_ns",
                "workspace_bytes",
                "stall_mask",
            )
        },
    }
    transfers = np.ctypeslib.as_array(transfer_buffer)[:transfer_count]
    spelled = sorted(item.value for item in _DIRECTIONS.values())
    direction_rank = np.zeros(max(_DIRECTIONS) + 1, dtype=np.int64)
    for code, direction in _DIRECTIONS.items():
        direction_rank[code] = spelled.index(direction.value)
    device_rank = np.zeros(max(1, len(projection.device_ids)), dtype=np.int64)
    for rank, index in enumerate(
        sorted(range(len(projection.device_ids)), key=projection.device_ids.__getitem__)
    ):
        device_rank[index] = rank
    transfers = transfers[
        np.lexsort(
            (
                transfers["sequence"],
                device_rank[transfers["device"]],
                direction_rank[transfers["direction"]],
                transfers["start_ns"],
            )
        )
    ]
    columns.update(
        {f"transfers.{name}": transfers[name] for name, _ in TRANSFER_COLUMNS}
    )
    return {
        name: np.ascontiguousarray(values, dtype=dtype)
        for name, values in columns.items()
        for dtype in (_COLUMN_TYPES[name],)
    }


_COLUMN_TYPES = {
    **{f"tasks.{name}": dtype for name, dtype in TASK_COLUMNS},
    **{f"transfers.{name}": dtype for name, dtype in TRANSFER_COLUMNS},
}


def intervals_from_columns(
    columns: Mapping[str, np.ndarray], names: IntervalNames
) -> tuple[tuple[TaskInterval, ...], tuple[TransferInterval, ...]]:
    """Interval records from columns already in decoded order, named."""

    task_intervals = tuple(
        map(
            TaskInterval,
            _names(names.task_ids, columns["tasks.task"]),
            _names(names.device_ids, columns["tasks.device"]),
            [_RESOURCES[code] for code in columns["tasks.resource_kind"].tolist()],
            columns["tasks.resource_lane"].tolist(),
            columns["tasks.ready_ns"].tolist(),
            columns["tasks.start_ns"].tolist(),
            columns["tasks.end_ns"].tolist(),
            columns["tasks.workspace_bytes"].tolist(),
            _stall_reason_lists(columns["tasks.stall_mask"]),
        )
    )
    transfer_intervals = tuple(
        map(
            TransferInterval,
            _names(names.alias_ids, columns["transfers.alias"]),
            _names(names.task_ids, columns["transfers.trigger_task"]),
            _names(names.device_ids, columns["transfers.device"]),
            [_DIRECTIONS[code] for code in columns["transfers.direction"].tolist()],
            [_KINDS[code] for code in columns["transfers.kind"].tolist()],
            columns["transfers.sequence"].tolist(),
            columns["transfers.ready_ns"].tolist(),
            columns["transfers.start_ns"].tolist(),
            columns["transfers.end_ns"].tolist(),
            columns["transfers.bytes"].tolist(),
            _stall_reason_lists(columns["transfers.stall_mask"]),
        )
    )
    return task_intervals, transfer_intervals


def interval_columns(
    result: SimulationResult,
) -> tuple[dict[str, np.ndarray], IntervalNames]:
    """A result's intervals as columns and the names they index: the columns
    the simulator decoded them from when it produced the result, else built
    from the records."""

    arrays = result.interval_arrays
    columns = getattr(arrays, "columns", None)
    names = getattr(arrays, "names", None)
    if columns is not None and names is not None:
        return dict(columns), names
    tasks = result.task_intervals
    transfers = result.transfer_intervals
    task_ids = tuple(
        dict.fromkeys(
            [item.task_id for item in tasks]
            + [item.trigger_task_id for item in transfers]
        )
    )
    alias_ids = tuple(dict.fromkeys(item.alias_group_id for item in transfers))
    device_ids = tuple(
        dict.fromkeys(
            [item.device_id for item in tasks] + [item.device_id for item in transfers]
        )
    )
    task_at = {name: index for index, name in enumerate(task_ids)}
    alias_at = {name: index for index, name in enumerate(alias_ids)}
    device_at = {name: index for index, name in enumerate(device_ids)}
    bits = {name: bit for bit, name in _STALL_REASONS}
    built = {
        "tasks.task": [task_at[item.task_id] for item in tasks],
        "tasks.device": [device_at[item.device_id] for item in tasks],
        "tasks.resource_kind": [
            RESOURCE_KIND_CODE[item.resource_kind] for item in tasks
        ],
        "tasks.resource_lane": [item.resource_lane for item in tasks],
        "tasks.ready_ns": [item.ready_ns for item in tasks],
        "tasks.start_ns": [item.start_ns for item in tasks],
        "tasks.end_ns": [item.end_ns for item in tasks],
        "tasks.workspace_bytes": [item.workspace_bytes for item in tasks],
        "tasks.stall_mask": [
            sum(bits[name] for name in item.stall_reasons) for item in tasks
        ],
        "transfers.alias": [alias_at[item.alias_group_id] for item in transfers],
        "transfers.trigger_task": [task_at[item.trigger_task_id] for item in transfers],
        "transfers.device": [device_at[item.device_id] for item in transfers],
        "transfers.direction": [_DIRECTION_CODE[item.direction] for item in transfers],
        "transfers.kind": [_ACTION_CODE[item.kind] for item in transfers],
        "transfers.sequence": [item.sequence for item in transfers],
        "transfers.ready_ns": [item.ready_ns for item in transfers],
        "transfers.start_ns": [item.start_ns for item in transfers],
        "transfers.end_ns": [item.end_ns for item in transfers],
        "transfers.bytes": [item.bytes for item in transfers],
        "transfers.stall_mask": [
            sum(bits[name] for name in item.stall_reasons) for item in transfers
        ],
    }
    return (
        {
            name: np.array(values, dtype=_COLUMN_TYPES[name])
            for name, values in built.items()
        },
        IntervalNames(task_ids, alias_ids, device_ids),
    )


def _stall_reason_lists(masks: np.ndarray) -> list[tuple[str, ...]]:
    """The reasons of each mask, spelled once per distinct mask."""

    distinct, positions = np.unique(masks, return_inverse=True)
    spelled = [_stall_reasons(int(mask)) for mask in distinct.tolist()]
    return [spelled[position] for position in positions.tolist()]


def _names(table: tuple[object, ...], indices: np.ndarray) -> list[object]:
    """`table[index]` for every index, looked up in one array operation."""

    lookup = np.empty(len(table), dtype=object)
    for position, value in enumerate(table):
        lookup[position] = value
    return lookup[indices].tolist()


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
    columns = _interval_columns(
        task_buffer,
        int(result.task_interval_count),
        transfer_buffer,
        int(result.transfer_interval_count),
        projection,
    )
    names = IntervalNames(
        projection.task_ids, projection.alias_ids, projection.device_ids
    )
    task_intervals, transfer_intervals = intervals_from_columns(columns, names)
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
            template=projection.template,
            columns=columns,
            names=names,
        )
    )
    return simulated
