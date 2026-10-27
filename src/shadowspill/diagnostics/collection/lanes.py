"""Summarize measured transfer durations, bytes, and runtime lane statistics."""

import ctypes
from dataclasses import dataclass

from shadowspill.diagnostics.step import LaneSummary, TransferRecord
from shadowspill.runtime.abi.statistics import LaneStatistics
from shadowspill.runtime.plan.bridge import RuntimeBridge


@dataclass(frozen=True, slots=True)
class LaneRecords:
    """A lane's records in FIFO order and its summary, before referencing."""

    records: tuple[TransferRecord, ...]
    summary: LaneSummary


def _lane_duration(record: TransferRecord) -> float | None:
    """The copy's measured time on its lane, or `None` if it was not traced."""

    if (
        record.lane_started_at_seconds is None
        or record.lane_finished_at_seconds is None
    ):
        return None
    return record.lane_finished_at_seconds - record.lane_started_at_seconds


def _simulated_duration(record: TransferRecord) -> float | None:
    """The lane time the simulator priced for this transfer."""

    if (
        record.simulated_started_at_seconds is None
        or record.simulated_finished_at_seconds is None
    ):
        return None
    return record.simulated_finished_at_seconds - record.simulated_started_at_seconds


def lane_statistics(bridge: RuntimeBridge, direction: str) -> LaneStatistics | None:
    """What the lane carrying one direction reports, or None if it reports none.

    The route is found by the pool pair it joins rather than by a name, because
    which pools a plan uses is the plan's choice and the registry is keyed by
    whatever the caller called them. A fetch runs spill to execution and an
    evict the other way, which is the whole of the direction.

    One call serves every kind of lane: the runtime resolves the route's lane
    and asks it through the contract, so nothing here knows which implementation
    answered. UNSUPPORTED is a lane declining to keep a count, and stays None
    rather than becoming zeroes a reader would take for "moved nothing".
    """

    source, destination = (
        (bridge.spill_pool_id, bridge.execution_pool_id)
        if direction == "fetch"
        else (bridge.execution_pool_id, bridge.spill_pool_id)
    )
    route_id = next(
        (
            route.route_id
            for route in bridge.runtime.routes.values()
            if route.source_pool_id == source
            and route.destination_pool_id == destination
        ),
        None,
    )
    if route_id is None:
        return None
    statistics = LaneStatistics()
    status = int(
        bridge.runtime_library.shadowspill_route_lane_statistics(
            bridge.runtime._runtime_handle, int(route_id), ctypes.byref(statistics)
        )
    )
    return statistics if status == 0 else None


def lane_summary(
    direction: str,
    records: tuple[TransferRecord, ...],
    lane_statistics: LaneStatistics | None,
) -> LaneSummary:
    measured = tuple(item for item in records if _lane_duration(item) is not None)
    lane_busy = sum(_lane_duration(item) or 0.0 for item in measured)
    measured_bytes = sum(item.bytes for item in measured)
    drift = max(
        (item for item in measured if item.start_delta_seconds is not None),
        key=lambda item: abs(item.start_delta_seconds or 0.0),
        default=None,
    )
    return LaneSummary(
        direction=direction,
        transfers=len(records),
        bytes=sum(item.bytes for item in records),
        simulated_busy_seconds=sum(
            _simulated_duration(item) or 0.0 for item in records
        ),
        measured_transfers=len(measured),
        lane_busy_seconds=lane_busy,
        effective_bandwidth_bytes_per_second=(
            measured_bytes / lane_busy if lane_busy > 0.0 else None
        ),
        largest_start_delta_seconds=(
            None if drift is None else drift.start_delta_seconds
        ),
        largest_start_delta_transfer_id=None if drift is None else drift.transfer_id,
        lane_statistics=lane_statistics,
    )
