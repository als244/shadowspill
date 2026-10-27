"""Match ordered transfers to runtime events, preserving incomplete traces."""

from collections import defaultdict, deque

from shadowspill.diagnostics.step import TransferRecord
from shadowspill.diagnostics.timing import ArmedExecutionTiming
from shadowspill.ir.indexing import MEMORY_ACTION_CODE
from shadowspill.ir.program import canonical_index
from shadowspill.runtime.plan.bridge import RuntimeBridge
from shadowspill.runtime.trace import RuntimeTraceEvent, RuntimeTraceEventKind
from shadowspill.simulator import SimulationResult, TransferInterval

from .evidence import TraceEvidence
from .lanes import LaneRecords, lane_statistics, lane_summary

_DIRECTIONS = ("fetch", "evict")


def transfer_lanes(
    timing: ArmedExecutionTiming,
    simulation: SimulationResult,
    evidence: TraceEvidence,
    bridge: RuntimeBridge,
    origin_ns: int,
) -> dict[str, LaneRecords]:
    """Join every scheduled transfer with its trace events, lane by lane.

    A runtime trace that ran out of room holds only the step's first events,
    and joining those would hand each lane a summary of part of the step that
    reads as the whole of it. So an overflowed trace joins nothing: every
    scheduled transfer keeps its record, simulated side and relations, with
    every instant the trace would have supplied left `None`. In a complete
    trace a count that disagrees with the schedule is a real disagreement,
    and raises.
    """

    joined = not evidence.runtime_trace.event_overflow
    events = evidence.runtime_trace.events if joined else ()
    dispatches = _lane_events(events, RuntimeTraceEventKind.TRANSFER_DISPATCHED)
    completions = _lane_events(events, RuntimeTraceEventKind.TRANSFER_COMPLETED)
    queued = _boundary_events(events, RuntimeTraceEventKind.ACTION_QUEUED)
    reserved = _boundary_events(events, RuntimeTraceEventKind.DESTINATION_RESERVED)
    execution_ids = {
        task_id: f"execution_{timing.tasks[task_id].execution_ordinal:06d}"
        for task_id in timing.task_order
    }
    lanes: dict[str, LaneRecords] = {}
    for direction in _DIRECTIONS:
        intervals = sorted(
            (
                item
                for item in simulation.transfer_intervals
                if item.direction.value == direction
            ),
            key=lambda item: item.sequence,
        )
        lane_dispatches = dispatches[direction]
        lane_completions = completions[direction]
        if joined and not (
            len(intervals) == len(lane_dispatches) == len(lane_completions)
        ):
            raise RuntimeError(
                f"runtime/simulator {direction} transfer count mismatch: "
                f"scheduled={len(intervals)}, dispatched={len(lane_dispatches)}, "
                f"completed={len(lane_completions)}"
            )
        records: list[TransferRecord] = []
        for interval in intervals:
            relations = _object_relations(
                timing, interval.alias_group_id, interval.trigger_task_id
            )
            triggered_by = execution_ids[interval.trigger_task_id]
            if not joined:
                records.append(
                    _transfer_record(
                        interval,
                        direction,
                        triggered_by,
                        relations,
                        None,
                        None,
                        None,
                        None,
                        origin_ns,
                    )
                )
                continue
            dispatch = lane_dispatches[interval.sequence]
            completion = lane_completions[interval.sequence]
            task_number = canonical_index(interval.trigger_task_id, "task_")
            object_number = bridge.objects.runtime_object_id(interval.alias_group_id)
            _validate_transfer_event(interval, dispatch, task_number, object_number)
            _validate_transfer_event(interval, completion, task_number, object_number)
            key = (task_number, object_number, MEMORY_ACTION_CODE[interval.kind])
            queued_event = queued[key].popleft() if queued[key] else None
            reserved_event = reserved[key].popleft() if reserved[key] else None
            records.append(
                _transfer_record(
                    interval,
                    direction,
                    triggered_by,
                    relations,
                    dispatch,
                    completion,
                    queued_event,
                    reserved_event,
                    origin_ns,
                )
            )
        lanes[direction] = LaneRecords(
            records=tuple(records),
            summary=lane_summary(
                direction,
                tuple(records),
                lane_statistics(bridge, direction),
            ),
        )
    return lanes


def _object_relations(
    timing: ArmedExecutionTiming, alias_group_id: str, trigger_task_id: str
) -> tuple[str, str, str]:
    """The object's previous access, next access, and last modifier.

    All three are relative to the trigger task: the previous access and the
    modifier are the latest at or before it, the next access the earliest
    after it. `init` and `persistent` stand for none before and none after.
    """

    trigger = timing.tasks[trigger_task_id].execution_ordinal
    accesses = timing.alias_accesses.get(alias_group_id, ())
    before = [ordinal for ordinal, _is_write in accesses if ordinal <= trigger]
    after = [ordinal for ordinal, _is_write in accesses if ordinal > trigger]
    modifiers = [
        ordinal for ordinal, is_write in accesses if is_write and ordinal <= trigger
    ]
    return (
        f"execution_{max(before):06d}" if before else "init",
        f"execution_{min(after):06d}" if after else "persistent",
        f"execution_{max(modifiers):06d}" if modifiers else "init",
    )


def _transfer_record(
    interval: TransferInterval,
    direction: str,
    triggered_by: str,
    relations: tuple[str, str, str],
    dispatch: RuntimeTraceEvent | None,
    completion: RuntimeTraceEvent | None,
    queued: RuntimeTraceEvent | None,
    reserved: RuntimeTraceEvent | None,
    origin_ns: int,
) -> TransferRecord:
    simulated_start = interval.start_ns / 1e9
    simulated_end = interval.end_ns / 1e9
    lane_started = lane_finished = None
    start_delta = end_delta = None
    if (
        completion is not None
        and completion.lane_started_at_ns is not None
        and completion.lane_finished_at_ns is not None
    ):
        lane_started = completion.lane_started_at_ns / 1e9
        lane_finished = completion.lane_finished_at_ns / 1e9
        start_delta = lane_started - simulated_start
        end_delta = lane_finished - simulated_end
    # Reported on its own: a lane may know when it was handed a transfer and
    # not when the bytes moved, and the wait is worth having either way.
    lane_issued = (
        None
        if completion is None or completion.lane_issued_at_ns is None
        else completion.lane_issued_at_ns / 1e9
    )
    return TransferRecord(
        transfer_id=f"{direction}_{interval.sequence:06d}",
        direction=direction,
        sequence=interval.sequence,
        triggered_by=triggered_by,
        alias_group_id=interval.alias_group_id,
        bytes=interval.bytes,
        previous_access=relations[0],
        next_access=relations[1],
        modified_by=relations[2],
        simulated_ready_at_seconds=interval.ready_ns / 1e9,
        simulated_started_at_seconds=simulated_start,
        simulated_finished_at_seconds=simulated_end,
        lane_issued_at_seconds=lane_issued,
        lane_started_at_seconds=lane_started,
        lane_finished_at_seconds=lane_finished,
        start_delta_seconds=start_delta,
        end_delta_seconds=end_delta,
        queued_at_seconds=_event_seconds(queued, origin_ns),
        reserved_at_seconds=_event_seconds(reserved, origin_ns),
        dispatched_at_seconds=_event_seconds(dispatch, origin_ns),
        completion_observed_at_seconds=_event_seconds(completion, origin_ns),
    )


def _direction(event: RuntimeTraceEvent) -> str | None:
    return {0: "fetch", 1: "evict"}.get(event.detail_0)


def _lane_events(
    events: tuple[RuntimeTraceEvent, ...],
    kind: RuntimeTraceEventKind,
) -> dict[str, tuple[RuntimeTraceEvent, ...]]:
    grouped: dict[str, list[RuntimeTraceEvent]] = {name: [] for name in _DIRECTIONS}
    for event in events:
        direction = _direction(event)
        if event.kind is kind and direction is not None:
            grouped[direction].append(event)
    return {name: tuple(items) for name, items in grouped.items()}


def _boundary_events(
    events: tuple[RuntimeTraceEvent, ...],
    kind: RuntimeTraceEventKind,
) -> defaultdict[tuple[int, int, int], deque[RuntimeTraceEvent]]:
    grouped: defaultdict[tuple[int, int, int], deque[RuntimeTraceEvent]] = defaultdict(
        deque
    )
    for event in events:
        if (
            event.kind is kind
            and event.task_id is not None
            and event.object_id is not None
            and event.detail_0 in {1, 2}
        ):
            grouped[(event.task_id, event.object_id, event.detail_0)].append(event)
    return grouped


def _validate_transfer_event(
    interval: TransferInterval,
    event: RuntimeTraceEvent,
    task_number: int,
    object_number: int,
) -> None:
    if (
        event.task_id != task_number
        or event.object_id != object_number
        or event.bytes != interval.bytes
    ):
        raise RuntimeError(
            "runtime/simulator transfer identity mismatch: "
            f"expected task={task_number}, object={object_number}, "
            f"bytes={interval.bytes}; observed task={event.task_id}, "
            f"object={event.object_id}, bytes={event.bytes}"
        )


def _event_seconds(event: RuntimeTraceEvent | None, origin_ns: int) -> float | None:
    return None if event is None else (event.timestamp_ns - origin_ns) / 1e9
