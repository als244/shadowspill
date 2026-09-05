"""Resolve one armed execution trace into immutable public diagnostics."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from itertools import pairwise

from shadowspill.planner.diagnostics.mapping import FrozenMapping
from shadowspill.pytorch.diagnostics.timing import (
    ArmedExecutionTiming,
    ArmedTaskTiming,
)
from shadowspill.pytorch.runtime_adapter.abi import AdapterStatistics
from shadowspill.pytorch.runtime_adapter.bridge import (
    RuntimeBridge,
)
from shadowspill.pytorch.runtime_adapter.trace import (
    CapturedRuntimeTrace,
    RuntimeTraceEvent,
    RuntimeTraceEventKind,
)
from shadowspill.simulator import SimulationResult, TaskInterval, TransferInterval

from .execution import (
    AllocatorTrace,
    LaneSummary,
    PhaseTimingComparison,
    RuntimeTrace,
    StepDiagnostics,
    StepTimingSummary,
    TaskRecord,
    Timelines,
    TransferLane,
    TransferRecord,
    TransferRecords,
)

_DIRECTIONS = ("fetch", "evict")
_ACTION_KINDS = {"fetch": 2, "evict": 1}


@dataclass(frozen=True, slots=True)
class _TraceEvidence:
    runtime_trace: CapturedRuntimeTrace
    statistics_before: AdapterStatistics
    statistics_after: AdapterStatistics


@dataclass(frozen=True, slots=True)
class _LaneRecords:
    """A lane's records in FIFO order and its summary, before referencing."""

    records: tuple[TransferRecord, ...]
    summary: LaneSummary


@dataclass(frozen=True, slots=True)
class _StreamTask:
    """One selected task's device markers, before the simulation is joined."""

    task: ArmedTaskTiming
    task_id: str
    reached: float
    started: float
    finished: float
    input_wait: float
    reuse_wait: float


def collect_step_diagnostics(
    timing: ArmedExecutionTiming,
    bridge: RuntimeBridge,
) -> StepDiagnostics:
    """Synchronize and assemble all evidence for one traced real step."""

    _validate_completed_timing(timing)
    evidence = _resolve_trace_evidence(timing, bridge)
    simulation = timing.simulation
    if simulation is None:
        raise RuntimeError("execution trace omitted selected simulator evidence")
    stream_tasks = tuple(_stream_task(timing, task_id) for task_id in timing.task_order)
    intervals = _selected_intervals(simulation, stream_tasks)
    simulated_origin_ns = min(item.start_ns for item in intervals.values())
    alignment = min(item.started for item in stream_tasks)
    origin_ns = evidence.runtime_trace.began_at_ns
    compute = tuple(
        _compute_record(
            item, intervals[item.task_id], simulated_origin_ns, alignment, origin_ns
        )
        for item in stream_tasks
    )
    lanes = _transfer_lanes(
        timing, simulation, evidence, bridge, simulated_origin_ns, alignment, origin_ns
    )
    runtime = _build_runtime_trace(evidence)
    transfers = TransferRecords(
        fetch=FrozenMapping(
            {item.transfer_id: item for item in lanes["fetch"].records}
        ),
        evict=FrozenMapping(
            {item.transfer_id: item for item in lanes["evict"].records}
        ),
    )
    timelines = Timelines(
        first_task_started_at_seconds=alignment,
        compute=tuple(item.execution_task_id for item in compute),
        fetch=TransferLane(
            order=tuple(item.transfer_id for item in lanes["fetch"].records),
            summary=lanes["fetch"].summary,
        ),
        evict=TransferLane(
            order=tuple(item.transfer_id for item in lanes["evict"].records),
            summary=lanes["evict"].summary,
        ),
    )
    return StepDiagnostics(
        summary=_build_step_summary(timing, simulation, compute, intervals, runtime),
        tasks=FrozenMapping({item.execution_task_id: item for item in compute}),
        transfers=transfers,
        timelines=timelines,
        allocator=_build_allocator_trace(evidence),
        runtime=runtime,
    )


def _validate_completed_timing(timing: ArmedExecutionTiming) -> None:
    if not timing.started:
        raise RuntimeError("no execution timing measurement has started")
    if not timing.finished:
        raise RuntimeError("the execution timing measurement has not finished")
    if timing.stream is None:
        raise RuntimeError("execution timing has no compute stream")
    if timing.statistics_before is None:
        raise RuntimeError("execution timing omitted its initial statistics")


def _resolve_trace_evidence(
    timing: ArmedExecutionTiming,
    bridge: RuntimeBridge,
) -> _TraceEvidence:
    """Drain the explicitly traced call before copying its evidence."""

    timing.end_event.synchronize()
    if timing.stream is None:
        raise AssertionError("validated execution timing lost its compute stream")
    timing.stream.synchronize()
    # Terminal transfers must be included in the same invocation trace.
    bridge.wait_idle()
    runtime_trace = bridge.end_and_read_runtime_trace()
    statistics_before = timing.statistics_before
    if statistics_before is None:
        raise AssertionError("validated execution timing lost initial statistics")
    return _TraceEvidence(
        runtime_trace=runtime_trace,
        statistics_before=statistics_before,
        statistics_after=bridge.statistics(),
    )


def _stream_task(execution: ArmedExecutionTiming, task_id: str) -> _StreamTask:
    task = execution.tasks[task_id]
    origin = execution.origin_event
    return _StreamTask(
        task=task,
        task_id=task_id,
        reached=float(origin.elapsed_time(task.readiness_event)) / 1e3,
        started=float(origin.elapsed_time(task.start_event)) / 1e3,
        finished=float(origin.elapsed_time(task.end_event)) / 1e3,
        input_wait=float(task.readiness_event.elapsed_time(task.inputs_ready_event))
        / 1e3,
        reuse_wait=float(task.inputs_ready_event.elapsed_time(task.start_event)) / 1e3,
    )


def _selected_intervals(
    simulation: SimulationResult, tasks: tuple[_StreamTask, ...]
) -> dict[str, TaskInterval]:
    intervals = {item.task_id: item for item in simulation.task_intervals}
    missing = tuple(item.task_id for item in tasks if item.task_id not in intervals)
    if missing:
        raise RuntimeError(f"simulator evidence omitted selected tasks: {missing!r}")
    return {item.task_id: intervals[item.task_id] for item in tasks}


def _compute_record(
    item: _StreamTask,
    interval: TaskInterval,
    simulated_origin_ns: int,
    alignment: float,
    origin_ns: int,
) -> TaskRecord:
    task = item.task
    simulated_start = (interval.start_ns - simulated_origin_ns) / 1e9
    simulated_end = (interval.end_ns - simulated_origin_ns) / 1e9
    return TaskRecord(
        execution_task_id=f"execution_{task.execution_ordinal:06d}",
        task_id=item.task_id,
        execution_ordinal=task.execution_ordinal,
        semantic_name=task.semantic_name,
        phase=task.entrypoint.phase,
        microbatch=task.entrypoint.microbatch,
        simulated_ready_at_seconds=(interval.ready_ns - simulated_origin_ns) / 1e9,
        simulated_started_at_seconds=simulated_start,
        simulated_finished_at_seconds=simulated_end,
        expected_profile_seconds=task.expected_profile_seconds,
        compute_reached_at_seconds=item.reached,
        compute_started_at_seconds=item.started,
        compute_finished_at_seconds=item.finished,
        input_readiness_wait_seconds=item.input_wait,
        allocation_reuse_wait_seconds=item.reuse_wait,
        start_delta_seconds=item.started - (simulated_start + alignment),
        end_delta_seconds=item.finished - (simulated_end + alignment),
        before_task_entered_at_seconds=_relative_seconds(
            task.before_task_enter_ns, origin_ns
        ),
        before_task_exited_at_seconds=_relative_seconds(
            task.before_task_exit_ns, origin_ns
        ),
        after_task_entered_at_seconds=_relative_seconds(
            task.after_task_enter_ns, origin_ns
        ),
        after_task_exited_at_seconds=_relative_seconds(
            task.after_task_exit_ns, origin_ns
        ),
        dispatch_input_lookup_seconds=task.dispatch_input_lookup_ns / 1e9,
        dispatch_storage_rebind_seconds=task.dispatch_storage_rebind_ns / 1e9,
        dispatch_input_acquire_seconds=task.dispatch_input_acquire_ns / 1e9,
        dispatch_allocation_reuse_seconds=task.dispatch_allocation_reuse_ns / 1e9,
        dispatch_argument_assembly_seconds=task.dispatch_argument_assembly_ns / 1e9,
        dispatch_output_flatten_seconds=task.dispatch_output_flatten_ns / 1e9,
        dispatch_output_classification_seconds=(
            task.dispatch_output_classification_ns / 1e9
        ),
        dispatch_output_adoption_seconds=task.dispatch_output_adoption_ns / 1e9,
        dispatch_output_state_publish_seconds=(
            task.dispatch_output_state_publish_ns / 1e9
        ),
        dispatch_output_publish_seconds=task.dispatch_output_publish_ns / 1e9,
        dispatch_dematerialize_seconds=task.dispatch_dematerialize_ns / 1e9,
        dispatch_cleanup_seconds=task.dispatch_cleanup_ns / 1e9,
    )


def _relative_seconds(timestamp_ns: int, origin_ns: int) -> float | None:
    return (timestamp_ns - origin_ns) / 1e9 if timestamp_ns and origin_ns else None


def _transfer_lanes(
    timing: ArmedExecutionTiming,
    simulation: SimulationResult,
    evidence: _TraceEvidence,
    bridge: RuntimeBridge,
    simulated_origin_ns: int,
    alignment: float,
    origin_ns: int,
) -> dict[str, _LaneRecords]:
    """Join every scheduled transfer with its trace events, lane by lane."""

    events = evidence.runtime_trace.events
    selected_task_numbers = {
        _plan_index(task_id, "task_") for task_id in timing.task_order
    }
    scheduled = tuple(item for item in events if item.task_id in selected_task_numbers)
    dispatches = _lane_events(scheduled, RuntimeTraceEventKind.TRANSFER_DISPATCHED)
    completions = _lane_events(scheduled, RuntimeTraceEventKind.TRANSFER_COMPLETED)
    queued = _boundary_events(scheduled, RuntimeTraceEventKind.ACTION_QUEUED)
    reserved = _boundary_events(scheduled, RuntimeTraceEventKind.DESTINATION_RESERVED)
    opening_events = tuple(
        item for item in events if item.task_id not in selected_task_numbers
    )
    opening_dispatches = _lane_events(
        opening_events, RuntimeTraceEventKind.TRANSFER_DISPATCHED
    )
    opening_completions = _lane_events(
        opening_events, RuntimeTraceEventKind.TRANSFER_COMPLETED
    )
    alias_by_object = {
        bridge.runtime_object_id(action.alias_group_id): action.alias_group_id
        for action in timing.actions
    }
    execution_ids = {
        task_id: f"execution_{timing.tasks[task_id].execution_ordinal:06d}"
        for task_id in timing.task_order
    }
    lanes: dict[str, _LaneRecords] = {}
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
        records: list[TransferRecord] = [
            _opening_record(
                timing,
                direction,
                index,
                dispatch,
                completion,
                alias_by_object,
                origin_ns,
            )
            for index, (dispatch, completion) in enumerate(
                zip(
                    opening_dispatches[direction],
                    opening_completions[direction],
                    strict=True,
                )
            )
        ]
        opening = tuple(records)
        for interval in intervals:
            if interval.sequence >= len(lane_dispatches) or (
                interval.sequence >= len(lane_completions)
            ):
                raise RuntimeError(
                    f"runtime trace omitted {direction} transfer {interval.sequence}"
                )
            dispatch = lane_dispatches[interval.sequence]
            completion = lane_completions[interval.sequence]
            task_number = _plan_index(interval.trigger_task_id, "task_")
            object_number = bridge.runtime_object_id(interval.alias_group_id)
            _validate_transfer_event(interval, dispatch, task_number, object_number)
            _validate_transfer_event(interval, completion, task_number, object_number)
            key = (task_number, object_number, _ACTION_KINDS[direction])
            queued_event = queued[key].popleft() if queued[key] else None
            reserved_event = reserved[key].popleft() if reserved[key] else None
            records.append(
                _transfer_record(
                    interval,
                    direction,
                    execution_ids[interval.trigger_task_id],
                    _object_relations(
                        timing, interval.alias_group_id, interval.trigger_task_id
                    ),
                    dispatch,
                    completion,
                    queued_event,
                    reserved_event,
                    simulated_origin_ns,
                    alignment,
                    origin_ns,
                )
            )
        lanes[direction] = _LaneRecords(
            records=tuple(records),
            summary=_lane_summary(direction, tuple(records), opening),
        )
    return lanes


def _opening_record(
    timing: ArmedExecutionTiming,
    direction: str,
    index: int,
    dispatch: RuntimeTraceEvent,
    completion: RuntimeTraceEvent,
    alias_by_object: dict[int, str],
    origin_ns: int,
) -> TransferRecord:
    """One transfer of the opening placement batch: measured, never simulated."""

    if dispatch.object_id is None or dispatch.object_id not in alias_by_object:
        raise RuntimeError(
            f"opening {direction} transfer {index} moved an object the plan"
            " did not name"
        )
    if completion.object_id != dispatch.object_id or completion.bytes != dispatch.bytes:
        raise RuntimeError(
            f"opening {direction} transfer {index} completed as a different copy"
        )
    alias_group_id = alias_by_object[dispatch.object_id]
    accesses = timing.alias_accesses.get(alias_group_id, ())
    lane_started = lane_finished = None
    if (
        completion.lane_started_at_ns is not None
        and completion.lane_finished_at_ns is not None
    ):
        lane_started = completion.lane_started_at_ns / 1e9
        lane_finished = completion.lane_finished_at_ns / 1e9
    return TransferRecord(
        transfer_id=f"{direction}_opening_{index:06d}",
        direction=direction,
        sequence=index,
        triggered_by="init",
        alias_group_id=alias_group_id,
        bytes=dispatch.bytes,
        previous_access="init",
        next_access=(
            f"execution_{min(ordinal for ordinal, _ in accesses):06d}"
            if accesses
            else "persistent"
        ),
        modified_by="init",
        simulated_ready_at_seconds=None,
        simulated_started_at_seconds=None,
        simulated_finished_at_seconds=None,
        lane_started_at_seconds=lane_started,
        lane_finished_at_seconds=lane_finished,
        start_delta_seconds=None,
        end_delta_seconds=None,
        queued_at_seconds=None,
        reserved_at_seconds=None,
        dispatched_at_seconds=(dispatch.timestamp_ns - origin_ns) / 1e9,
        completion_observed_at_seconds=(completion.timestamp_ns - origin_ns) / 1e9,
    )


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
    dispatch: RuntimeTraceEvent,
    completion: RuntimeTraceEvent,
    queued: RuntimeTraceEvent | None,
    reserved: RuntimeTraceEvent | None,
    simulated_origin_ns: int,
    alignment: float,
    origin_ns: int,
) -> TransferRecord:
    simulated_start = (interval.start_ns - simulated_origin_ns) / 1e9
    simulated_end = (interval.end_ns - simulated_origin_ns) / 1e9
    lane_started = lane_finished = None
    start_delta = end_delta = None
    if (
        completion.lane_started_at_ns is not None
        and completion.lane_finished_at_ns is not None
    ):
        lane_started = completion.lane_started_at_ns / 1e9
        lane_finished = completion.lane_finished_at_ns / 1e9
        start_delta = lane_started - (simulated_start + alignment)
        end_delta = lane_finished - (simulated_end + alignment)
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
        simulated_ready_at_seconds=(interval.ready_ns - simulated_origin_ns) / 1e9,
        simulated_started_at_seconds=simulated_start,
        simulated_finished_at_seconds=simulated_end,
        lane_started_at_seconds=lane_started,
        lane_finished_at_seconds=lane_finished,
        start_delta_seconds=start_delta,
        end_delta_seconds=end_delta,
        queued_at_seconds=_event_seconds(queued, origin_ns),
        reserved_at_seconds=_event_seconds(reserved, origin_ns),
        dispatched_at_seconds=(dispatch.timestamp_ns - origin_ns) / 1e9,
        completion_observed_at_seconds=(completion.timestamp_ns - origin_ns) / 1e9,
    )


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


def _lane_summary(
    direction: str,
    records: tuple[TransferRecord, ...],
    opening: tuple[TransferRecord, ...],
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
        opening_transfers=len(opening),
        opening_bytes=sum(item.bytes for item in opening),
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


def _plan_index(value: str, prefix: str) -> int:
    suffix = value.removeprefix(prefix)
    if not value.startswith(prefix) or not suffix.isdigit():
        raise RuntimeError(f"non-canonical indexed identity {value!r}")
    return int(suffix)


def _build_allocator_trace(evidence: _TraceEvidence) -> AllocatorTrace:
    before = evidence.statistics_before.runtime
    after = evidence.statistics_after.runtime
    return AllocatorTrace(
        events=evidence.runtime_trace.allocation_events,
        live_allocations_before=int(before.live_allocations),
        live_allocations_after=int(after.live_allocations),
        allocated_bytes_before=int(before.allocated_bytes),
        allocated_bytes_after=int(after.allocated_bytes),
        peak_allocated_bytes=int(after.peak_allocated_bytes),
        free_bytes_after=int(after.free_bytes),
        free_prefix_bytes_after=int(after.free_prefix_bytes),
        largest_free_range_bytes_after=int(after.largest_free_range_bytes),
        external_fragmentation_bytes_after=int(after.external_fragmentation_bytes),
        blocked_allocators_after=int(after.blocked_allocators),
        overflow=bool(after.allocation_event_overflow),
    )


def _build_runtime_trace(evidence: _TraceEvidence) -> RuntimeTrace:
    before = evidence.statistics_before
    after = evidence.statistics_after
    allocation_requests = int(after.allocation_callbacks - before.allocation_callbacks)
    zero_byte_requests = int(
        after.zero_size_allocation_callbacks - before.zero_size_allocation_callbacks
    )
    runtime_trace = evidence.runtime_trace
    return RuntimeTrace(
        wait_events_inserted=int(
            after.runtime.wait_events_inserted - before.runtime.wait_events_inserted
        ),
        allocation_requests=allocation_requests,
        zero_byte_allocation_requests=zero_byte_requests,
        materialized_allocation_requests=allocation_requests - zero_byte_requests,
        free_requests=int(after.free_callbacks - before.free_callbacks),
        record_stream_callbacks=int(
            after.record_stream_callbacks - before.record_stream_callbacks
        ),
        event_queries=int(after.backend.event_queries - before.backend.event_queries),
        queued_actions_after=int(after.runtime.queued_actions),
        pending_retirements_after=int(after.runtime.pending_retirements),
        callback_failures_after=int(after.callback_failures),
        step_id=runtime_trace.step_id,
        began_at_ns=runtime_trace.began_at_ns,
        ended_at_ns=runtime_trace.ended_at_ns,
        event_capacity=runtime_trace.event_capacity,
        allocation_event_capacity=runtime_trace.allocation_event_capacity,
        event_overflow=runtime_trace.event_overflow,
        allocation_event_overflow=runtime_trace.allocation_event_overflow,
        events=runtime_trace.events,
    )


def _idle_composition(
    tasks: tuple[TaskRecord, ...],
) -> tuple[float, float, float]:
    """Split compute-stream idle into waiting and not being reached.

    Between one task ending and the next computing, the stream does two
    things: it travels to the next task's readiness marker, then it waits
    there. The waiting has two causes -- the task's inputs are still being
    fetched, and the ranges its allocations will reuse are still owned by a
    transfer -- and both are counted here. The first is what dispatch costs,
    the rest is what residency costs, and together they are the whole idle:
    the span starts at the first task's compute, so nothing else fits between
    the two ends.

    The first task's wait is returned separately because it finishes where the
    span starts, which puts it outside every span-relative number here while
    still being time the step spent.
    """

    ordered = sorted(tasks, key=lambda item: item.compute_started_at_seconds)
    if not ordered:
        return 0.0, 0.0, 0.0
    readiness = 0.0
    dispatch = 0.0
    for previous, current in pairwise(ordered):
        readiness += (
            current.input_readiness_wait_seconds + current.allocation_reuse_wait_seconds
        )
        dispatch += max(
            0.0,
            current.compute_reached_at_seconds - previous.compute_finished_at_seconds,
        )
    first = (
        ordered[0].input_readiness_wait_seconds
        + ordered[0].allocation_reuse_wait_seconds
    )
    return readiness, dispatch, first


def _compute_duration(record: TaskRecord) -> float:
    """How long the task's kernels ran, between its two compute instants."""

    return record.compute_finished_at_seconds - record.compute_started_at_seconds


def _frontend_lead(record: TaskRecord) -> float | None:
    """How long after the frontend handed the task off the stream reached it."""

    if record.before_task_exited_at_seconds is None:
        return None
    return record.compute_reached_at_seconds - record.before_task_exited_at_seconds


def _build_step_summary(
    timing: ArmedExecutionTiming,
    simulation: SimulationResult,
    tasks: tuple[TaskRecord, ...],
    intervals: dict[str, TaskInterval],
    runtime: RuntimeTrace,
) -> StepTimingSummary:
    profiled_task_seconds = sum(item.expected_profile_seconds for item in tasks)
    real_task_seconds = sum(_compute_duration(item) for item in tasks)
    selected = tuple(intervals.values())
    simulated_start_ns = min(item.start_ns for item in selected)
    simulated_end_ns = max(item.end_ns for item in selected)
    simulated_span_seconds = (simulated_end_ns - simulated_start_ns) / 1e9
    real_span_seconds = float(timing.start_event.elapsed_time(timing.end_event)) / 1e3
    phases = sorted({item.phase for item in tasks})
    phase_comparisons = tuple(
        PhaseTimingComparison(
            phase=phase,
            profiled_task_seconds=sum(
                item.expected_profile_seconds for item in tasks if item.phase == phase
            ),
            real_task_event_seconds=sum(
                _compute_duration(item) for item in tasks if item.phase == phase
            ),
            delta_seconds=sum(
                _compute_duration(item) - item.expected_profile_seconds
                for item in tasks
                if item.phase == phase
            ),
        )
        for phase in phases
    )
    simulated_idle = max(0.0, simulated_span_seconds - profiled_task_seconds)
    real_idle = max(0.0, real_span_seconds - real_task_seconds)
    readiness, dispatch, initial_readiness = _idle_composition(tasks)
    leads = [
        lead for lead in (_frontend_lead(item) for item in tasks) if lead is not None
    ]
    optimizer = tuple(item for item in tasks if item.phase == "optimizer")
    makespan_seconds = simulation.makespan_ns / 1e9
    return StepTimingSummary(
        profiled_task_seconds=profiled_task_seconds,
        real_task_event_seconds=real_task_seconds,
        task_event_delta_seconds=real_task_seconds - profiled_task_seconds,
        simulated_inter_task_idle_seconds=simulated_idle,
        real_inter_task_idle_seconds=real_idle,
        inter_task_idle_delta_seconds=real_idle - simulated_idle,
        simulated_inter_task_readiness_wait_seconds=(
            sum(item.stall_ns for item in selected) / 1e9
        ),
        real_inter_task_readiness_wait_seconds=readiness,
        real_inter_task_exposed_overhead_seconds=dispatch,
        real_initial_readiness_wait_seconds=initial_readiness,
        real_minimum_frontend_lead_seconds=min(leads) if leads else 0.0,
        simulated_selected_span_seconds=simulated_span_seconds,
        real_selected_span_seconds=real_span_seconds,
        selected_span_delta_seconds=real_span_seconds - simulated_span_seconds,
        simulator_makespan_seconds=makespan_seconds,
        simulator_terminal_tail_seconds=max(
            0.0, makespan_seconds - simulated_end_ns / 1e9
        ),
        call_seconds=(
            timing.dispatch_call_finished_ns - timing.dispatch_call_started_ns
        )
        / 1e9,
        prior_invocation_drain_seconds=timing.prior_invocation_drain_ns / 1e9,
        initial_actions_seconds=timing.dispatch_initial_actions_ns / 1e9,
        trace_setup_seconds=timing.trace_setup_ns / 1e9,
        optimizer_span_seconds=(
            max(item.compute_finished_at_seconds for item in optimizer)
            - min(item.compute_started_at_seconds for item in optimizer)
            if optimizer
            else 0.0
        ),
        phase_comparisons=phase_comparisons,
        trace_complete=not (
            runtime.event_overflow or runtime.allocation_event_overflow
        ),
    )


__all__ = ["collect_step_diagnostics"]
