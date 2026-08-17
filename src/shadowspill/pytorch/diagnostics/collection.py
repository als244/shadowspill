"""Resolve one armed execution trace into immutable public diagnostics."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Mapping
from dataclasses import dataclass

from shadowspill.pytorch.diagnostics.timing import (
    ArmedExecutionTiming,
    ArmedTaskTiming,
)
from shadowspill.pytorch.runtime_adapter.abi import AdapterStatistics
from shadowspill.pytorch.runtime_adapter.bridge import (
    RuntimeBridge,
    TaskHostTimestamps,
)
from shadowspill.pytorch.runtime_adapter.trace import (
    CapturedRuntimeTrace,
    RuntimeTraceEvent,
    RuntimeTraceEventKind,
)
from shadowspill.simulator import TransferInterval

from ._mapping import FrozenMapping
from .execution import (
    AllocatorTrace,
    ExecutionTiming,
    PhaseTimingComparison,
    RuntimeTrace,
    SimulatorTaskComparison,
    SimulatorTransferComparison,
    StepDiagnostics,
    StepTimingSummary,
    TaskExecutionTiming,
    TransferTrace,
)


@dataclass(frozen=True, slots=True)
class _TraceEvidence:
    callbacks: Mapping[str, TaskHostTimestamps]
    native: CapturedRuntimeTrace
    statistics_before: AdapterStatistics
    statistics_after: AdapterStatistics


def collect_step_diagnostics(
    timing: ArmedExecutionTiming,
    bridge: RuntimeBridge,
) -> StepDiagnostics:
    """Synchronize and assemble all evidence for one traced real step."""

    _validate_completed_timing(timing)
    evidence = _resolve_trace_evidence(timing, bridge)
    tasks, phase_seconds = _build_task_timings(timing, evidence)
    tasks_by_execution_id = FrozenMapping(
        {item.execution_task_id: item for item in tasks}
    )
    execution_timing = _build_execution_timing(timing, tasks, phase_seconds)
    allocator = _build_allocator_trace(evidence)
    runtime = _build_runtime_trace(evidence)
    return StepDiagnostics(
        timing=execution_timing,
        tasks=tasks_by_execution_id,
        allocator=allocator,
        transfers=_build_transfer_trace(timing, evidence, bridge),
        runtime=runtime,
        simulator_comparison=_build_simulator_comparison(timing, tasks),
        summary=_build_step_summary(timing, tasks, execution_timing, runtime),
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
    """Drain the explicitly traced call before copying callback evidence."""

    timing.end_event.synchronize()
    if timing.stream is None:
        raise AssertionError("validated execution timing lost its compute stream")
    # The end event precedes after_task's completion callback.
    timing.stream.synchronize()
    try:
        callbacks = bridge.read_debug_task_timing()
    finally:
        bridge.disable_debug_task_timing()
    # Terminal transfers must be included in the same invocation trace.
    bridge.wait_idle()
    native = bridge.end_and_read_runtime_trace()
    statistics_before = timing.statistics_before
    if statistics_before is None:
        raise AssertionError("validated execution timing lost initial statistics")
    return _TraceEvidence(
        callbacks=callbacks,
        native=native,
        statistics_before=statistics_before,
        statistics_after=bridge.statistics(),
    )


def _build_task_timings(
    timing: ArmedExecutionTiming,
    evidence: _TraceEvidence,
) -> tuple[tuple[TaskExecutionTiming, ...], Mapping[str, float]]:
    tasks: list[TaskExecutionTiming] = []
    phase_seconds: dict[str, float] = {}
    for task_id in timing.task_order:
        task = timing.tasks[task_id]
        result = _build_task_timing(
            timing,
            task_id,
            task,
            evidence.callbacks.get(task_id),
            evidence.native.begin_timestamp_ns,
        )
        tasks.append(result)
        phase_seconds[result.phase] = (
            phase_seconds.get(result.phase, 0.0) + result.gpu_duration_seconds
        )
    return tuple(tasks), phase_seconds


def _build_task_timing(
    execution: ArmedExecutionTiming,
    task_id: str,
    task: ArmedTaskTiming,
    callback: TaskHostTimestamps | None,
    callback_origin_ns: int,
) -> TaskExecutionTiming:
    gpu_start = float(execution.start_event.elapsed_time(task.start_event)) / 1_000.0
    gpu_end = float(execution.start_event.elapsed_time(task.end_event)) / 1_000.0
    gpu_duration = float(task.start_event.elapsed_time(task.end_event)) / 1_000.0
    readiness_seconds = (
        float(execution.origin_event.elapsed_time(task.readiness_event)) / 1_000.0
    )
    compute_start_seconds = (
        float(execution.origin_event.elapsed_time(task.start_event)) / 1_000.0
    )
    compute_end_seconds = (
        float(execution.origin_event.elapsed_time(task.end_event)) / 1_000.0
    )
    sequence_base = task.execution_ordinal * 3
    return TaskExecutionTiming(
        task_id=task_id,
        execution_ordinal=task.execution_ordinal,
        execution_task_id=f"execution_{task.execution_ordinal:06d}",
        semantic_name=task.semantic_name,
        phase=task.entrypoint.phase,
        microbatch=task.entrypoint.microbatch,
        expected_profile_seconds=task.expected_profile_seconds,
        before_task_enter_timestamp_ns=(
            callback.before_task_enter_ns if callback is not None else 0
        ),
        before_task_exit_timestamp_ns=(
            callback.before_task_exit_ns if callback is not None else 0
        ),
        after_task_enter_timestamp_ns=(
            callback.after_task_enter_ns if callback is not None else 0
        ),
        after_task_exit_timestamp_ns=(
            callback.after_task_exit_ns if callback is not None else 0
        ),
        before_readiness_waits_timestamp_ns=int(readiness_seconds * 1e9),
        before_task_compute_timestamp_ns=int(compute_start_seconds * 1e9),
        after_task_compute_timestamp_ns=int(compute_end_seconds * 1e9),
        gpu_start_seconds=gpu_start,
        gpu_end_seconds=gpu_end,
        gpu_duration_seconds=gpu_duration,
        before_readiness_waits_seconds=readiness_seconds,
        before_task_compute_seconds=compute_start_seconds,
        after_task_compute_seconds=compute_end_seconds,
        readiness_wait_seconds=(
            float(task.readiness_event.elapsed_time(task.start_event)) / 1_000.0
        ),
        task_compute_seconds=gpu_duration,
        before_readiness_waits_sequence=sequence_base + 1,
        before_task_compute_sequence=sequence_base + 2,
        after_task_compute_sequence=sequence_base + 3,
        native_before_task_enter_seconds=_relative_seconds(
            callback.before_task_enter_ns if callback is not None else 0,
            callback_origin_ns,
        ),
        native_before_task_exit_seconds=_relative_seconds(
            callback.before_task_exit_ns if callback is not None else 0,
            callback_origin_ns,
        ),
        native_after_task_enter_seconds=_relative_seconds(
            callback.after_task_enter_ns if callback is not None else 0,
            callback_origin_ns,
        ),
        native_after_task_exit_seconds=_relative_seconds(
            callback.after_task_exit_ns if callback is not None else 0,
            callback_origin_ns,
        ),
        host_before_task_seconds=(task.host_before_finished_ns - task.host_started_ns)
        / 1e9,
        host_stream_resolution_seconds=task.host_stream_resolution_ns / 1e9,
        host_readiness_marker_seconds=task.host_readiness_marker_ns / 1e9,
        host_native_before_task_seconds=task.host_native_before_task_ns / 1e9,
        host_input_lookup_seconds=task.host_input_lookup_ns / 1e9,
        host_storage_rebind_seconds=task.host_storage_rebind_ns / 1e9,
        host_generation_publish_seconds=task.host_generation_publish_ns / 1e9,
        host_argument_assembly_seconds=task.host_argument_assembly_ns / 1e9,
        host_rebind_seconds=task.host_rebind_ns / 1e9,
        host_dispatch_seconds=task.host_dispatch_ns / 1e9,
        host_output_flatten_seconds=task.host_output_flatten_ns / 1e9,
        host_output_classification_seconds=task.host_output_classification_ns / 1e9,
        host_output_adoption_seconds=task.host_output_adoption_ns / 1e9,
        host_output_state_publish_seconds=task.host_output_state_publish_ns / 1e9,
        host_gradient_accumulation_seconds=(task.host_gradient_accumulation_ns / 1e9),
        host_output_publish_seconds=task.host_output_publish_ns / 1e9,
        host_dematerialize_seconds=task.host_dematerialize_ns / 1e9,
        host_postprocess_seconds=task.host_postprocess_ns / 1e9,
        host_native_after_task_seconds=task.host_native_after_task_ns / 1e9,
        host_cleanup_seconds=task.host_cleanup_ns / 1e9,
        host_after_task_seconds=(task.host_finished_ns - task.host_after_started_ns)
        / 1e9,
        host_total_seconds=(task.host_finished_ns - task.host_started_ns) / 1e9,
    )


def _relative_seconds(timestamp_ns: int, origin_ns: int) -> float | None:
    return (timestamp_ns - origin_ns) / 1e9 if timestamp_ns and origin_ns else None


def _build_execution_timing(
    timing: ArmedExecutionTiming,
    tasks: tuple[TaskExecutionTiming, ...],
    phase_seconds: Mapping[str, float],
) -> ExecutionTiming:
    optimizer = tuple(item for item in tasks if item.phase == "optimizer")
    optimizer_seconds = (
        max(item.gpu_end_seconds for item in optimizer)
        - min(item.gpu_start_seconds for item in optimizer)
        if optimizer
        else 0.0
    )
    return ExecutionTiming(
        compute_seconds=float(timing.start_event.elapsed_time(timing.end_event))
        / 1_000.0,
        optimizer_seconds=optimizer_seconds,
        host_call_seconds=(timing.host_call_finished_ns - timing.host_call_started_ns)
        / 1e9,
        host_startup_wait_seconds=timing.host_startup_wait_ns / 1e9,
        host_initial_actions_seconds=timing.host_initial_actions_ns / 1e9,
        trace_setup_seconds=timing.trace_setup_ns / 1e9,
        phase_gpu_seconds=tuple(sorted(phase_seconds.items())),
        tasks=FrozenMapping({item.execution_task_id: item for item in tasks}),
    )


def _build_allocator_trace(evidence: _TraceEvidence) -> AllocatorTrace:
    before = evidence.statistics_before.runtime
    after = evidence.statistics_after.runtime
    return AllocatorTrace(
        events=evidence.native.allocation_events,
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


def _build_transfer_trace(
    timing: ArmedExecutionTiming,
    evidence: _TraceEvidence,
    bridge: RuntimeBridge,
) -> TransferTrace:
    before = evidence.statistics_before.runtime
    after = evidence.statistics_after.runtime
    transfer_kinds = {
        RuntimeTraceEventKind.ACTION_QUEUED,
        RuntimeTraceEventKind.DESTINATION_RESERVED,
        RuntimeTraceEventKind.TRANSFER_DISPATCHED,
        RuntimeTraceEventKind.TRANSFER_COMPLETED,
    }
    selected_task_numbers = {
        _dense_id(task_id, "task_") for task_id in timing.task_order
    }
    initial_dispatches = tuple(
        item
        for item in evidence.native.events
        if item.kind is RuntimeTraceEventKind.TRANSFER_DISPATCHED
        and item.task_id not in selected_task_numbers
        and item.detail_0 == 0
    )
    return TransferTrace(
        actions=timing.actions,
        fetch_transfers=int(after.fetch_transfers - before.fetch_transfers),
        evict_transfers=int(after.evict_transfers - before.evict_transfers),
        bytes_fetched=int(after.bytes_fetched - before.bytes_fetched),
        bytes_evicted=int(after.bytes_evicted - before.bytes_evicted),
        initial_fetch_transfers=len(initial_dispatches),
        initial_bytes_fetched=sum(item.bytes for item in initial_dispatches),
        events=tuple(
            item for item in evidence.native.events if item.kind in transfer_kinds
        ),
        simulator_comparison=_build_transfer_comparison(
            timing, evidence.native.events, selected_task_numbers, bridge
        ),
    )


def _build_transfer_comparison(
    timing: ArmedExecutionTiming,
    events: tuple[RuntimeTraceEvent, ...],
    selected_task_numbers: set[int],
    bridge: RuntimeBridge,
) -> Mapping[str, SimulatorTransferComparison]:
    simulation = timing.simulation
    if simulation is None or not simulation.transfer_intervals:
        return FrozenMapping({})
    scheduled_events = tuple(
        item for item in events if item.task_id in selected_task_numbers
    )
    dispatches = _transfer_events_by_direction(
        scheduled_events, RuntimeTraceEventKind.TRANSFER_DISPATCHED
    )
    completions = _transfer_events_by_direction(
        scheduled_events, RuntimeTraceEventKind.TRANSFER_COMPLETED
    )
    queued = _transfer_boundary_events(
        scheduled_events, RuntimeTraceEventKind.ACTION_QUEUED
    )
    reserved = _transfer_boundary_events(
        scheduled_events, RuntimeTraceEventKind.DESTINATION_RESERVED
    )
    intervals = tuple(
        sorted(
            simulation.transfer_intervals,
            key=lambda item: (item.direction.value, item.sequence),
        )
    )
    simulated_origin_ns = min(item.start_ns for item in intervals)
    all_dispatches = tuple(item for lane in dispatches.values() for item in lane)
    if not all_dispatches:
        raise RuntimeError("runtime trace omitted every scheduled transfer dispatch")
    real_origin_ns = min(item.timestamp_ns for item in all_dispatches)
    execution_ids = {
        task_id: f"execution_{timing.tasks[task_id].execution_ordinal:06d}"
        for task_id in timing.task_order
    }
    result: dict[str, SimulatorTransferComparison] = {}
    for interval in intervals:
        direction = interval.direction.value
        lane_dispatches = dispatches[direction]
        lane_completions = completions[direction]
        if interval.sequence >= len(lane_dispatches) or (
            interval.sequence >= len(lane_completions)
        ):
            raise RuntimeError(
                f"runtime trace omitted {direction} transfer {interval.sequence}"
            )
        dispatch = lane_dispatches[interval.sequence]
        completion = lane_completions[interval.sequence]
        task_number = _dense_id(interval.trigger_task_id, "task_")
        object_number = bridge.runtime_object_id(interval.alias_group_id)
        _validate_transfer_event(interval, dispatch, task_number, object_number)
        _validate_transfer_event(interval, completion, task_number, object_number)
        action_kind = 2 if direction == "fetch" else 1
        key = (task_number, object_number, action_kind)
        queued_event = queued[key].popleft() if queued[key] else None
        reserved_event = reserved[key].popleft() if reserved[key] else None
        simulated_ready = (interval.ready_ns - simulated_origin_ns) / 1e9
        simulated_start = (interval.start_ns - simulated_origin_ns) / 1e9
        simulated_end = (interval.end_ns - simulated_origin_ns) / 1e9
        real_dispatch = (dispatch.timestamp_ns - real_origin_ns) / 1e9
        real_completion = (completion.timestamp_ns - real_origin_ns) / 1e9
        transfer_id = f"{direction}_{interval.sequence:06d}"
        result[transfer_id] = SimulatorTransferComparison(
            transfer_id=transfer_id,
            direction=direction,
            sequence=interval.sequence,
            trigger_task_id=interval.trigger_task_id,
            execution_task_id=execution_ids[interval.trigger_task_id],
            alias_group_id=interval.alias_group_id,
            bytes=interval.bytes,
            simulated_ready_seconds=simulated_ready,
            simulated_start_seconds=simulated_start,
            simulated_end_seconds=simulated_end,
            simulated_duration_seconds=(interval.end_ns - interval.start_ns) / 1e9,
            real_queued_seconds=_event_seconds(queued_event, real_origin_ns),
            real_reserved_seconds=_event_seconds(reserved_event, real_origin_ns),
            real_dispatch_timestamp_ns=dispatch.timestamp_ns,
            real_completion_timestamp_ns=completion.timestamp_ns,
            real_dispatch_seconds=real_dispatch,
            real_completion_seconds=real_completion,
            real_frontier_duration_seconds=(
                completion.timestamp_ns - dispatch.timestamp_ns
            )
            / 1e9,
            start_delta_seconds=real_dispatch - simulated_start,
            end_delta_seconds=real_completion - simulated_end,
            duration_delta_seconds=(
                (completion.timestamp_ns - dispatch.timestamp_ns) / 1e9
                - (interval.end_ns - interval.start_ns) / 1e9
            ),
        )
    return FrozenMapping(result)


def _transfer_events_by_direction(
    events: tuple[RuntimeTraceEvent, ...],
    kind: RuntimeTraceEventKind,
) -> dict[str, tuple[RuntimeTraceEvent, ...]]:
    grouped: dict[str, list[RuntimeTraceEvent]] = {"fetch": [], "evict": []}
    for event in events:
        if event.kind is kind and event.detail_0 in {0, 1}:
            grouped["fetch" if event.detail_0 == 0 else "evict"].append(event)
    return {name: tuple(items) for name, items in grouped.items()}


def _transfer_boundary_events(
    events: tuple[RuntimeTraceEvent, ...],
    kind: RuntimeTraceEventKind,
) -> defaultdict[tuple[int, int, int], deque[RuntimeTraceEvent]]:
    grouped: defaultdict[tuple[int, int, int], deque[RuntimeTraceEvent]] = (
        defaultdict(deque)
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


def _event_seconds(
    event: RuntimeTraceEvent | None, origin_ns: int
) -> float | None:
    return None if event is None else (event.timestamp_ns - origin_ns) / 1e9


def _dense_id(value: str, prefix: str) -> int:
    suffix = value.removeprefix(prefix)
    if not value.startswith(prefix) or not suffix.isdigit():
        raise RuntimeError(f"non-canonical dense identity {value!r}")
    return int(suffix)


def _build_runtime_trace(evidence: _TraceEvidence) -> RuntimeTrace:
    before = evidence.statistics_before
    after = evidence.statistics_after
    allocation_requests = int(after.allocation_callbacks - before.allocation_callbacks)
    zero_byte_requests = int(
        after.zero_size_allocation_callbacks - before.zero_size_allocation_callbacks
    )
    native = evidence.native
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
        event_queries=int(after.cuda.event_queries - before.cuda.event_queries),
        queued_actions_after=int(after.runtime.queued_actions),
        pending_retirements_after=int(after.runtime.pending_retirements),
        callback_failures_after=int(after.callback_failures),
        step_id=native.step_id,
        begin_timestamp_ns=native.begin_timestamp_ns,
        end_timestamp_ns=native.end_timestamp_ns,
        event_capacity=native.event_capacity,
        allocation_event_capacity=native.allocation_event_capacity,
        event_overflow=native.event_overflow,
        allocation_event_overflow=native.allocation_event_overflow,
        events=native.events,
    )


def _build_simulator_comparison(
    timing: ArmedExecutionTiming,
    tasks: tuple[TaskExecutionTiming, ...],
) -> Mapping[str, SimulatorTaskComparison]:
    simulation = timing.simulation
    if simulation is None:
        raise RuntimeError("execution trace omitted selected simulator evidence")
    intervals = {item.task_id: item for item in simulation.task_intervals}
    missing = tuple(item.task_id for item in tasks if item.task_id not in intervals)
    if missing:
        raise RuntimeError(f"simulator evidence omitted selected tasks: {missing!r}")
    simulated_origin_ns = min(intervals[item.task_id].start_ns for item in tasks)
    real_origin_seconds = min(item.gpu_start_seconds for item in tasks)
    return FrozenMapping(
        {
            item.execution_task_id: SimulatorTaskComparison(
                execution_task_id=item.execution_task_id,
                task_id=item.task_id,
                simulated_start_ns=intervals[item.task_id].start_ns,
                simulated_end_ns=intervals[item.task_id].end_ns,
                simulated_start_seconds=(
                    intervals[item.task_id].start_ns - simulated_origin_ns
                )
                / 1e9,
                real_start_seconds=item.gpu_start_seconds - real_origin_seconds,
                start_delta_seconds=(
                    item.gpu_start_seconds
                    - real_origin_seconds
                    - (intervals[item.task_id].start_ns - simulated_origin_ns) / 1e9
                ),
                simulated_end_seconds=(
                    intervals[item.task_id].end_ns - simulated_origin_ns
                )
                / 1e9,
                real_end_seconds=item.gpu_end_seconds - real_origin_seconds,
                end_delta_seconds=(
                    item.gpu_end_seconds
                    - real_origin_seconds
                    - (intervals[item.task_id].end_ns - simulated_origin_ns) / 1e9
                ),
                expected_profile_seconds=item.expected_profile_seconds,
                observed_gpu_seconds=item.gpu_duration_seconds,
                duration_delta_seconds=(
                    item.gpu_duration_seconds - item.expected_profile_seconds
                ),
            )
            for item in tasks
        }
    )


def _build_step_summary(
    timing: ArmedExecutionTiming,
    tasks: tuple[TaskExecutionTiming, ...],
    execution: ExecutionTiming,
    runtime: RuntimeTrace,
) -> StepTimingSummary:
    simulation = timing.simulation
    if simulation is None:
        raise RuntimeError("execution trace omitted selected simulator evidence")
    profiled_task_seconds = sum(item.expected_profile_seconds for item in tasks)
    real_task_seconds = sum(item.gpu_duration_seconds for item in tasks)
    simulated_intervals = {
        item.task_id: item for item in simulation.task_intervals
    }
    selected_intervals = tuple(
        simulated_intervals[item.task_id]
        for item in tasks
        if item.task_id in simulated_intervals
    )
    if len(selected_intervals) != len(tasks):
        missing = tuple(
            item.task_id for item in tasks if item.task_id not in simulated_intervals
        )
        raise RuntimeError(f"simulator evidence omitted selected tasks: {missing!r}")
    simulated_start_ns = min(item.start_ns for item in selected_intervals)
    simulated_end_ns = max(item.end_ns for item in selected_intervals)
    simulated_span_seconds = (simulated_end_ns - simulated_start_ns) / 1e9
    real_span_seconds = execution.compute_seconds
    phases = sorted({item.phase for item in tasks})
    phase_comparisons = tuple(
        PhaseTimingComparison(
            phase=phase,
            profiled_task_seconds=sum(
                item.expected_profile_seconds for item in tasks if item.phase == phase
            ),
            real_task_event_seconds=sum(
                item.gpu_duration_seconds for item in tasks if item.phase == phase
            ),
            delta_seconds=sum(
                item.gpu_duration_seconds - item.expected_profile_seconds
                for item in tasks
                if item.phase == phase
            ),
        )
        for phase in phases
    )
    simulated_gaps = max(0.0, simulated_span_seconds - profiled_task_seconds)
    real_gaps = max(0.0, real_span_seconds - real_task_seconds)
    makespan_seconds = simulation.makespan_ns / 1e9
    return StepTimingSummary(
        profiled_task_seconds=profiled_task_seconds,
        real_task_event_seconds=real_task_seconds,
        task_event_delta_seconds=real_task_seconds - profiled_task_seconds,
        simulated_inter_task_gap_seconds=simulated_gaps,
        real_inter_task_gap_seconds=real_gaps,
        inter_task_gap_delta_seconds=real_gaps - simulated_gaps,
        simulated_selected_span_seconds=simulated_span_seconds,
        real_selected_span_seconds=real_span_seconds,
        selected_span_delta_seconds=real_span_seconds - simulated_span_seconds,
        simulator_makespan_seconds=makespan_seconds,
        simulator_terminal_tail_seconds=max(
            0.0, makespan_seconds - simulated_end_ns / 1e9
        ),
        phase_comparisons=phase_comparisons,
        trace_complete=not (
            runtime.event_overflow or runtime.allocation_event_overflow
        ),
    )


__all__ = ["collect_step_diagnostics"]
