"""Compare measured and simulated invocation spans, idle time, and phases."""

from itertools import pairwise

from shadowspill.diagnostics.step import (
    PhaseTimingComparison,
    RuntimeTrace,
    StepTimingSummary,
    TaskRecord,
)
from shadowspill.diagnostics.timing import ArmedExecutionTiming
from shadowspill.simulator import SimulationResult, TaskInterval

from .compute import seconds_between
from .lanes import LaneRecords


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


def build_step_summary(
    timing: ArmedExecutionTiming,
    simulation: SimulationResult,
    tasks: tuple[TaskRecord, ...],
    intervals: dict[str, TaskInterval],
    runtime: RuntimeTrace,
    lanes: dict[str, LaneRecords],
) -> StepTimingSummary:
    profiled_task_seconds = sum(item.expected_profile_seconds for item in tasks)
    real_task_seconds = sum(_compute_duration(item) for item in tasks)
    selected = tuple(intervals[item.task_id] for item in tasks)
    simulated_start_ns = min(item.start_ns for item in selected)
    simulated_end_ns = max(item.end_ns for item in selected)
    simulated_span_seconds = (simulated_end_ns - simulated_start_ns) / 1e9
    real_start = min(item.compute_started_at_seconds for item in tasks)
    real_end = max(item.compute_finished_at_seconds for item in tasks)
    real_span_seconds = real_end - real_start
    transfers = tuple(item for lane in lanes.values() for item in lane.records)
    all_transfers_timed = all(
        item.lane_finished_at_seconds is not None for item in transfers
    )
    real_invocation = (
        max(
            (item.lane_finished_at_seconds or 0.0 for item in transfers),
            default=real_end,
        )
        if all_transfers_timed
        else None
    )
    if real_invocation is not None:
        real_invocation = max(real_end, real_invocation)
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
    simulated_task_seconds = sum(item.end_ns - item.start_ns for item in selected) / 1e9
    simulated_idle = max(0.0, simulated_span_seconds - simulated_task_seconds)
    real_idle = max(0.0, real_span_seconds - real_task_seconds)
    readiness, dispatch, initial_readiness = _idle_composition(tasks)
    leads = [
        lead for lead in (_frontend_lead(item) for item in tasks) if lead is not None
    ]
    optimizer = tuple(item for item in tasks if item.phase == "optimizer")
    makespan_seconds = simulation.makespan_ns / 1e9
    cycle_seconds: float | None = None
    exposed_tail_seconds: float | None = None
    entry_delay_seconds = real_start
    timeline = timing.timeline
    if timeline is not None and timeline.successor is not None:
        successor = timeline.successor
        successor.wait()
        cycle_seconds = seconds_between(timeline.origin, successor)
        exposed_tail_seconds = seconds_between(timeline.span_end, successor)
    return StepTimingSummary(
        profiled_task_seconds=profiled_task_seconds,
        real_task_event_seconds=real_task_seconds,
        task_event_delta_seconds=real_task_seconds - profiled_task_seconds,
        simulated_inter_task_idle_seconds=simulated_idle,
        real_inter_task_idle_seconds=real_idle,
        inter_task_idle_delta_seconds=real_idle - simulated_idle,
        simulated_inter_task_readiness_wait_seconds=(
            sum(item.stall_ns for item in selected[1:]) / 1e9
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
        simulated_entry_delay_seconds=simulated_start_ns / 1e9,
        real_invocation_seconds=real_invocation,
        real_terminal_tail_seconds=(
            None if real_invocation is None else real_invocation - real_end
        ),
        cycle_seconds=cycle_seconds,
        entry_delay_seconds=entry_delay_seconds,
        exposed_tail_seconds=exposed_tail_seconds,
        call_seconds=(
            timing.dispatch_call_finished_ns - timing.dispatch_call_started_ns
        )
        / 1e9,
        prior_invocation_drain_seconds=timing.prior_invocation_drain_ns / 1e9,
        trace_setup_seconds=timing.trace_setup_ns / 1e9,
        optimizer_span_seconds=(
            max(item.compute_finished_at_seconds for item in optimizer)
            - min(item.compute_started_at_seconds for item in optimizer)
            if optimizer
            else 0.0
        ),
        phase_comparisons=phase_comparisons,
        trace_complete=not (
            runtime.event_overflow
            or not timing.collectives.complete
            or runtime.allocation_event_overflow
            or not all_transfers_timed
        ),
    )
