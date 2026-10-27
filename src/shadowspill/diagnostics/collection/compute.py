"""Join task markers and frontend timestamps with simulated task intervals."""

from dataclasses import dataclass

from shadowspill.diagnostics.step import TaskRecord
from shadowspill.diagnostics.timing import ArmedExecutionTiming, ArmedTaskTiming
from shadowspill.runtime.timing import Marker, nanoseconds_between
from shadowspill.simulator import SimulationResult, TaskInterval


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


def seconds_between(earlier: Marker, later: Marker) -> float:
    """The seconds from one instant to another, both already reached."""

    return nanoseconds_between(earlier, later) / 1e9


def stream_task(execution: ArmedExecutionTiming, task_id: str) -> _StreamTask:
    task = execution.tasks[task_id]
    origin = execution.origin_event
    return _StreamTask(
        task=task,
        task_id=task_id,
        reached=seconds_between(origin, task.readiness_event),
        started=seconds_between(origin, task.start_event),
        finished=seconds_between(origin, task.end_event),
        input_wait=seconds_between(task.readiness_event, task.inputs_ready_event),
        reuse_wait=seconds_between(task.inputs_ready_event, task.start_event),
    )


def selected_intervals(
    simulation: SimulationResult, tasks: tuple[_StreamTask, ...]
) -> dict[str, TaskInterval]:
    intervals = {item.task_id: item for item in simulation.task_intervals}
    missing = tuple(item.task_id for item in tasks if item.task_id not in intervals)
    if missing:
        raise RuntimeError(f"simulator evidence omitted selected tasks: {missing!r}")
    return {item.task_id: intervals[item.task_id] for item in tasks}


def compute_record(
    item: _StreamTask,
    interval: TaskInterval,
    origin_ns: int,
) -> TaskRecord:
    task = item.task
    simulated_start = interval.start_ns / 1e9
    simulated_end = interval.end_ns / 1e9
    return TaskRecord(
        execution_task_id=f"execution_{task.execution_ordinal:06d}",
        task_id=item.task_id,
        execution_ordinal=task.execution_ordinal,
        semantic_name=task.semantic_name,
        phase=task.entrypoint.options.phase,
        microbatch=task.entrypoint.options.repetition,
        simulated_ready_at_seconds=interval.ready_ns / 1e9,
        simulated_started_at_seconds=simulated_start,
        simulated_finished_at_seconds=simulated_end,
        expected_profile_seconds=task.expected_profile_seconds,
        compute_reached_at_seconds=item.reached,
        compute_started_at_seconds=item.started,
        compute_finished_at_seconds=item.finished,
        input_readiness_wait_seconds=item.input_wait,
        allocation_reuse_wait_seconds=item.reuse_wait,
        start_delta_seconds=item.started - simulated_start,
        end_delta_seconds=item.finished - simulated_end,
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
