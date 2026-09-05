"""Translate timing-free physical admission into simulator accounting facts."""

from __future__ import annotations

from shadowspill.ir import MemorySchedule, Program, TaskAlternativeChoice, TaskSpec
from shadowspill.planner.admission.admission_replay import (
    AdmissionReplay,
    AdmissionReplayPurpose,
    AdmissionReplayStep,
    CausalAdmissionDependency,
)
from shadowspill.runtime import AdmissionReplayOperationKind
from shadowspill.simulator import (
    ActionPhysicalDelta,
    MemoryReuseDependency,
    SimulationAdmission,
    TaskPhysicalDelta,
)


def simulation_admission_from_replay(
    replay: AdmissionReplay,
    program: Program,
    schedule: MemorySchedule,
    *,
    selections: tuple[TaskAlternativeChoice, ...] = (),
    device_capacity_bytes: int | None = None,
) -> SimulationAdmission:
    """Project exact pool deltas onto task and action timing boundaries."""

    tasks = program.selected_tasks(selections)
    if len(program.devices) != 1:
        raise ValueError(
            "one AdmissionReplay currently describes exactly one execution pool; "
            f"Program has {len(program.devices)} devices"
        )
    device_id = program.devices[0].device_id
    task_ids = {item.task_id for item in tasks}
    task_sequences: dict[str, list[int]] = {item.task_id: [] for item in tasks}
    action_deltas = {index: [0, 0] for index in range(len(schedule.actions))}
    initial_bytes = 0

    if len(replay.operations) != len(replay.pool.decisions):
        raise ValueError("AdmissionReplay operation and decision counts do not match")
    for step, decision in zip(replay.operations, replay.pool.decisions, strict=True):
        if decision.operation_index != step.operation.sequence:
            raise ValueError(
                "AdmissionReplay decision order does not match its operations"
            )
        delta = decision.physical_bytes_delta
        if step.purpose is AdmissionReplayPurpose.INITIAL_OBJECT:
            initial_bytes += delta
        elif step.purpose in {
            AdmissionReplayPurpose.TASK_WORKSPACE,
            AdmissionReplayPurpose.TASK_OUTPUT,
            AdmissionReplayPurpose.MUTATION_REPLACEMENT,
        }:
            _append_task_delta(task_sequences, step, delta)
        elif step.purpose in {
            AdmissionReplayPurpose.RELEASE,
            AdmissionReplayPurpose.EVICTION,
            AdmissionReplayPurpose.FETCH_DESTINATION,
        }:
            _add_action_delta(
                action_deltas,
                step,
                delta,
                completion=(
                    step.operation.kind
                    is AdmissionReplayOperationKind.COMPLETE_RETIREMENT
                ),
            )
        elif step.purpose is AdmissionReplayPurpose.TERMINAL_COMPLETION:
            _add_action_delta(action_deltas, step, delta, completion=True)
        else:  # pragma: no cover - enum exhaustiveness guard
            raise ValueError(f"unsupported admission purpose {step.purpose!r}")

    _validate_implicit_task_dependencies(replay, tasks, schedule)
    dependencies = tuple(
        _simulation_dependency(item)
        for item in replay.dependencies
        if item.predecessor_purpose is AdmissionReplayPurpose.EVICTION
    )
    task_deltas = {
        task_id: _task_peak_deltas(deltas) for task_id, deltas in task_sequences.items()
    }
    result = SimulationAdmission(
        initial_physical_bytes=((device_id, initial_bytes),),
        device_capacity_bytes=(
            ()
            if device_capacity_bytes is None
            else ((device_id, device_capacity_bytes),)
        ),
        task_deltas=tuple(
            TaskPhysicalDelta(task.task_id, *task_deltas[task.task_id])
            for task in tasks
        ),
        action_deltas=tuple(
            ActionPhysicalDelta(index, *action_deltas[index])
            for index in range(len(schedule.actions))
        ),
        reuse_dependencies=dependencies,
    )
    _validate_projection(replay, result, task_ids, len(schedule.actions))
    return result


def _validate_implicit_task_dependencies(
    replay: AdmissionReplay,
    tasks: tuple[TaskSpec, ...],
    schedule: MemorySchedule,
) -> None:
    task_order = {task.task_id: index for index, task in enumerate(tasks)}
    action_task_order = tuple(
        task_order[action.trigger_task_id] for action in schedule.actions
    )
    for dependency in replay.dependencies:
        if dependency.predecessor_purpose is AdmissionReplayPurpose.EVICTION:
            continue
        try:
            predecessor = task_order[dependency.predecessor_task_id]
        except KeyError as exc:
            raise ValueError(
                "admission dependency predecessor task is unknown"
            ) from exc
        if dependency.successor_action_index is not None:
            successor = action_task_order[dependency.successor_action_index]
        elif dependency.successor_task_id is not None:
            successor = task_order[dependency.successor_task_id]
        else:
            raise ValueError("admission dependency lacks its successor")
        if successor < predecessor:
            raise ValueError(
                "task-completion memory reuse points backward in execution order"
            )


def _simulation_dependency(
    dependency: CausalAdmissionDependency,
) -> MemoryReuseDependency:
    # Keep this helper narrow: task-completion predecessors are already
    # ordered by the compute stream and need no extra simulator edge.
    predecessor = dependency.predecessor_action_index
    if predecessor is None:
        raise ValueError("simulator dependency lacks its eviction action")
    if dependency.successor_action_index is not None:
        return MemoryReuseDependency(
            predecessor,
            successor_action_index=dependency.successor_action_index,
        )
    successor_task = dependency.successor_task_id
    if successor_task is None:
        raise ValueError("simulator dependency lacks its successor")
    return MemoryReuseDependency(predecessor, successor_task_id=successor_task)


def _append_task_delta(
    totals: dict[str, list[int]],
    step: AdmissionReplayStep,
    delta: int,
) -> None:
    task_id = step.task_id
    if task_id is None or task_id not in totals:
        raise ValueError(
            f"{step.purpose.value} operation lacks a selected task identity"
        )
    totals[task_id].append(delta)


def _task_peak_deltas(deltas: list[int]) -> list[int]:
    """Collapse an intra-task allocation trace to simulator boundaries."""

    current = 0
    peak = 0
    for delta in deltas:
        current += delta
        peak = max(peak, current)
    return [peak, current - peak]


def _add_action_delta(
    totals: dict[int, list[int]],
    step: AdmissionReplayStep,
    delta: int,
    *,
    completion: bool,
) -> None:
    action_index = step.action_index
    if action_index is None or action_index not in totals:
        raise ValueError(
            f"{step.purpose.value} operation lacks a schedule action identity"
        )
    totals[action_index][int(completion)] += delta


def _validate_projection(
    replay: AdmissionReplay,
    admission: SimulationAdmission,
    task_ids: set[str],
    action_count: int,
) -> None:
    initial = sum(value for _, value in admission.initial_physical_bytes)
    net = initial
    net += sum(
        item.start_bytes + item.completion_bytes for item in admission.task_deltas
    )
    net += sum(
        item.trigger_bytes + item.completion_bytes for item in admission.action_deltas
    )
    if net != replay.pool.final_allocated_bytes:
        raise ValueError(
            "simulator admission deltas do not reconcile with MemoryPool: "
            f"projected={net}, pool={replay.pool.final_allocated_bytes}"
        )
    for dependency in admission.reuse_dependencies:
        if dependency.predecessor_action_index >= action_count:
            raise ValueError("admission dependency predecessor action is unknown")
        if (
            dependency.successor_action_index is not None
            and dependency.successor_action_index >= action_count
        ):
            raise ValueError("admission dependency successor action is unknown")
        if (
            dependency.successor_task_id is not None
            and dependency.successor_task_id not in task_ids
        ):
            raise ValueError("admission dependency successor task is unknown")


__all__ = ["simulation_admission_from_replay"]
