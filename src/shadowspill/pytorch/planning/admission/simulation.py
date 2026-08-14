"""Translate timing-free physical admission into simulator accounting facts."""

from __future__ import annotations

from shadowspill.ir import MemorySchedule, Program, RecomputationSelection
from shadowspill.runtime import AdmissionReplayOperationKind
from shadowspill.simulator import (
    ActionPhysicalDelta,
    MemoryReuseDependency,
    SimulationAdmission,
    TaskPhysicalDelta,
)

from .admission_replay import (
    AdmissionReplay,
    AdmissionReplayPurpose,
    AdmissionReplayStep,
)

_ACQUIRE_OPERATIONS = frozenset(
    {
        AdmissionReplayOperationKind.ACQUIRE,
        AdmissionReplayOperationKind.RESERVE,
        AdmissionReplayOperationKind.ACQUIRE_RESERVED,
    }
)


def simulation_admission_from_replay(
    replay: AdmissionReplay,
    program: Program,
    schedule: MemorySchedule,
    *,
    selections: tuple[RecomputationSelection, ...] = (),
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
    task_deltas = {item.task_id: [0, 0] for item in tasks}
    action_deltas = {index: [0, 0] for index in range(len(schedule.actions))}
    initial_bytes = 0

    if len(replay.operations) != len(replay.pool.decisions):
        raise ValueError(
            "AdmissionReplay operation and decision counts do not match"
        )
    for step, decision in zip(
        replay.operations, replay.pool.decisions, strict=True
    ):
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
            _add_task_delta(task_deltas, step, delta)
        elif step.purpose in {
            AdmissionReplayPurpose.RELEASE,
            AdmissionReplayPurpose.EVICTION,
            AdmissionReplayPurpose.FETCH_DESTINATION,
        }:
            _add_action_delta(action_deltas, step, delta, completion=False)
        elif step.purpose is AdmissionReplayPurpose.TERMINAL_COMPLETION:
            _add_action_delta(action_deltas, step, delta, completion=True)
        else:  # pragma: no cover - enum exhaustiveness guard
            raise ValueError(f"unsupported admission purpose {step.purpose!r}")

    dependencies = tuple(
        MemoryReuseDependency(
            item.predecessor_action_index,
            successor_action_index=item.successor_action_index,
        )
        if item.successor_action_index is not None
        else MemoryReuseDependency(
            item.predecessor_action_index,
            successor_task_id=item.successor_task_id,
        )
        for item in replay.dependencies
    )
    result = SimulationAdmission(
        initial_physical_bytes=((device_id, initial_bytes),),
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


def _add_task_delta(
    totals: dict[str, list[int]],
    step: AdmissionReplayStep,
    delta: int,
) -> None:
    task_id = step.task_id
    if task_id is None or task_id not in totals:
        raise ValueError(
            f"{step.purpose.value} operation lacks a selected task identity"
        )
    boundary = (
        0 if step.operation.kind in _ACQUIRE_OPERATIONS else 1
    )
    totals[task_id][boundary] += delta


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
        item.start_bytes + item.completion_bytes
        for item in admission.task_deltas
    )
    net += sum(
        item.trigger_bytes + item.completion_bytes
        for item in admission.action_deltas
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
