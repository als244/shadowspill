"""Translate memory-reuse and computation dependencies into stream waits."""

from dataclasses import dataclass

from shadowspill.ir import ShadowSpillProgram
from shadowspill.ir.program import canonical_index
from shadowspill.planner.admission.admission_replay import AdmissionReplayPurpose
from shadowspill.planner.admission.layout.model import (
    NONE,
    PURPOSES,
    FixedPhysicalLayout,
)
from shadowspill.runtime.admission_capi import NO_ID
from shadowspill.runtime.fixed_layout import (
    RuntimeFixedDependency,
    RuntimePlacementKind,
)


@dataclass(frozen=True, slots=True)
class ActionIdentity:
    task_id: int
    ordinal: int
    object_id: int


def runtime_dependencies(
    layout: FixedPhysicalLayout,
    program: ShadowSpillProgram,
    actions: dict[int, ActionIdentity],
    allocations: dict[tuple[str, int], tuple[int, ...]],
) -> tuple[RuntimeFixedDependency, ...]:
    reuses = layout.reuse_dependencies
    tasks = {
        task.task_id: task for task in program.tasks if task.task_id in reuses.task_ids
    }
    result: set[RuntimeFixedDependency] = set()
    for task in tasks.values():
        for predecessor in task.dependencies:
            if predecessor not in tasks or tasks[predecessor].resource == task.resource:
                continue
            result.add(
                RuntimeFixedDependency(
                    canonical_index(predecessor, "task_"),
                    NO_ID,
                    canonical_index(task.task_id, "task_"),
                    NO_ID,
                    RuntimePlacementKind.TASK_START,
                )
            )
    eviction = PURPOSES.index(AdmissionReplayPurpose.EVICTION)
    for purpose, before_action, before_task, lease, after_task, after_action in zip(
        reuses.predecessor_purpose.tolist(),
        reuses.predecessor_action.tolist(),
        reuses.predecessor_task.tolist(),
        reuses.successor_lease_id.tolist(),
        reuses.successor_task.tolist(),
        reuses.successor_action.tolist(),
        strict=True,
    ):
        if purpose == eviction:
            if before_action == NONE:
                raise ValueError("fixed eviction dependency has no action identity")
            action = actions[before_action]
            predecessor_id, predecessor_ordinal = (
                action.task_id,
                action.ordinal,
            )
        else:
            if before_task == NONE:
                raise ValueError("fixed reuse dependency has no predecessor task")
            predecessor_id = canonical_index(reuses.task_ids[before_task], "task_")
            predecessor_ordinal = NO_ID
        if after_action != NONE:
            successor = actions[after_action]
            result.add(
                RuntimeFixedDependency(
                    predecessor_id,
                    predecessor_ordinal,
                    successor.task_id,
                    successor.ordinal,
                    RuntimePlacementKind.ACTION_DESTINATION,
                )
            )
            continue
        if after_task == NONE:
            raise ValueError("fixed reuse dependency has no successor")
        name = reuses.task_ids[after_task]
        if (
            purpose != eviction
            and tasks[reuses.task_ids[before_task]].resource == tasks[name].resource
        ):
            # Includes reuse within one task; its allocation contract handles it.
            continue
        ordinals = allocations.get((name, lease))
        if not ordinals:
            raise ValueError(f"fixed task lease {lease} has no ordinal")
        result.add(
            RuntimeFixedDependency(
                predecessor_id,
                predecessor_ordinal,
                canonical_index(name, "task_"),
                min(ordinals),
                RuntimePlacementKind.TASK_ALLOCATION,
            )
        )
    return tuple(result)
