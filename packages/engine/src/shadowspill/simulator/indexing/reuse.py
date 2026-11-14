"""Index task/eviction completion edges for physical address reuse."""

import ctypes

from shadowspill.ir import MemoryActionKind, MemorySchedule
from shadowspill.simulator.capi import NO_INDEX
from shadowspill.simulator.model import SimulationAdmission

from .arrays import _u32_array
from .template import IndexedSimulationTemplate


def reuse_arrays(
    template: IndexedSimulationTemplate,
    schedule: MemorySchedule,
    admission: SimulationAdmission,
) -> tuple[
    ctypes.Array[ctypes.c_uint32],
    ctypes.Array[ctypes.c_uint32],
    ctypes.Array[ctypes.c_uint32],
    ctypes.Array[ctypes.c_uint32],
]:
    """Index completion edges, preserving task and eviction predecessors."""

    predecessor_actions: list[int] = []
    predecessor_tasks: list[int] = []
    successor_tasks: list[int] = []
    successor_actions: list[int] = []
    for dependency in admission.reuse_dependencies:
        predecessor = dependency.predecessor_action_index
        if predecessor is not None and predecessor >= len(schedule.actions):
            raise ValueError(
                f"memory-reuse predecessor action is unknown: {predecessor}"
            )
        if (
            predecessor is not None
            and schedule.actions[predecessor].kind is not MemoryActionKind.EVICT
        ):
            raise ValueError(
                f"memory-reuse predecessor must be an EVICT action: {predecessor}"
            )
        predecessor_actions.append(NO_INDEX if predecessor is None else predecessor)
        if dependency.predecessor_task_id is None:
            predecessor_tasks.append(NO_INDEX)
        else:
            try:
                predecessor_tasks.append(
                    template.task_index[dependency.predecessor_task_id]
                )
            except KeyError as exc:
                raise ValueError(
                    "memory-reuse predecessor task is unknown: "
                    + dependency.predecessor_task_id
                ) from exc
        if dependency.successor_task_id is None:
            assert dependency.successor_action_index is not None
            if dependency.successor_action_index >= len(schedule.actions):
                raise ValueError(
                    "memory-reuse successor action is unknown: "
                    f"{dependency.successor_action_index}"
                )
            successor_tasks.append(NO_INDEX)
            successor_actions.append(dependency.successor_action_index)
        else:
            try:
                successor_tasks.append(
                    template.task_index[dependency.successor_task_id]
                )
            except KeyError as exc:
                raise ValueError(
                    "memory-reuse successor task is unknown: "
                    f"{dependency.successor_task_id!r}"
                ) from exc
            successor_actions.append(NO_INDEX)
    return (
        _u32_array(tuple(predecessor_actions)),
        _u32_array(tuple(predecessor_tasks)),
        _u32_array(tuple(successor_tasks)),
        _u32_array(tuple(successor_actions)),
    )
