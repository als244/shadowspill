"""Selected tasks and storage accesses needed to reconcile a runtime trace."""

from collections.abc import Iterable
from dataclasses import dataclass

from shadowspill.ir import MemoryAction, ShadowSpillProgram, TaskSpec
from shadowspill.planner.diagnostics.mapping import FrozenMapping
from shadowspill.simulator import SimulationResult
from shadowspill.task.entrypoints import TaskEntrypoint


@dataclass(frozen=True, slots=True)
class TracedTask:
    """One selected task, under the three names a trace reports it by."""

    entrypoint: TaskEntrypoint
    expected_profile_seconds: float
    execution_ordinal: int
    semantic_name: str


@dataclass(frozen=True, slots=True)
class TracedInvocation:
    """One invocation as a trace needs to see it, whatever runs it.

    Actions come directly from the schedule, including the fetches triggered
    by the invocation's start control task.
    """

    tasks: tuple[TracedTask, ...]
    actions: tuple[MemoryAction, ...]
    simulation: SimulationResult
    alias_accesses: FrozenMapping[str, tuple[tuple[int, bool], ...]]


def alias_accesses(
    program: ShadowSpillProgram,
    tasks: Iterable[tuple[int, TaskSpec]],
) -> FrozenMapping[str, tuple[tuple[int, bool], ...]]:
    """Each alias group's reads and writes by the selected tasks, in order."""

    alias_of = {item.object_id: item.alias_group_id for item in program.objects}
    accesses: dict[str, list[tuple[int, bool]]] = {}
    for execution_ordinal, task in tasks:
        for object_id in task.inputs:
            accesses.setdefault(alias_of[object_id], []).append(
                (execution_ordinal, False)
            )
        written = tuple(task.outputs) + tuple(item.object_id for item in task.mutations)
        for object_id in written:
            accesses.setdefault(alias_of[object_id], []).append(
                (execution_ordinal, True)
            )
    return FrozenMapping({key: tuple(value) for key, value in accesses.items()})
