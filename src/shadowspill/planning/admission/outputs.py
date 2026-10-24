"""Persistent returned storage, independent of the framework that built it."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from shadowspill.ir import TaskSpec
from shadowspill.task.entrypoints import TaskEntrypoint


@dataclass(frozen=True, slots=True)
class TaskOutputBinding:
    """Map one returned tensor leaf to its persistent ShadowSpillProgram alias group."""

    leaf_index: int
    alias_group_id: str
    replacement: bool = False
    source_alias_group_id: str | None = None

    def __post_init__(self) -> None:
        if self.leaf_index < 0:
            raise ValueError("output leaf index must be non-negative")
        if not self.alias_group_id:
            raise ValueError("output alias group ID must be non-empty")
        if self.source_alias_group_id == self.alias_group_id:
            raise ValueError("storage handoff source and destination must differ")


def output_bindings_for_entrypoints(
    tasks: Sequence[TaskSpec],
    entrypoints: Sequence[TaskEntrypoint],
    alias_by_object: Mapping[str, str],
) -> dict[str, tuple[TaskOutputBinding, ...]]:
    """Describe which returned tensor allocations become persistent outputs."""

    task_by_id = {task.task_id: task for task in tasks}
    result: dict[str, tuple[TaskOutputBinding, ...]] = {}
    for entrypoint in entrypoints:
        task = task_by_id.get(entrypoint.task_id)
        if task is None:
            continue
        # A task that contributes to objects rather than producing them binds
        # its contributions; every other task binds what it returned.
        slots = entrypoint.options.contribution_slots or entrypoint.output_slots
        output_objects = set(task.outputs)
        replacement_leaves = set(entrypoint.replacement_output_leaves)
        handoff_by_leaf = {
            item.leaf_index: item for item in entrypoint.storage_handoffs
        }
        seen_aliases: set[str] = set()
        bindings: list[TaskOutputBinding] = []
        for slot in slots:
            replacement = slot.leaf_index in replacement_leaves
            if slot.object_id not in output_objects and not replacement:
                continue
            alias_id = alias_by_object[slot.object_id]
            if alias_id in seen_aliases:
                continue
            bindings.append(
                TaskOutputBinding(
                    slot.leaf_index,
                    alias_id,
                    replacement,
                    (
                        alias_by_object[
                            handoff_by_leaf[slot.leaf_index].source_object_id
                        ]
                        if slot.leaf_index in handoff_by_leaf
                        else None
                    ),
                )
            )
            seen_aliases.add(alias_id)
        result[task.task_id] = tuple(bindings)
    return result
