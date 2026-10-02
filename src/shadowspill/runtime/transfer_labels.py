"""Cold-path semantic labels for transfer profiler ranges."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field

from shadowspill.ir import MemoryAction, MemoryActionKind, ShadowSpillProgram

_UNSAFE_COMPONENT = re.compile(r"[^A-Za-z0-9_.-]+")


def _component(value: str) -> str:
    sanitized = _UNSAFE_COMPONENT.sub("_", value).strip("_.")
    return sanitized or "unknown"


@dataclass(frozen=True, slots=True)
class TransferLabelIndex:
    """Precompute graph relationships used by worker-thread profiler labels."""

    program: ShadowSpillProgram
    task_labels: Mapping[str, str]
    _task_positions: Mapping[str, int] = field(init=False, repr=False, compare=False)
    _aliases_by_object: Mapping[str, str] = field(init=False, repr=False, compare=False)
    _alias_labels: Mapping[str, str] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        roles: dict[str, set[str]] = {}
        aliases_by_object: dict[str, str] = {}
        for item in self.program.objects:
            aliases_by_object[item.object_id] = item.alias_group_id
            roles.setdefault(item.alias_group_id, set()).add(item.role.value)
        alias_labels = {}
        for group in self.program.alias_groups:
            alias = group.alias_group_id
            role = "-".join(sorted(roles.get(alias, ()))) or "unknown"
            alias_labels[alias] = (
                f"{_component(alias)}.role_{_component(role)}.bytes_{group.size_bytes}"
            )
        object.__setattr__(
            self,
            "_task_positions",
            {
                task.task_id: position
                for position, task in enumerate(self.program.tasks)
            },
        )
        object.__setattr__(self, "_aliases_by_object", aliases_by_object)
        object.__setattr__(self, "_alias_labels", alias_labels)

    def labels_for(self, actions: tuple[MemoryAction, ...]) -> tuple[str, ...]:
        """Return one immutable profiler label for every ordered action."""

        return tuple(self._label(action) for action in actions)

    def _label(self, action: MemoryAction) -> str:
        alias = action.alias_group_id
        trigger_position = self._task_positions[action.trigger_task_id]
        trigger = self._task_label(action.trigger_task_id)

        prefix = (
            f"shadowspill.runtime.transfer.{self._operation(action.kind)}."
            f"{self._alias_labels[alias]}"
        )
        if action.kind is MemoryActionKind.FETCH:
            consumer = self._next_consumer(
                alias,
                trigger_position,
                self._aliases_by_object,
            )
            relationship = (
                f"for_input.{self._task_label(consumer)}"
                if consumer is not None
                else "for_input.no_later_consumer"
            )
        elif action.kind is MemoryActionKind.EVICT:
            relation, producer = self._latest_source(
                alias,
                trigger_position,
                self._aliases_by_object,
            )
            relationship = (
                f"from_{relation}.{self._task_label(producer)}"
                if producer is not None
                else "from_persistent_state"
            )
        else:
            relationship = "release_only"
        return f"{prefix}.{relationship}.trigger.{trigger}"[:1024]

    def _task_label(self, task_id: str | None) -> str:
        if task_id is None:
            return "unknown_task"
        return _component(self.task_labels.get(task_id, task_id))

    def _next_consumer(
        self,
        alias: str,
        trigger_position: int,
        aliases_by_object: Mapping[str, str],
    ) -> str | None:
        for task in self.program.tasks[trigger_position + 1 :]:
            if any(aliases_by_object[object_id] == alias for object_id in task.inputs):
                return task.task_id
        return None

    def _latest_source(
        self,
        alias: str,
        trigger_position: int,
        aliases_by_object: Mapping[str, str],
    ) -> tuple[str, str | None]:
        for task in reversed(self.program.tasks[: trigger_position + 1]):
            if any(aliases_by_object[object_id] == alias for object_id in task.outputs):
                return "output", task.task_id
            if any(
                aliases_by_object[mutation.object_id] == alias
                for mutation in task.mutations
            ):
                return "mutation", task.task_id
        for task in reversed(self.program.tasks[: trigger_position + 1]):
            if any(aliases_by_object[object_id] == alias for object_id in task.inputs):
                return "last_input", task.task_id
        return "persistent", None

    @staticmethod
    def _operation(kind: MemoryActionKind) -> str:
        if kind is MemoryActionKind.FETCH:
            return "fetch"
        if kind is MemoryActionKind.EVICT:
            return "evict"
        if kind is MemoryActionKind.WRITE_BACK:
            return "write_back"
        return "release"


__all__ = ["TransferLabelIndex"]
