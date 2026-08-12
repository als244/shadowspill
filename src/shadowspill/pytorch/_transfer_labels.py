"""Cold-path semantic labels for transfer profiler ranges."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

from shadowspill.ir import MemoryAction, MemoryActionKind, Program

_UNSAFE_COMPONENT = re.compile(r"[^A-Za-z0-9_.-]+")


def _component(value: str) -> str:
    sanitized = _UNSAFE_COMPONENT.sub("_", value).strip("_.")
    return sanitized or "unknown"


@dataclass(frozen=True, slots=True)
class TransferLabelIndex:
    """Precompute graph relationships used by worker-thread NVTX labels."""

    program: Program
    task_labels: Mapping[str, str]

    def labels_for(
        self, actions: tuple[MemoryAction, ...]
    ) -> tuple[str, ...]:
        """Return one immutable profiler label for every ordered action."""

        return tuple(self._label(action) for action in actions)

    def _label(self, action: MemoryAction) -> str:
        task_positions = {
            task.task_id: position
            for position, task in enumerate(self.program.tasks)
        }
        aliases_by_object = {
            item.object_id: item.alias_group_id
            for item in self.program.objects
        }
        alias = action.alias_group_id
        members = tuple(
            item for item in self.program.objects if item.alias_group_id == alias
        )
        roles = "-".join(sorted({item.role.value for item in members})) or "unknown"
        sizes = {
            item.alias_group_id: item.size_bytes
            for item in self.program.alias_groups
        }
        trigger_position = task_positions[action.trigger_task_id]
        trigger = self._task_label(action.trigger_task_id)

        prefix = (
            f"shadowspill.runtime.transfer.{self._operation(action.kind)}."
            f"{_component(alias)}.role_{_component(roles)}."
            f"bytes_{sizes[alias]}"
        )
        if action.kind is MemoryActionKind.PREFETCH:
            consumer = self._next_consumer(
                alias,
                trigger_position,
                aliases_by_object,
            )
            relationship = (
                f"for_input.{self._task_label(consumer)}"
                if consumer is not None
                else "for_input.no_later_consumer"
            )
        elif action.kind is MemoryActionKind.OFFLOAD:
            relation, producer = self._latest_source(
                alias,
                trigger_position,
                aliases_by_object,
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
        if kind is MemoryActionKind.PREFETCH:
            return "fetch"
        if kind is MemoryActionKind.OFFLOAD:
            return "evict"
        return "release"


__all__ = ["TransferLabelIndex"]
