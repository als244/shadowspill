"""What each task's graph pair offers, and what each option costs.

A graph pair exposes alternatives for one task: today `save` and `recompute`,
which is a binary choice, but nothing here assumes that. Lowering decides which
alternatives exist and profiling measures the tasks each one activates. This
joins the two, so that choosing among them is a decision about costs rather
than a second traversal of the Program.

The inventory is orthogonal to the choice. Every option a graph pair exposes
reaches PressureFit; `recomputation` decides which combinations of them are
worth evaluating, and resolving one combination yields the concrete program a
candidate is planned against.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

from shadowspill.ir import Program

_SAVE = "save"
_RECOMPUTE = "recompute"


@dataclass(frozen=True, slots=True)
class GraphPairOption:
    """One alternative a graph pair exposes, with what choosing it costs."""

    option_id: str
    #: Bytes this option keeps resident rather than recomputing.
    retained_bytes: int
    #: Measured runtime of every task this option activates.
    runtime_ns: int


@dataclass(frozen=True, slots=True)
class GraphPairGroup:
    """Every alternative one graph pair exposes, in the Program's order."""

    group_id: str
    options: tuple[GraphPairOption, ...]
    #: The one index structure leaves, when the choice is not free. A forward
    #: sink has to keep its value: nothing downstream would recompute it.
    pinned_index: int | None

    @property
    def binary_endpoints(self) -> tuple[int, int] | None:
        """Return ``(save, recompute)`` when this group offers exactly those."""

        by_id = {option.option_id: index for index, option in enumerate(self.options)}
        if len(self.options) != 2 or set(by_id) != {_SAVE, _RECOMPUTE}:
            return None
        return (by_id[_SAVE], by_id[_RECOMPUTE])

    def by_retained_bytes(self) -> tuple[int, ...]:
        """Option indices from least to most retained, ties broken by runtime."""

        return tuple(
            index
            for index, _option in sorted(
                enumerate(self.options),
                key=lambda item: (
                    item[1].retained_bytes,
                    item[1].runtime_ns,
                    item[0],
                ),
            )
        )

    def fastest_index(self) -> int:
        """The option with the least measured runtime, ties broken by bytes."""

        return min(
            range(len(self.options)),
            key=lambda index: (
                self.options[index].runtime_ns,
                self.options[index].retained_bytes,
                index,
            ),
        )


@dataclass(frozen=True, slots=True)
class GraphPairOptions:
    """Every graph pair in one Program, costed."""

    groups: tuple[GraphPairGroup, ...]

    @classmethod
    def from_program(cls, program: Program) -> Self:
        """Cost every option against the Program's alias sizes and profiles."""

        alias_bytes = {
            alias.alias_group_id: alias.size_bytes for alias in program.alias_groups
        }
        profiles = {profile.profile_id: profile for profile in program.profiles}
        tasks = {task.task_id: task for task in program.tasks}
        pinned = _forward_sink_saves(program)
        return cls(
            groups=tuple(
                GraphPairGroup(
                    group_id=group.group_id,
                    options=tuple(
                        GraphPairOption(
                            option_id=option.option_id,
                            retained_bytes=sum(
                                alias_bytes[alias_id]
                                for alias_id in option.retained_alias_group_ids
                            ),
                            runtime_ns=sum(
                                profiles[tasks[task_id].profile_id].runtime_ns
                                for task_id in option.active_task_ids
                            ),
                        )
                        for option in group.options
                    ),
                    pinned_index=pinned.get(group_index),
                )
                for group_index, group in enumerate(program.recomputation_groups)
            )
        )

    def __len__(self) -> int:
        return len(self.groups)

    @property
    def pinned(self) -> dict[int, int]:
        """Group index to the option index structure leaves it."""

        return {
            index: group.pinned_index
            for index, group in enumerate(self.groups)
            if group.pinned_index is not None
        }

    @property
    def combination_count(self) -> int:
        """How many distinct selections exist once pinning is applied."""

        total = 1
        for group in self.groups:
            total *= 1 if group.pinned_index is not None else len(group.options)
        return total

    @property
    def binary_endpoints(self) -> tuple[tuple[int, int], ...] | None:
        """Every group's ``(save, recompute)`` indices, or None if any is not."""

        endpoints: list[tuple[int, int]] = []
        for group in self.groups:
            pair = group.binary_endpoints
            if pair is None:
                return None
            endpoints.append(pair)
        return tuple(endpoints)


def _forward_sink_saves(program: Program) -> dict[int, int]:
    """Pin every forward-DAG sink group to its structural ``save`` variant.

    A group whose forward tasks nothing else in the forward graph consumes is
    producing a value the backward pass will read. Recomputing it would mean
    recomputing it from nothing, so the choice is not free.
    """

    forward_task_ids = {
        task.task_id for task in program.tasks if task.phase == "forward"
    }
    consumed_by_forward = {
        dependency
        for task in program.tasks
        if task.phase == "forward"
        for dependency in task.dependencies
        if dependency in forward_task_ids
    }
    pinned: dict[int, int] = {}
    for group_index, group in enumerate(program.recomputation_groups):
        group_forward_tasks = {
            task_id
            for option in group.options
            for task_id in option.active_task_ids
            if task_id in forward_task_ids
        }
        if not group_forward_tasks or not group_forward_tasks.isdisjoint(
            consumed_by_forward
        ):
            continue
        save_indices = tuple(
            index
            for index, option in enumerate(group.options)
            if option.option_id == _SAVE
        )
        if len(save_indices) != 1:
            raise ValueError(
                "terminal forward recomputation group must expose exactly one "
                f"'save' option: {group.group_id!r}"
            )
        pinned[group_index] = save_indices[0]
    return pinned


__all__ = ["GraphPairGroup", "GraphPairOption", "GraphPairOptions"]
