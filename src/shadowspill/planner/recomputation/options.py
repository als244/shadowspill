"""What each task-alternative group offers, and what each option costs.

A group exposes alternatives for one task: today `save` and `recompute`,
which is a binary choice, but nothing here assumes that. Lowering decides which
alternatives exist and profiling measures the tasks each one activates. This
joins the two, so that choosing among them is a decision about costs rather
than a second traversal of the Program.

The inventory is orthogonal to the choice. Every option a group exposes
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
class TaskAlternativeOption:
    """One alternative a group exposes, with what choosing it costs."""

    option_id: str
    #: Bytes this option keeps resident rather than recomputing.
    retained_bytes: int
    #: Measured runtime of every task this option activates.
    runtime_ns: int


@dataclass(frozen=True, slots=True)
class TaskAlternativeGroup:
    """Every alternative one group exposes, in the Program's order."""

    group_id: str
    options: tuple[TaskAlternativeOption, ...]
    #: The one index left when the choice is not free, or None when the group
    #: is a real decision. Two things force a group: a forward sink has to keep
    #: its value, because nothing downstream would recompute it; and options
    #: that keep the same bytes are the same plan spelled twice.
    forced_index: int | None

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
class TaskAlternativeOptions:
    """Every task-alternative group in one Program, costed."""

    groups: tuple[TaskAlternativeGroup, ...]

    @classmethod
    def from_program(cls, program: Program) -> Self:
        """Cost every option against the Program's alias sizes and profiles."""

        alias_bytes = {
            alias.alias_group_id: alias.size_bytes for alias in program.alias_groups
        }
        profiles = {profile.profile_id: profile for profile in program.profiles}
        tasks = {task.task_id: task for task in program.tasks}
        structural = _forward_sink_saves(program)
        groups: list[TaskAlternativeGroup] = []
        for group_index, group in enumerate(program.task_alternative_groups):
            options = tuple(
                TaskAlternativeOption(
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
            )
            forced = structural.get(group_index)
            if forced is None:
                forced = _forced_by_equal_retention(options)
            groups.append(
                TaskAlternativeGroup(
                    group_id=group.group_id,
                    options=options,
                    forced_index=forced,
                )
            )
        return cls(groups=tuple(groups))

    def __len__(self) -> int:
        return len(self.groups)

    @property
    def forced(self) -> dict[int, int]:
        """Group index to the one option index left, for groups with no choice."""

        return {
            index: group.forced_index
            for index, group in enumerate(self.groups)
            if group.forced_index is not None
        }

    @property
    def flexible_count(self) -> int:
        """How many groups are a real decision.

        This is the population a resolution share is taken of, so it is also
        the honest denominator for reporting how many groups recompute.
        """

        return sum(1 for group in self.groups if group.forced_index is None)

    @property
    def combination_count(self) -> int:
        """How many distinct selections exist once forced groups are settled."""

        total = 1
        for group in self.groups:
            total *= 1 if group.forced_index is not None else len(group.options)
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
    """Force every sink of the forward phase to its ``save`` option.

    A task is a **sink of a phase** when no other task in that same phase
    consumes it, reading the graph the way values travel: producer to
    consumer. (``TaskSpec.dependencies`` stores the opposite orientation, so
    read that field literally and a sink looks like a source.) A group whose
    forward tasks are forward sinks is producing a value the backward pass
    will read, and recomputing it would mean recomputing it from nothing, so
    the choice is not free and the group is forced.

    The rule deliberately names one phase rather than generalising to "sinks
    of whatever phase the group enters first". That generalisation is not
    behaviour-preserving: a Program whose tasks carry no ``forward`` phase
    forces nothing here and keeps every alternative open, and phrasing the
    rule in the abstract would instead force all of its terminal groups and
    delete
    its recomputation search entirely. Scoping to ``forward`` is what confines
    this piece of training knowledge to programs that declare they are
    training. See the phases-and-sinks section of the IR architecture page.
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
    forced: dict[int, int] = {}
    for group_index, group in enumerate(program.task_alternative_groups):
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
                "a group that is a sink of the forward phase must expose "
                f"exactly one 'save' option: {group.group_id!r}"
            )
        forced[group_index] = save_indices[0]
    return forced


def _forced_by_equal_retention(
    options: tuple[TaskAlternativeOption, ...],
) -> int | None:
    """Force a group whose options keep the same bytes, to its fastest.

    An alternative trades runtime for retained bytes. When every option keeps
    the same bytes there is nothing to trade, so the group is not a decision:
    the search would carry a dimension whose two ends are one plan spelled
    twice, and resolve it arbitrarily -- which also makes the plan digest
    depend on nothing. Taking the fastest costs nothing and removes it.

    Named by what is true of the options rather than by which stage they
    belong to, so it holds for any Program.
    """

    if len(options) < 2:
        return 0 if options else None
    if len({option.retained_bytes for option in options}) != 1:
        return None
    return min(
        range(len(options)),
        key=lambda index: (options[index].runtime_ns, index),
    )


__all__ = ["TaskAlternativeGroup", "TaskAlternativeOption", "TaskAlternativeOptions"]
