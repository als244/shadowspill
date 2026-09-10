"""Which resolutions of a program's task alternatives are worth planning.

A program arrives with its alternatives open: each group offers options --
today `save` and `recompute`, though nothing here assumes two. Fixing one
option for every group gives a *resolved program*, and this module answers
which of those are worth the search's time.

It does that in two steps, and both live here because neither is useful
without the other. First the alternatives are **costed** against the
program's own alias sizes and profiles, which is what says whether a group
is a real decision at all: a forward sink has to keep its value, and two
options retaining the same bytes are one plan spelled twice. Then a
**share** of the groups that remain is chosen to recompute, distributed
through deterministic group order.

The costing is a fact about the program; the share is the caller's policy.
Small inventories are enumerated exhaustively whatever was named. The
share is *across* groups, not within one.

Names here are deliberately not the IR's. `shadowspill.ir` owns
`TaskAlternativeGroup` and `TaskAlternativeOption`, which are what lowering
declared; `CostedGroup` and `CostedOption` are what they cost, which is a
different thing about the same subject.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from typing import Self

from shadowspill.ir import ShadowSpillProgram, TaskAlternativeChoice

#: One option chosen per group -- what makes a program concrete.
Resolution = tuple[TaskAlternativeChoice, ...]

#: How a caller may spell one share: an exact fraction, an integer (only 0
#: and 1 are in range) or a string `Fraction` reads, such as ``"3/8"``.
ShareValue = Fraction | int | str

#: The resolution options planned when a caller names none: every quarter
#: of the flexible groups recomputing, from none to all.
DEFAULT_RESOLUTION_OPTIONS: tuple[Fraction, ...] = tuple(
    Fraction(n, 4) for n in range(5)
)

_EXHAUSTIVE_COMBINATION_LIMIT = 64
_QUARTER_DENOMINATOR = 4
_WITHIN_GROUP_MEMORY_QUANTILES = (0, 1, 2, 3, 4)


def validate_resolution_options(
    values: Iterable[ShareValue],
) -> tuple[Fraction, ...]:
    """Return the options as exact fractions in ``[0, 1]``, sorted and unique.

    Each option is a share of the flexible groups to recompute. Floats are
    refused rather than converted: ``0.1`` is not one tenth, and an option is
    part of a planned program's identity, so it is spelled as a `Fraction`,
    an integer or a string such as ``"1/8"``.
    """

    if isinstance(values, str):
        raise ValueError(
            "resolution options are a sequence of shares, not one string"
        )
    chosen: list[Fraction] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Fraction | int | str):
            raise ValueError(
                "resolution options are Fractions, integers or strings such as"
                f" '3/8'; got {value!r}"
            )
        try:
            share = Fraction(value)
        except (ValueError, ZeroDivisionError) as error:
            raise ValueError(
                f"resolution option {value!r} is not a fraction"
            ) from error
        if not 0 <= share <= 1:
            raise ValueError(f"resolution option {share} is outside [0, 1]")
        chosen.append(share)
    if not chosen:
        raise ValueError("resolution options must name at least one share")
    return tuple(sorted(set(chosen)))


def resolutions(
    program: ShadowSpillProgram,
    resolution_options: Iterable[ShareValue],
) -> tuple[Resolution, ...]:
    """Return a small deterministic set of legal resolutions.

    ``resolution_options`` names the fractions of flexible groups to
    recompute when the inventory is too large to enumerate. It is required:
    a search that wants the usual set asks for
    `DEFAULT_RESOLUTION_OPTIONS` by name, so there is no reading of this
    call that silently plans something other than what was asked for.
    """

    chosen = validate_resolution_options(resolution_options)
    if not program.task_alternative_groups:
        return ((),)
    return _select(CostedAlternatives.from_program(program), chosen)


def _select(
    options: CostedAlternatives,
    resolution_options: Iterable[ShareValue],
) -> tuple[Resolution, ...]:
    """Choose which resolutions of the inventory to plan."""

    chosen = validate_resolution_options(resolution_options)
    if not options.groups:
        return ((),)

    forced = options.forced
    if options.combination_count <= _EXHAUSTIVE_COMBINATION_LIMIT:
        per_group = tuple(
            (forced[index],) if index in forced else tuple(range(len(group.options)))
            for index, group in enumerate(options.groups)
        )
        return tuple(_resolution(options, item) for item in product(*per_group))

    endpoints = options.binary_endpoints
    if endpoints is not None:
        return tuple(
            _resolution(options, indices)
            for indices in _group_fractions(endpoints, forced, chosen)
        )
    return tuple(
        _resolution(options, item) for item in _within_group_quantiles(options)
    )


def _group_fractions(
    endpoints: tuple[tuple[int, int], ...],
    forced: dict[int, int],
    resolution_options: tuple[Fraction, ...],
) -> tuple[tuple[int, ...], ...]:
    """Build one evenly distributed resolution per option, a share recomputing."""

    flexible = tuple(index for index in range(len(endpoints)) if index not in forced)
    result: list[tuple[int, ...]] = []
    for share in resolution_options:
        # Rounded half up, so a share lands on the nearer whole group.
        recompute_count = int(len(flexible) * share + Fraction(1, 2))
        recomputing = {
            flexible[position]
            for position in _evenly_spaced_indices(len(flexible), recompute_count)
        }
        result.append(
            tuple(
                forced.get(index, recompute if index in recomputing else save)
                for index, (save, recompute) in enumerate(endpoints)
            )
        )
    # Two shares of a small group count can round to the same resolution;
    # planning it twice would answer nothing new.
    return _unique(result)


def _within_group_quantiles(
    options: CostedAlternatives,
) -> tuple[tuple[int, ...], ...]:
    """Walk each group's own inventory, from least retained to most."""

    forced = options.forced
    memory_order = tuple(group.by_retained_bytes() for group in options.groups)
    raw: list[tuple[int, ...]] = [
        tuple(0 for _group in options.groups),
        tuple(len(group.options) - 1 for group in options.groups),
        tuple(order[0] for order in memory_order),
        tuple(group.fastest_index() for group in options.groups),
    ]
    for numerator in _WITHIN_GROUP_MEMORY_QUANTILES:
        raw.append(
            tuple(
                order[(len(order) - 1) * numerator // _QUARTER_DENOMINATOR]
                for order in memory_order
            )
        )
    return _unique(
        [
            tuple(
                forced.get(index, option_index)
                for index, option_index in enumerate(item)
            )
            for item in raw
        ]
    )


def _evenly_spaced_indices(total: int, count: int) -> frozenset[int]:
    """Choose ``count`` centered, evenly spaced positions from ``total``."""

    if not 0 <= count <= total:
        raise ValueError("evenly spaced selection count is outside its domain")
    if count == 0:
        return frozenset()
    if count == total:
        return frozenset(range(total))
    return frozenset(
        (2 * position + 1) * total // (2 * count) for position in range(count)
    )


def _resolution(
    options: CostedAlternatives,
    indices: tuple[int, ...],
) -> Resolution:
    return tuple(
        TaskAlternativeChoice(group.group_id, group.options[index].option_id)
        for group, index in zip(options.groups, indices, strict=True)
    )


def _unique(values: list[tuple[int, ...]]) -> tuple[tuple[int, ...], ...]:
    result: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return tuple(result)




_SAVE = "save"
_RECOMPUTE = "recompute"


@dataclass(frozen=True, slots=True)
class CostedOption:
    """One alternative a group exposes, with what choosing it costs."""

    option_id: str
    #: Bytes this option keeps resident rather than recomputing.
    retained_bytes: int
    #: Measured runtime of every task this option activates.
    runtime_ns: int


@dataclass(frozen=True, slots=True)
class CostedGroup:
    """Every alternative one group exposes, in the ShadowSpillProgram's order."""

    group_id: str
    options: tuple[CostedOption, ...]
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
class CostedAlternatives:
    """Every task-alternative group in one ShadowSpillProgram, costed."""

    groups: tuple[CostedGroup, ...]

    @classmethod
    def from_program(cls, program: ShadowSpillProgram) -> Self:
        """Cost every option against the program's alias sizes and profiles."""

        alias_bytes = {
            alias.alias_group_id: alias.size_bytes for alias in program.alias_groups
        }
        profiles = {profile.profile_id: profile for profile in program.profiles}
        tasks = {task.task_id: task for task in program.tasks}
        structural = _forward_sink_saves(program)
        groups: list[CostedGroup] = []
        for group_index, group in enumerate(program.task_alternative_groups):
            options = tuple(
                CostedOption(
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
                CostedGroup(
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


def _forward_sink_saves(program: ShadowSpillProgram) -> dict[int, int]:
    """Force every sink of the forward phase to its ``save`` option.

    A task is a **sink of a phase** when no other task in that same phase
    consumes it, reading the graph the way values travel: producer to
    consumer. (``TaskSpec.dependencies`` stores the opposite orientation, so
    read that field literally and a sink looks like a source.) A group whose
    forward tasks are forward sinks is producing a value the backward pass
    will read, and recomputing it would mean recomputing it from nothing, so
    the choice is not free and the group is forced.

    The rule names the ``forward`` phase literally, which confines this piece
    of training knowledge to programs that declare they are training: a
    program whose tasks carry no ``forward`` phase forces nothing here and
    keeps every alternative open. See the phases-and-sinks section of the IR
    architecture page.
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
    options: tuple[CostedOption, ...],
) -> int | None:
    """Force a group whose options keep the same bytes, to its fastest.

    An alternative trades runtime for retained bytes. When every option keeps
    the same bytes there is nothing to trade, so the group is not a decision:
    the search would carry a dimension whose two ends are one plan spelled
    twice, and resolve it arbitrarily -- which also makes the plan digest
    depend on nothing. Taking the fastest costs nothing and removes it.

    Named by what is true of the options rather than by which stage they
    belong to, so it holds for any ShadowSpillProgram.
    """

    if len(options) < 2:
        return 0 if options else None
    if len({option.retained_bytes for option in options}) != 1:
        return None
    return min(
        range(len(options)),
        key=lambda index: (options[index].runtime_ns, index),
    )


__all__ = ["CostedAlternatives", "CostedGroup", "CostedOption"]


__all__ = [
    "DEFAULT_RESOLUTION_OPTIONS",
    "Resolution",
    "ShareValue",
    "resolutions",
    "validate_resolution_options",
]
