"""Which resolutions of a program's task alternatives are worth planning.

A program arrives with its alternatives open: each group offers options --
`save` and `recompute`, though nothing here assumes two. Fixing one
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

from collections.abc import Iterable, Mapping
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

#: Resolution sets by name, as a command line or a config spells them: the
#: shares of the flexible groups to recompute at every quarter, at every
#: eighth, or at none, half and all.
NAMED_RESOLUTION_OPTIONS: Mapping[str, tuple[str, ...]] = {
    "quarters": tuple(str(share) for share in DEFAULT_RESOLUTION_OPTIONS),
    "eighths": tuple(str(Fraction(n, 8)) for n in range(9)),
    "halves": ("0", "1/2", "1"),
}

_EXHAUSTIVE_COMBINATION_LIMIT = 64
_QUARTER_DENOMINATOR = 4
_WITHIN_GROUP_MEMORY_QUANTILES = (0, 1, 2, 3, 4)


def validate_resolution_options(
    values: Iterable[ShareValue],
) -> tuple[Fraction, ...]:
    """Check resolution options and return them as exact, sorted fractions.

    Each option is a share of the flexible task-alternative groups to
    recompute. A float is refused rather than converted, since ``0.1`` is
    not one tenth and an option is part of the plan key.

    Parameters
    ----------
    values : iterable of Fraction, int or str
        The shares, each a ``Fraction``, an integer or a string such as
        ``"3/8"``, in ``[0, 1]``. At least one.

    Returns
    -------
    tuple of Fraction
        The distinct shares, ascending.

    Raises
    ------
    ValueError
        If ``values`` is a single string or empty, or a share is a float, a
        bool, not a fraction, or outside ``[0, 1]``.
    """

    if isinstance(values, str):
        raise ValueError("resolution options are a sequence of shares, not one string")
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


def named_resolution_options(value: str | Iterable[ShareValue]) -> tuple[str, ...]:
    """Resolve a named or listed set of resolution options to strings.

    Strings are how a request records the options and how a command line or
    a config spells them; `validate_resolution_options` turns them into the
    fractions a search plans with.

    Parameters
    ----------
    value : str or iterable of Fraction, int or str
        A name in ``NAMED_RESOLUTION_OPTIONS`` (``"quarters"``,
        ``"eighths"`` or ``"halves"``), the shares as a comma-separated
        string such as ``"0, 1/2, 1"``, or a sequence of shares.

    Returns
    -------
    tuple of str
        The distinct shares, ascending, such as ``("0", "1/2", "1")``.

    Raises
    ------
    ValueError
        If a share is invalid, as `validate_resolution_options` checks.
    """

    if isinstance(value, str):
        if value in NAMED_RESOLUTION_OPTIONS:
            return NAMED_RESOLUTION_OPTIONS[value]
        shares: Iterable[ShareValue] = [
            item.strip() for item in value.split(",") if item.strip()
        ]
    else:
        shares = value
    return tuple(str(share) for share in validate_resolution_options(shares))


def validate_tail_save_counts(values: Iterable[int]) -> tuple[int, ...]:
    """Check tail save counts and return them sorted and distinct.

    A tail count is how many final eligible forward invocations of each
    forward/backward pass are saved; zero saves none.

    Parameters
    ----------
    values : iterable of int
        The counts, each a non-negative integer. At least one.

    Returns
    -------
    tuple of int
        The distinct counts, ascending.

    Raises
    ------
    ValueError
        If ``values`` is a string, not iterable or empty, or a count is not a
        non-negative integer.
    """
    if isinstance(values, str) or not isinstance(values, Iterable):
        raise ValueError("tail save counts must be a sequence of nonnegative integers")
    counts = tuple(values)
    if not counts:
        raise ValueError("tail save counts must name at least one count")
    if any(type(value) is not int or value < 0 for value in counts):
        raise ValueError("tail save counts must be nonnegative integers")
    return tuple(sorted(set(counts)))


def resolutions(
    program: ShadowSpillProgram,
    resolution_options: Iterable[ShareValue],
    *,
    tail_save_counts: Iterable[int] = (0,),
) -> tuple[Resolution, ...]:
    """Choose a small, deterministic set of resolutions of a program to plan.

    A resolution fixes one option for every task-alternative group. A group
    with no real choice keeps its one option. When the remaining
    combinations are few (64 or fewer), every one is returned. Otherwise,
    when every group is a save-or-recompute pair, each tail count is
    crossed with each share: the tail saves the final eligible forward
    invocations of each forward/backward pass, and the share recomputes
    evenly spaced groups among the remaining flexible ones, rounded to the
    nearer whole group. Groups of other shapes, with no nonzero tail, are
    walked through their own options from least retained to most.
    Duplicates are returned once.

    Parameters
    ----------
    program : ShadowSpillProgram
        The program, its task alternatives open.
    resolution_options : iterable of Fraction, int or str
        Shares of the flexible groups to recompute, as
        `validate_resolution_options` takes them, such as
        ``DEFAULT_RESOLUTION_OPTIONS``.
    tail_save_counts : iterable of int, default (0,)
        Counts of final forward invocations to save in each pass, as
        `validate_tail_save_counts` takes them.

    Returns
    -------
    tuple of tuple of TaskAlternativeChoice
        The resolutions, each one choice per group; a single empty
        resolution for a program without alternatives.

    Raises
    ------
    ValueError
        If an option or count is invalid, a forward-sink group lacks a single
        ``save`` option, or a nonzero tail is asked of a large inventory whose
        groups are not all save-or-recompute pairs or whose passes cannot be
        told apart.
    """

    chosen = validate_resolution_options(resolution_options)
    counts = validate_tail_save_counts(tail_save_counts)
    if not program.task_alternative_groups:
        return ((),)
    costed = CostedAlternatives.from_program(program)
    if counts == (0,) or costed.combination_count <= _EXHAUSTIVE_COMBINATION_LIMIT:
        return _select(costed, chosen)
    endpoints = costed.binary_endpoints
    if endpoints is None:
        raise ValueError("tail saves require binary save/recompute alternatives")
    from shadowspill.planning.memory_search.toolkit.traversals import (
        alternative_traversals,
    )

    flexible = tuple(
        i for i, group in enumerate(costed.groups) if group.forced_index is None
    )
    passes = alternative_traversals(program, flexible)
    selected: list[Resolution] = []
    for count in counts:
        saved = {i for group in passes for i in group[-count:]} if count else set()
        forced = {**costed.forced, **{i: endpoints[i][0] for i in saved}}
        selected.extend(
            _resolution(costed, indices)
            for indices in _group_fractions(endpoints, forced, chosen)
        )
    return tuple(dict.fromkeys(selected))


def recompute_share(
    program: ShadowSpillProgram, selections: Iterable[TaskAlternativeChoice]
) -> Fraction:
    """The share of the program's flexible groups a selection recomputes.

    The flexible groups are the population a resolution option's share is
    taken of, so this is what a kept resolution reports itself by. For a
    program whose groups are not all save-or-recompute pairs, it is the
    share of every group choosing ``recompute``.
    """

    costed = CostedAlternatives.from_program(program)
    chosen = {item.group_id: item.option_id for item in selections}
    endpoints = costed.binary_endpoints
    if endpoints is None or costed.flexible_count == 0:
        if not costed.groups:
            return Fraction(0)
        return Fraction(
            sum(
                1 for group in costed.groups if chosen.get(group.group_id) == _RECOMPUTE
            ),
            len(costed.groups),
        )
    forced = costed.forced
    recomputing = sum(
        1
        for index, group in enumerate(costed.groups)
        if index not in forced
        and group.options[endpoints[index][1]].option_id == chosen.get(group.group_id)
    )
    return Fraction(recomputing, costed.flexible_count)


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
    return tuple(dict.fromkeys(result))


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
    return tuple(
        dict.fromkeys(
            [
                tuple(
                    forced.get(index, option_index)
                    for index, option_index in enumerate(item)
                )
                for item in raw
            ]
        )
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


_SAVE = "save"
_RECOMPUTE = "recompute"


@dataclass(frozen=True, slots=True)
class CostedOption:
    """One option of a task-alternative group, with what choosing it costs.

    `CostedGroup.options` holds these.

    Attributes
    ----------
    option_id : str
        The option, as the program names it.
    retained_bytes : int
        Bytes of the alias groups the option keeps rather than recomputing.
    runtime_ns : int
        Profiled runtime of every task the option activates, in
        nanoseconds.
    """

    option_id: str
    #: Bytes this option keeps resident rather than recomputing.
    retained_bytes: int
    #: Measured runtime of every task this option activates.
    runtime_ns: int


@dataclass(frozen=True, slots=True)
class CostedGroup:
    """One task-alternative group, its options costed.

    `CostedAlternatives.groups` holds these.

    Attributes
    ----------
    group_id : str
        The group, as the program names it.
    options : tuple of CostedOption
        Its options, in the program's order.
    forced_index : int or None
        The option a forced group must take, or None when the group is a
        real choice. A group whose forward tasks are sinks of the forward
        phase is forced to its ``save`` option; one whose options all retain
        the same bytes, to its fastest.
    """

    group_id: str
    options: tuple[CostedOption, ...]
    #: The one index left when the choice is not free, or None when the group
    #: is a real decision. Two things force a group: a forward sink has to keep
    #: its value, because nothing downstream would recompute it; and options
    #: that keep the same bytes are the same plan spelled twice.
    forced_index: int | None

    @property
    def binary_endpoints(self) -> tuple[int, int] | None:
        """The indices of ``save`` and ``recompute``, if they are its only options.

        None for any other set of options.
        """

        by_id = {option.option_id: index for index, option in enumerate(self.options)}
        if len(self.options) != 2 or set(by_id) != {_SAVE, _RECOMPUTE}:
            return None
        return (by_id[_SAVE], by_id[_RECOMPUTE])

    def by_retained_bytes(self) -> tuple[int, ...]:
        """Order the options from least retained bytes to most.

        Returns
        -------
        tuple of int
            Option indices, ties broken by runtime and then by index.
        """

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
        """Find the option with the least profiled runtime.

        Returns
        -------
        int
            Its index, ties broken by retained bytes and then by index.
        """

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
    """Every task-alternative group of one program, costed.

    `CostedAlternatives.from_program` builds one. Its length is the number
    of groups.

    Attributes
    ----------
    groups : tuple of CostedGroup
        One per group, in the program's order.
    """

    groups: tuple[CostedGroup, ...]

    @classmethod
    def from_program(cls, program: ShadowSpillProgram) -> Self:
        """Cost every option against the program's alias sizes and profiles.

        Parameters
        ----------
        program : ShadowSpillProgram
            The program, its task alternatives open.

        Returns
        -------
        CostedAlternatives
            Its groups, costed and with forced groups marked.

        Raises
        ------
        ValueError
            If a group whose forward tasks are forward-phase sinks does not
            offer exactly one ``save`` option.
        """

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
        """The forced groups: group index to the index of its one option."""

        return {
            index: group.forced_index
            for index, group in enumerate(self.groups)
            if group.forced_index is not None
        }

    @property
    def flexible_count(self) -> int:
        """How many groups are a real choice.

        A resolution option's share is taken of these, so it is also the
        denominator for reporting how many groups recompute.
        """

        return sum(1 for group in self.groups if group.forced_index is None)

    @property
    def combination_count(self) -> int:
        """How many distinct resolutions exist once forced groups are settled."""

        total = 1
        for group in self.groups:
            total *= 1 if group.forced_index is not None else len(group.options)
        return total

    @property
    def binary_endpoints(self) -> tuple[tuple[int, int], ...] | None:
        """Every group's `CostedGroup.binary_endpoints`, or None if any has none."""

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


__all__ = [
    "DEFAULT_RESOLUTION_OPTIONS",
    "NAMED_RESOLUTION_OPTIONS",
    "CostedAlternatives",
    "CostedGroup",
    "CostedOption",
    "Resolution",
    "ShareValue",
    "named_resolution_options",
    "recompute_share",
    "resolutions",
    "validate_resolution_options",
    "validate_tail_save_counts",
]
