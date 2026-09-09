"""Which resolutions of the task-alternative options are worth planning.

`options` says what each graph pair offers and what each alternative costs.
This decides which combinations of those alternatives PressureFit should
actually plan, which is a different question: the inventory is a fact about the
Program, and this is a search policy over it.

The policy is coarse and deterministic rather than adaptive, and the
resolution options it plans are the caller's to name. Small inventories are
evaluated exhaustively whatever was named. Larger ones whose every group is
the current binary
`save`/`recompute` pair get one selection per *share*: a target fraction of
the flexible groups recomputing, distributed evenly through deterministic
group order. That fraction is *across* groups, not within one group's
inventory. The library's default options, `DEFAULT_RESOLUTION_OPTIONS`, are
every quarter from 0% to 100%. Eighths were the default for a day: on the
llama3 frontier the odd eighths won 67 of 293 points by a median of 0.00% and
a mean of 0.77% over the best quarter rung, for 1.75x the search wall, so the
finer options are something a caller asks for, through `plan_program()` or
anything built on it, rather than what everyone pays for.

The within-group quantiles below are the path a non-binary inventory would
take, and are reached today only by a Program whose groups are not binary.
"""

from __future__ import annotations

from collections.abc import Iterable
from fractions import Fraction
from itertools import product

from shadowspill.ir import Program, TaskAlternativeChoice

from .options import TaskAlternativeOptions

#: One option chosen per graph pair - what makes a Program concrete.
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


def resolution_options_or_default(
    values: Iterable[ShareValue] | None,
) -> tuple[Fraction, ...]:
    """The validated options, or the library's default when none are named.

    What comes back is what is planned, keyed and recorded, so a caller that
    spells out the default asks exactly the question one that names nothing
    asks.
    """

    if values is None:
        return DEFAULT_RESOLUTION_OPTIONS
    return validate_resolution_options(values)


def resolutions(
    program: Program,
    resolution_options: Iterable[ShareValue] | None = None,
) -> tuple[Resolution, ...]:
    """Return a small deterministic set of legal resolutions.

    ``resolution_options`` names the fractions of flexible groups to recompute
    when the inventory is too large to enumerate; ``None`` plans
    `DEFAULT_RESOLUTION_OPTIONS`.
    """

    chosen = resolution_options_or_default(resolution_options)
    if not program.task_alternative_groups:
        return ((),)
    return select(TaskAlternativeOptions.from_program(program), chosen)


def select(
    options: TaskAlternativeOptions,
    resolution_options: Iterable[ShareValue] | None = None,
) -> tuple[Resolution, ...]:
    """Choose which resolutions of the inventory to plan."""

    chosen = resolution_options_or_default(resolution_options)
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
        # Rounded half up, which is where the quarter rungs always landed.
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
    options: TaskAlternativeOptions,
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
    options: TaskAlternativeOptions,
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


__all__ = [
    "DEFAULT_RESOLUTION_OPTIONS",
    "Resolution",
    "ShareValue",
    "resolution_options_or_default",
    "resolutions",
    "select",
    "validate_resolution_options",
]
