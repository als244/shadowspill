"""Which resolutions of the task-alternative options are worth planning.

`options` says what each graph pair offers and what each alternative costs.
This decides which combinations of those alternatives PressureFit should
actually plan, which is a different question: the inventory is a fact about the
Program, and this is a search policy over it.

The policy is coarse and deterministic rather than adaptive. Small inventories
are evaluated exhaustively. Larger ones whose every group is the current binary
`save`/`recompute` pair get nine selections - every eighth from 0% to 100% of
groups recomputing - distributed evenly through deterministic group order. That
fraction is *across* groups, not within one group's inventory. Eighths rather
than quarters because the answer under pressure sits near the top of the
ladder, where a quarter step is the difference between a plan and none.

The within-group quantiles below are the path a non-binary inventory would
take, and are reached today only by a Program whose groups are not binary.
"""

from __future__ import annotations

from itertools import product

from shadowspill.ir import Program, TaskAlternativeChoice

from .options import TaskAlternativeOptions

#: One option chosen per graph pair - what makes a Program concrete.
Resolution = tuple[TaskAlternativeChoice, ...]

_EXHAUSTIVE_COMBINATION_LIMIT = 64
_EIGHTH_DENOMINATOR = 8
_GROUP_RECOMPUTE_EIGHTHS = tuple(range(9))
_QUARTER_DENOMINATOR = 4
_WITHIN_GROUP_MEMORY_QUANTILES = (0, 1, 2, 3, 4)


def resolutions(program: Program) -> tuple[Resolution, ...]:
    """Return a small deterministic family of legal resolutions."""

    if not program.task_alternative_groups:
        return ((),)
    return select(TaskAlternativeOptions.from_program(program))


def select(options: TaskAlternativeOptions) -> tuple[Resolution, ...]:
    """Choose which resolutions of the inventory to plan."""

    if not options.groups:
        return ((),)

    pinned = options.pinned
    if options.combination_count <= _EXHAUSTIVE_COMBINATION_LIMIT:
        per_group = tuple(
            (pinned[index],) if index in pinned else tuple(range(len(group.options)))
            for index, group in enumerate(options.groups)
        )
        return tuple(_resolution(options, item) for item in product(*per_group))

    endpoints = options.binary_endpoints
    if endpoints is not None:
        return tuple(
            _resolution(options, indices)
            for indices in _group_fractions(endpoints, pinned)
        )
    return tuple(
        _resolution(options, item) for item in _within_group_quantiles(options)
    )


def _group_fractions(
    endpoints: tuple[tuple[int, int], ...],
    pinned: dict[int, int],
) -> tuple[tuple[int, ...], ...]:
    """Build evenly distributed resolutions at every eighth from 0% to 100%."""

    flexible = tuple(index for index in range(len(endpoints)) if index not in pinned)
    result: list[tuple[int, ...]] = []
    for numerator in _GROUP_RECOMPUTE_EIGHTHS:
        recompute_count = (
            len(flexible) * numerator + _EIGHTH_DENOMINATOR // 2
        ) // _EIGHTH_DENOMINATOR
        recomputing = {
            flexible[position]
            for position in _evenly_spaced_indices(len(flexible), recompute_count)
        }
        result.append(
            tuple(
                pinned.get(index, recompute if index in recomputing else save)
                for index, (save, recompute) in enumerate(endpoints)
            )
        )
    # Two eighths of a small group count can round to the same resolution;
    # planning it twice would answer nothing new.
    return _unique(result)


def _within_group_quantiles(
    options: TaskAlternativeOptions,
) -> tuple[tuple[int, ...], ...]:
    """Walk each group's own inventory, from least retained to most."""

    pinned = options.pinned
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
                pinned.get(index, option_index)
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


__all__ = ["Resolution", "resolutions", "select"]
