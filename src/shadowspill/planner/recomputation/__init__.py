"""Which combinations of graph-pair options are worth evaluating.

`options` says what each graph pair offers and what each alternative costs.
This decides which combinations of those alternatives PressureFit should
actually plan, which is a different question: the inventory is a fact about the
Program, and this is a search policy over it.

The policy is coarse and deterministic rather than adaptive. Small inventories
are evaluated exhaustively. Larger ones whose every group is the current binary
`save`/`recompute` pair get five selections - 0%, 25%, 50%, 75% and 100% of
groups recomputing - distributed evenly through deterministic group order. That
fraction is *across* groups, not within one group's inventory.

The within-group quantiles below are the path a non-binary inventory would
take, and are reached today only by a Program whose groups are not binary.
"""

from __future__ import annotations

from itertools import product

from shadowspill.ir import Program, RecomputationSelection

from .options import GraphPairOptions

_EXHAUSTIVE_COMBINATION_LIMIT = 64
_QUARTER_DENOMINATOR = 4
_GROUP_RECOMPUTE_QUARTERS = (0, 1, 2, 3, 4)
_WITHIN_GROUP_MEMORY_QUANTILES = (0, 1, 2, 3, 4)


def build_recomputation_portfolio(
    program: Program,
) -> tuple[tuple[RecomputationSelection, ...], ...]:
    """Return a small deterministic family of legal graph-pair selections."""

    if not program.recomputation_groups:
        return ((),)
    return select(GraphPairOptions.from_program(program))


def select(
    options: GraphPairOptions,
) -> tuple[tuple[RecomputationSelection, ...], ...]:
    """Choose which combinations of the inventory to plan."""

    if not options.groups:
        return ((),)

    pinned = options.pinned
    if options.combination_count <= _EXHAUSTIVE_COMBINATION_LIMIT:
        per_group = tuple(
            (pinned[index],) if index in pinned else tuple(range(len(group.options)))
            for index, group in enumerate(options.groups)
        )
        return tuple(_selections(options, item) for item in product(*per_group))

    endpoints = options.binary_endpoints
    if endpoints is not None:
        return tuple(
            _selections(options, indices)
            for indices in _group_fractions(endpoints, pinned)
        )
    return tuple(
        _selections(options, item) for item in _within_group_quantiles(options)
    )


def _group_fractions(
    endpoints: tuple[tuple[int, int], ...],
    pinned: dict[int, int],
) -> tuple[tuple[int, ...], ...]:
    """Build evenly distributed 0/25/50/75/100% recompute selections."""

    flexible = tuple(index for index in range(len(endpoints)) if index not in pinned)
    result: list[tuple[int, ...]] = []
    for numerator in _GROUP_RECOMPUTE_QUARTERS:
        recompute_count = (
            len(flexible) * numerator + _QUARTER_DENOMINATOR // 2
        ) // _QUARTER_DENOMINATOR
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
    return tuple(result)


def _within_group_quantiles(
    options: GraphPairOptions,
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


def _selections(
    options: GraphPairOptions,
    indices: tuple[int, ...],
) -> tuple[RecomputationSelection, ...]:
    return tuple(
        RecomputationSelection(group.group_id, group.options[index].option_id)
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


__all__ = ["build_recomputation_portfolio", "select"]
