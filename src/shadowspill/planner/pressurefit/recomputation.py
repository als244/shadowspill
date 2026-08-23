"""Deterministic recomputation selections evaluated by PressureFit."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from shadowspill.ir import Program, RecomputationGroup, RecomputationSelection

_EXHAUSTIVE_COMBINATION_LIMIT = 64
_QUARTER_DENOMINATOR = 4
_GROUP_RECOMPUTE_QUARTERS = (0, 1, 2, 3, 4)
_WITHIN_GROUP_MEMORY_QUANTILES = (0, 1, 2, 3, 4)


@dataclass(frozen=True, slots=True)
class _OptionCost:
    option_index: int
    retained_bytes: int
    runtime_ns: int


def build_recomputation_portfolio(
    program: Program,
) -> tuple[tuple[RecomputationSelection, ...], ...]:
    """Return a small deterministic family of legal graph-pair selections.

    Small portfolios remain exhaustive. Large portfolios whose groups expose
    the current binary ``save``/``recompute`` contract use five explicit
    selections: 0%, 25%, 50%, 75%, and 100% of groups recompute. Chosen groups
    are distributed evenly through deterministic group order. This fraction is
    across groups, not within one group's option inventory.

    The existing within-group retained-memory quantiles remain available for a
    future non-binary graph-pair inventory. Both paths are coarse deterministic
    controls rather than an adaptive recomputation search.
    """

    groups = program.recomputation_groups
    if not groups:
        return ((),)

    terminal_save_indices = _terminal_forward_save_indices(program)
    option_indices = tuple(
        (
            (terminal_save_indices[group_index],)
            if group_index in terminal_save_indices
            else tuple(range(len(group.options)))
        )
        for group_index, group in enumerate(groups)
    )
    combination_count = 1
    for group_indices in option_indices:
        combination_count *= len(group_indices)
    if combination_count <= _EXHAUSTIVE_COMBINATION_LIMIT:
        combinations = product(*option_indices)
        return tuple(_selections(groups, item) for item in combinations)

    binary_endpoints = _binary_endpoint_indices(groups)
    if binary_endpoints is not None:
        return tuple(
            _selections(groups, indices)
            for indices in _binary_group_fraction_indices(
                len(groups), binary_endpoints, terminal_save_indices
            )
        )

    costs = _option_costs(program)
    memory_order = tuple(
        tuple(
            item.option_index
            for item in sorted(
                group_costs,
                key=lambda item: (
                    item.retained_bytes,
                    item.runtime_ns,
                    item.option_index,
                ),
            )
        )
        for group_costs in costs
    )
    memory_minimum = tuple(item[0] for item in memory_order)
    compute_minimum = tuple(
        min(
            group_costs,
            key=lambda item: (
                item.runtime_ns,
                item.retained_bytes,
                item.option_index,
            ),
        ).option_index
        for group_costs in costs
    )

    raw: list[tuple[int, ...]] = [
        tuple(0 for _group in groups),
        tuple(len(group.options) - 1 for group in groups),
        memory_minimum,
        compute_minimum,
    ]
    for numerator in _WITHIN_GROUP_MEMORY_QUANTILES:
        raw.append(
            tuple(
                order[(len(order) - 1) * numerator // _QUARTER_DENOMINATOR]
                for order in memory_order
            )
        )
    constrained = [_apply_required_indices(item, terminal_save_indices) for item in raw]
    return tuple(_selections(groups, item) for item in _unique(constrained))


def _terminal_forward_save_indices(program: Program) -> dict[int, int]:
    """Pin every forward-DAG sink group to its structural ``save`` variant."""

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
    result: dict[int, int] = {}
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
            if option.option_id == "save"
        )
        if len(save_indices) != 1:
            raise ValueError(
                "terminal forward recomputation group must expose exactly one "
                f"'save' option: {group.group_id!r}"
            )
        result[group_index] = save_indices[0]
    return result


def _apply_required_indices(
    indices: tuple[int, ...],
    required: dict[int, int],
) -> tuple[int, ...]:
    return tuple(
        required.get(group_index, option_index)
        for group_index, option_index in enumerate(indices)
    )


def _binary_endpoint_indices(
    groups: tuple[RecomputationGroup, ...],
) -> tuple[tuple[int, int], ...] | None:
    """Return each group's ``(save, recompute)`` indices when it is binary."""

    result: list[tuple[int, int]] = []
    for group in groups:
        by_id = {option.option_id: index for index, option in enumerate(group.options)}
        if len(group.options) != 2 or set(by_id) != {"save", "recompute"}:
            return None
        result.append((by_id["save"], by_id["recompute"]))
    return tuple(result)


def _binary_group_fraction_indices(
    group_count: int,
    endpoints: tuple[tuple[int, int], ...],
    required_save_indices: dict[int, int],
) -> tuple[tuple[int, ...], ...]:
    """Build evenly distributed 0/25/50/75/100% recompute selections."""

    if group_count != len(endpoints):
        raise ValueError("binary recomputation endpoint count disagrees with groups")
    flexible_groups = tuple(
        index for index in range(group_count) if index not in required_save_indices
    )
    result: list[tuple[int, ...]] = []
    for numerator in _GROUP_RECOMPUTE_QUARTERS:
        recompute_count = (
            len(flexible_groups) * numerator + _QUARTER_DENOMINATOR // 2
        ) // _QUARTER_DENOMINATOR
        recompute_positions = _evenly_spaced_indices(
            len(flexible_groups), recompute_count
        )
        recompute_groups = {
            flexible_groups[position] for position in recompute_positions
        }
        selection: list[int] = []
        for index, (save, recompute) in enumerate(endpoints):
            if index in required_save_indices:
                selection.append(required_save_indices[index])
            elif index in recompute_groups:
                selection.append(recompute)
            else:
                selection.append(save)
        result.append(tuple(selection))
    return tuple(result)


def _evenly_spaced_indices(total: int, count: int) -> frozenset[int]:
    """Choose ``count`` centered, evenly spaced positions from ``total``."""

    if not 0 <= count <= total:
        raise ValueError("evenly spaced selection count is outside its domain")
    if count == 0:
        return frozenset()
    return frozenset(
        ((2 * ordinal + 1) * total) // (2 * count) for ordinal in range(count)
    )


def _option_costs(program: Program) -> tuple[tuple[_OptionCost, ...], ...]:
    alias_bytes = {
        alias.alias_group_id: alias.size_bytes for alias in program.alias_groups
    }
    profiles = {profile.profile_id: profile for profile in program.profiles}
    tasks = {task.task_id: task for task in program.tasks}
    return tuple(
        tuple(
            _OptionCost(
                option_index=option_index,
                retained_bytes=sum(
                    alias_bytes[alias_id]
                    for alias_id in option.retained_alias_group_ids
                ),
                runtime_ns=sum(
                    profiles[tasks[task_id].profile_id].runtime_ns
                    for task_id in option.active_task_ids
                ),
            )
            for option_index, option in enumerate(group.options)
        )
        for group in program.recomputation_groups
    )


def _selections(
    groups: tuple[RecomputationGroup, ...],
    indices: tuple[int, ...],
) -> tuple[RecomputationSelection, ...]:
    return tuple(
        RecomputationSelection(group.group_id, group.options[index].option_id)
        for group, index in zip(groups, indices, strict=True)
    )


def _unique(values: list[tuple[int, ...]]) -> tuple[tuple[int, ...], ...]:
    result: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return tuple(result)


__all__ = ["build_recomputation_portfolio"]
