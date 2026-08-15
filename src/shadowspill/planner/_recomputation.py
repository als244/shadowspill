"""Deterministic recomputation selections evaluated by PressureFit."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from shadowspill.ir import Program, RecomputationGroup, RecomputationSelection

_EXHAUSTIVE_COMBINATION_LIMIT = 64
_MEMORY_QUANTILES = (0, 1, 2, 3, 4)


@dataclass(frozen=True, slots=True)
class _OptionCost:
    option_index: int
    retained_bytes: int
    runtime_ns: int


def build_recomputation_portfolio(
    program: Program,
) -> tuple[tuple[RecomputationSelection, ...], ...]:
    """Return a small deterministic family of legal graph-pair selections.

    Small portfolios remain exhaustive. Large portfolios use a bounded seed
    family: canonical endpoints, retained-memory quantiles, the fastest
    endpoint, and both alternating memory/compute parities. Every seed is
    derived from framework-neutral Program costs and is simulator-verified by
    PressureFit; no model, frontend, or operation identity enters selection.
    """

    groups = program.recomputation_groups
    if not groups:
        return ((),)

    option_counts = tuple(len(group.options) for group in groups)
    combination_count = 1
    for count in option_counts:
        combination_count *= count
    if combination_count <= _EXHAUSTIVE_COMBINATION_LIMIT:
        indices = product(*(range(count) for count in option_counts))
        return tuple(_selections(groups, item) for item in indices)

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
        tuple(count - 1 for count in option_counts),
        memory_minimum,
        compute_minimum,
    ]
    for numerator in _MEMORY_QUANTILES:
        raw.append(
            tuple(
                order[(len(order) - 1) * numerator // 4]
                for order in memory_order
            )
        )
    raw.extend(
        (
            tuple(
                memory_minimum[index]
                if index % 2 == 0
                else compute_minimum[index]
                for index in range(len(groups))
            ),
            tuple(
                compute_minimum[index]
                if index % 2 == 0
                else memory_minimum[index]
                for index in range(len(groups))
            ),
        )
    )
    return tuple(_selections(groups, item) for item in _unique(raw))


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
