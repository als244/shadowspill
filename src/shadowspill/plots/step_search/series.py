"""The views of a search report every figure below reads.

A geometry's point at a budget is its fastest ordering there, and a geometry
is drawn only over the budgets where it planned, so a gap in a line is a
budget that could not fit rather than an interpolation across one.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib

from shadowspill.planner.diagnostics import GraphPairOutcome
from shadowspill.planner.diagnostics.plan import PlanSummary
from shadowspill.search import StepSearchPoint, StepSearchReport

GIB = 1 << 30


def winning_points(report: StepSearchReport) -> tuple[StepSearchPoint, ...]:
    spills = {spill for _execution, spill in report.budgets}
    if len(spills) != 1:
        raise ValueError(
            "step-search figures put the execution budget on the x axis and "
            f"need one spill budget, not {sorted(spills)}"
        )
    winners = report.winners
    if not winners:
        raise ValueError("no budget produced a winning geometry to plot")
    return tuple(sorted(winners, key=lambda point: point.execution_budget_bytes))


@dataclass(frozen=True, slots=True)
class GeometryPoint:
    """One geometry at one budget, with the numbers every figure below reads.

    A geometry's point at a budget is its best ordering there; which one is
    named so the tables and CSVs can say it.
    """

    budget_gib: float
    ordering_label: str
    step_seconds: float
    summary: PlanSummary
    #: Every graph-pair selection the search evaluated at this point.
    graph_pair_selections: tuple[GraphPairOutcome, ...]

    @property
    def wasted_seconds(self) -> float:
        return self.summary.recomputation_overhead_seconds + self.summary.idle_seconds


Series = tuple[tuple[tuple[int, int], tuple[GeometryPoint, ...]], ...]


def geometry_series(report: StepSearchReport) -> Series:
    """Every geometry that planned anywhere, largest microbatch first.

    A geometry's line is drawn only over the budgets where it planned, so a
    gap is a budget it could not fit rather than an interpolation across one.
    """

    grouped: dict[tuple[int, int], dict[float, GeometryPoint]] = {}
    for point in report.points:
        if point.summary is None or point.makespan_seconds is None:
            continue
        key = (point.sequences_per_microbatch, point.accumulation_count)
        candidate = GeometryPoint(
            budget_gib=point.execution_budget_bytes / GIB,
            ordering_label=point.ordering.label,
            step_seconds=point.makespan_seconds,
            summary=point.summary,
            graph_pair_selections=point.graph_pair_selections,
        )
        standing = grouped.setdefault(key, {}).get(candidate.budget_gib)
        # the geometry's point at a budget is its fastest ordering there
        if standing is None or candidate.step_seconds < standing.step_seconds:
            grouped[key][candidate.budget_gib] = candidate
    return tuple(
        (key, tuple(item for _budget, item in sorted(grouped[key].items())))
        for key in sorted(grouped, reverse=True)
    )


def ordering_series(
    report: StepSearchReport, key: tuple[int, int]
) -> tuple[tuple[str, tuple[tuple[float, float], ...]], ...]:
    """One geometry's orderings: label to (budget, step seconds) points."""
    grouped: dict[str, list[tuple[float, float]]] = {}
    for point in report.points:
        if (point.sequences_per_microbatch, point.accumulation_count) != key:
            continue
        if point.makespan_seconds is None:
            continue
        grouped.setdefault(point.ordering.label, []).append(
            (point.execution_budget_bytes / GIB, point.makespan_seconds)
        )
    return tuple((label, tuple(sorted(grouped[label]))) for label in sorted(grouped))


def winning_geometry(series: Series) -> dict[float, tuple[int, int]]:
    """The fastest geometry at each budget, which is the one a run would take.

    Every figure in this family marks it, so a reader can follow one budget's
    actual choice across step time, wasted compute, and distance from the
    floor rather than re-deriving it per figure.
    """

    fastest: dict[float, tuple[float, tuple[int, int]]] = {}
    for key, points in series:
        for item in points:
            standing = fastest.get(item.budget_gib)
            if standing is None or item.step_seconds < standing[0]:
                fastest[item.budget_gib] = (item.step_seconds, key)
    return {budget: key for budget, (_step, key) in fastest.items()}


def geometry_label(key: tuple[int, int]) -> str:
    return f"{key[0]} x {key[1]}"


def geometry_colours(
    series: Series,
) -> dict[tuple[int, int], tuple[float, float, float, float]]:
    """One colour per geometry, shared by every figure in the family.

    A search covers every way of splitting the step, so the count is the
    divisor count of the sequences per step and can exceed any one
    qualitative palette. Wrapping a palette would give two geometries the
    same colour without saying so, which is worse than a less distinct one,
    so the palette grows with the count instead.
    """

    total = len(series)
    if total <= 10:
        palette = matplotlib.colormaps["tab10"]
        pick = [palette(index) for index in range(total)]
    elif total <= 20:
        palette = matplotlib.colormaps["tab20"]
        pick = [palette(index) for index in range(total)]
    else:
        continuous = matplotlib.colormaps["turbo"]
        pick = [continuous(index / (total - 1)) for index in range(total)]
    return {key: pick[index] for index, (key, _points) in enumerate(series)}
