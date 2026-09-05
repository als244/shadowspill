"""Figures over a geometry search, with the execution budget on the x axis.

Two families, written as PNG files into a directory. The first follows each
budget's winning geometry: throughput and raw step time; recomputation and
stall overheads, raw and as shares of the simulated step; and fetch/evict
traffic, raw and as simulated lane utilization (bytes over assumed bandwidth
over step time).

The second draws every geometry as its own line, in one colour per geometry
held across the family, so a budget can be read as a choice between them
rather than only through the winner: simulated step time with the best
geometry at each budget circled, the same overhead split three ways per
geometry, and how close each geometry comes to its own compute floor.

Every value is plan-side, read from a point's :class:`PlanSummary`, so the
whole set renders from a search with nothing executed.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.patheffects import withStroke
from matplotlib.ticker import LogLocator, MaxNLocator, NullFormatter, NullLocator

from shadowspill.planner.diagnostics.plan import PlanSummary
from shadowspill.pytorch.step_search import (
    GraphPairOutcome,
    StepSearchPoint,
    StepSearchReport,
)

_GIB = 1 << 30


def _winners(report: StepSearchReport) -> tuple[StepSearchPoint, ...]:
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


def _budget_ticks(axes: Axes, budgets: Sequence[float]) -> None:
    """Tick the budgets that were searched, not a round interpolation of them.

    A budget is a value someone chose, so a tick between two of them names a
    point the search never visited. Crowded ladders lean their labels rather
    than dropping them.
    """

    values = sorted(set(budgets))
    axes.set_xticks(values)
    axes.set_xticklabels(
        [f"{value:g}" for value in values],
        rotation=45 if len(values) > 8 else 0,
        ha="right" if len(values) > 8 else "center",
    )


def _figure(
    path: Path,
    title: str,
    ylabel: str,
    budgets_gib: list[float],
    series: dict[str, list[float]],
    *,
    percent: bool = False,
) -> Path:
    figure = Figure(figsize=(6.4, 4.0), dpi=150)
    axes = figure.subplots()
    for label, values in series.items():
        axes.plot(budgets_gib, values, marker="o", label=label)
    _budget_ticks(axes, budgets_gib)
    axes.set_title(title)
    axes.set_xlabel("Execution Budget (GiB)")
    axes.set_ylabel(ylabel)
    axes.grid(True, alpha=0.3)
    if percent:
        axes.set_ylim(bottom=0.0)
        axes.yaxis.set_major_formatter(lambda value, _pos: f"{value:.0%}")
    if len(series) > 1:
        axes.legend()
    figure.tight_layout()
    figure.savefig(path)
    return path


@dataclass(frozen=True, slots=True)
class _GeometryPoint:
    """One geometry at one budget, with the numbers every figure below reads."""

    budget_gib: float
    step_seconds: float
    summary: PlanSummary
    #: Every graph-pair selection the search evaluated at this point.
    graph_pair_selections: tuple[GraphPairOutcome, ...]

    @property
    def wasted_seconds(self) -> float:
        return self.summary.recomputation_overhead_seconds + self.summary.idle_seconds


_Series = tuple[tuple[tuple[int, int], tuple[_GeometryPoint, ...]], ...]


def _geometry_series(report: StepSearchReport) -> _Series:
    """Every geometry that planned anywhere, largest microbatch first.

    A geometry's line is drawn only over the budgets where it planned, so a
    gap is a budget it could not fit rather than an interpolation across one.
    """

    grouped: dict[tuple[int, int], list[_GeometryPoint]] = {}
    for point in report.points:
        if point.summary is None or point.makespan_seconds is None:
            continue
        key = (point.sequences_per_microbatch, point.accumulation_count)
        grouped.setdefault(key, []).append(
            _GeometryPoint(
                budget_gib=point.execution_budget_bytes / _GIB,
                step_seconds=point.makespan_seconds,
                summary=point.summary,
                graph_pair_selections=point.graph_pair_selections,
            )
        )
    return tuple(
        (key, tuple(sorted(grouped[key], key=lambda item: item.budget_gib)))
        for key in sorted(grouped, reverse=True)
    )


def _geometry_colours(
    series: _Series,
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


def _label(key: tuple[int, int]) -> str:
    return f"{key[0]} x {key[1]}"


def _winning_geometry(series: _Series) -> dict[float, tuple[int, int]]:
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


def _circle_winners(
    axes: Axes,
    series: _Series,
    winners: dict[float, tuple[int, int]],
    measure: Callable[[_GeometryPoint], float],
) -> None:
    """Ring the winning geometry's own value at each budget."""

    marked = [
        (item.budget_gib, measure(item))
        for key, points in series
        for item in points
        if winners.get(item.budget_gib) == key
    ]
    marked.sort()
    axes.scatter(
        [budget for budget, _value in marked],
        [value for _budget, value in marked],
        s=170,
        facecolors="none",
        edgecolors="black",
        linewidths=1.2,
        zorder=5,
        label="Best at This Budget",
    )


#: Blank pitch between one budget's group of bars and the next.
_GROUP_GAP = 0.8

#: Spread past which a linear axis flattens the fast geometries into the floor.
_LOG_SPREAD = 4.0


def _mark_floor(axes: Axes, value: float) -> None:
    """Draw the best attainable value as a heavier line than the grid.

    Every one of these figures has an ideal its lines approach from one side,
    and reading how close a geometry is to it is the point. A gridline of the
    same weight as the others does not say which one that is.
    """

    axes.axhline(value, color="0.25", linewidth=1.8, zorder=1.5)


def _log_scale(axes: Axes, values: list[float], *, floor: float | None) -> bool:
    """Spread the axis logarithmically when a linear one would hide most of it.

    One slow geometry can be a thousand times another, which presses every
    fast line onto the axis floor. Where the ideal is zero the scale is
    symmetric-log, whose linear region below the smallest measured value is
    what lets zero be a tick at all: a plain log axis puts it at negative
    infinity and cannot draw it.
    """

    if not values or any(value <= 0.0 for value in values):
        return False
    if max(values) / min(values) < _LOG_SPREAD:
        return False
    smallest, largest = min(values), max(values)
    if floor == 0.0:
        digits = math.floor(math.log10(smallest))
        threshold = math.floor(smallest / 10.0**digits) * 10.0**digits
        axes.set_yscale("symlog", linthresh=threshold, linscale=0.25)
        ticks = [0.0] + [
            tick
            for tick in LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)).tick_values(
                threshold, largest
            )
            if threshold <= tick <= largest * 1.05
        ]
        axes.set_yticks(ticks)
        axes.set_ylim(bottom=0.0)
    else:
        axes.set_yscale("log")
        # A range under a decade gets one labelled tick from the default
        # locator, so ask for the subdivisions too.
        axes.yaxis.set_major_locator(
            LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 5.0, 7.0))
        )
    # "%g" rather than the default, which rounds 0.07 and 0.1 to the same
    # label and prints two identical ticks.
    axes.yaxis.set_major_formatter(lambda value, _pos: f"{value:g}")
    axes.yaxis.set_minor_locator(NullLocator())
    axes.yaxis.set_minor_formatter(NullFormatter())
    return True


#: How far above the fastest step the detail panel reaches. Wide enough to
#: hold every geometry worth choosing, narrow enough to separate them.
_DETAIL_SPAN = 1.25


def _geometry_step_time(
    path: Path,
    report: StepSearchReport,
    series: _Series,
    colours: dict[tuple[int, int], tuple[float, float, float, float]],
    best_geometry: dict[float, tuple[int, int]],
) -> Path:
    """Simulated throughput per geometry, with the best at each budget circled.

    Throughput leads and step time is the relabelled twin, because throughput
    is what a budget is chosen for and it reads the right way up: higher is
    better. The two are the same measurement, so the right axis is the same
    line under another name.

    Two panels over one x axis. The upper one holds every geometry, which
    means a slow split compresses the fast ones into a band; the lower one is
    that band, linear and to itself, because the geometries a reader is
    choosing between are exactly the ones the full range cannot separate.
    """

    tokens = report.tokens_per_step

    def rate(item: _GeometryPoint) -> float:
        return tokens / item.step_seconds

    figure = Figure(figsize=(7.6, 5.6), dpi=150)
    overview, detail = figure.subplots(2, 1, sharex=True, height_ratios=(2.0, 1.0))
    for axes in (overview, detail):
        for key, points in series:
            axes.plot(
                [item.budget_gib for item in points],
                [rate(item) for item in points],
                marker="o",
                markersize=4,
                color=colours[key],
                label=_label(key) if axes is overview else None,
            )
        _circle_winners(axes, series, best_geometry, rate)

    rates = [rate(item) for _key, points in series for item in points]
    overview.set_title("Simulated Throughput by Geometry")
    overview.set_ylabel("Tokens per Second")
    logarithmic = _log_scale(overview, rates, floor=None)
    overview.grid(True, alpha=0.3, which="both")
    handles, labels = overview.get_legend_handles_labels()
    overview.legend(handles, labels, fontsize="small", ncols=2)

    # The right axis is the same measurement read the other way round, so it
    # is a relabelled twin rather than a second series. Its ticks are chosen
    # in seconds and mapped back, because the reciprocal of evenly spaced
    # rates bunches into an unreadable smear at the slow end.
    low, high = overview.get_ylim()
    step_time = overview.twinx()
    if logarithmic:
        step_time.set_yscale("log")
    step_time.set_ylim(low, high)
    seconds = MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]).tick_values(
        tokens / high, tokens / low
    )
    inside = [value for value in seconds if value > 0 and low <= tokens / value <= high]
    if inside:
        step_time.set_yticks([tokens / value for value in inside])
        step_time.set_yticklabels([f"{value:g}" for value in inside])
    step_time.yaxis.set_minor_locator(NullLocator())
    step_time.yaxis.set_minor_formatter(NullFormatter())
    step_time.set_ylabel("Seconds per Step")

    fastest = max(rates)
    detail.set_ylim(fastest / _DETAIL_SPAN, fastest * 1.02)
    _budget_ticks(
        detail, [item.budget_gib for _key, points in series for item in points]
    )
    detail.set_xlabel("Execution Budget (GiB)")
    detail.set_ylabel("Tokens per Second (Detail)")
    detail.yaxis.set_major_locator(MaxNLocator(nbins=8, steps=[1, 2, 2.5, 5, 10]))
    detail.grid(True, alpha=0.3)
    detail.set_title(
        f"Within {_DETAIL_SPAN:g}x of the Best Throughput", fontsize="small"
    )
    figure.tight_layout()
    figure.savefig(path)
    return path


#: Room one label needs, in display points: comfortably more than the text's
#: own height, so that two segments of nearly the same size are either both
#: labelled or both not, and the width of the widest number these charts
#: print.
_LABEL_HEIGHT = 24.0
_LABEL_WIDTH = 32.0


def _label_segments(
    axes: Axes,
    segments: list[tuple[float, float, float, float]],
    bar_width: float,
) -> None:
    """Write each segment's value inside it, where the segment can hold it.

    A segment too short or a bar too narrow for the text would print over its
    neighbours, so it goes unlabelled rather than illegible. That is what
    makes the figure degrade rather than break as a search covers more
    budgets or more geometries. The white stroke keeps the digits readable
    over both opacities of every colour.
    """

    origin, step = axes.transData.transform([(0.0, 0.0), (bar_width, 0.0)])
    if step[0] - origin[0] < _LABEL_WIDTH:
        return
    for centre, top, bottom, value in segments:
        low, high = axes.transData.transform([(centre, bottom), (centre, top)])
        if high[1] - low[1] < _LABEL_HEIGHT:
            continue
        axes.text(
            centre,
            (bottom + top) / 2.0,
            f"{value:.1f}",
            ha="center",
            va="center",
            fontsize=9.0,
            color="0.1",
            path_effects=[withStroke(linewidth=1.6, foreground="white")],
            zorder=6,
        )


def _geometry_waste_bars(
    path: Path,
    series: _Series,
    colours: dict[tuple[int, int], tuple[float, float, float, float]],
    best_geometry: dict[float, tuple[int, int]],
    *,
    share: bool,
) -> Path:
    """The same waste as bars: one group per budget, one bar per geometry.

    Each bar is the total, split into the recomputation the plan chose and
    the stall it could not avoid, so a bar's height compares geometries and
    its two parts say which cause moved. Both segments carry their time in
    seconds, which is the number the share view would otherwise lose.

    The budgets are categories rather than a numeric axis, so they space
    evenly however unevenly they were chosen. Seconds are log-spaced,
    because one geometry wastes a thousand times another; shares are linear,
    because they already span one decade and a stacked bar reads honestly
    only on a linear axis, where a segment's drawn thickness is its value.
    """

    # Left to right within a group is ascending microbatch, the opposite of
    # the order the colours were assigned in, so the bars read smallest to
    # largest while a geometry keeps its colour across every figure.
    ordered = tuple(reversed(series))
    width = 0.88
    occupied: dict[float, set[int]] = {}
    for offset, (_key, points) in enumerate(ordered):
        for item in points:
            occupied.setdefault(item.budget_gib, set()).add(offset)
    centres, ticks, extent = _packed_groups(occupied, _GROUP_GAP)

    # Wide enough that a segment's label clears its neighbour's, up to a
    # width a reader can still scan. Past that the bars narrow and the
    # labels drop out on their own rather than the file growing without
    # bound.
    figure = Figure(figsize=(_packed_width(extent, 0.42, 30.0), 5.2), dpi=150)
    axes = figure.subplots()
    drawn: list[float] = []
    labels: list[tuple[float, float, float, float]] = []
    totals: list[tuple[float, float]] = []
    makespans: list[tuple[float, float, float]] = []
    for offset, (key, points) in enumerate(ordered):
        for item in points:
            scale = item.step_seconds if share else 1.0
            recompute = item.summary.recomputation_overhead_seconds / scale
            idle = item.summary.idle_seconds / scale
            drawn += [recompute, recompute + idle]
            centre = centres[(item.budget_gib, offset)]
            axes.bar(centre, recompute, width=width, color=colours[key], alpha=1.0)
            axes.bar(
                centre,
                idle,
                width=width,
                bottom=recompute,
                color=colours[key],
                alpha=0.45,
            )
            if best_geometry.get(item.budget_gib) == key:
                # One outline around the whole bar rather than around each
                # segment, which would draw a line along the boundary between
                # them and read as a third division.
                axes.bar(
                    centre,
                    recompute + idle,
                    width=width,
                    fill=False,
                    edgecolor="black",
                    linewidth=1.6,
                    zorder=4,
                )
            labels.append(
                (
                    centre,
                    recompute,
                    0.0,
                    item.summary.recomputation_overhead_seconds,
                )
            )
            labels.append(
                (
                    centre,
                    recompute + idle,
                    recompute,
                    item.summary.idle_seconds,
                )
            )
            totals.append((centre, recompute + idle))
            makespans.append((centre, recompute + idle, item.step_seconds))

    axes.set_title(
        "Recomputation and Stalls, Share of the Step"
        if share
        else "Recomputation and Stalls"
    )
    axes.set_xlabel("Execution Budget (GiB)")
    axes.set_ylabel("Share of Simulated Step" if share else "Seconds")
    drawn_budgets = sorted(ticks)
    axes.set_xticks([ticks[budget] for budget in drawn_budgets])
    axes.set_xticklabels([f"{budget:g}" for budget in drawn_budgets])
    if share:
        axes.set_ylim(0.0, max(drawn) * 1.30)
        axes.yaxis.set_major_locator(MaxNLocator(nbins=12, steps=[1, 2, 2.5, 5, 10]))
        axes.yaxis.set_major_formatter(lambda item, _pos: f"{item * 100:g}%")
    else:
        # _log_scale already labels a 1-2-5 ladder and keeps zero as a tick;
        # overriding its locator here would drop the zero and unclamp the
        # bottom, which is where every bar starts.
        _log_scale(axes, drawn, floor=0.0)
        axes.set_ylim(0.0, max(drawn) * 1.55)
    _mark_floor(axes, 0.0)
    axes.grid(True, axis="y", alpha=0.3, which="major")
    axes.set_axisbelow(True)
    _label_segments(axes, labels, width)
    for centre, total in totals:
        axes.annotate(
            f"{total * 100:.0f}%" if share else f"{total:.1f}",
            (centre, total),
            textcoords="offset points",
            xytext=(0, 4),
            ha="center",
            va="bottom",
            fontsize=10.0,
            color="0.1",
        )
    _annotate_makespans(axes, makespans, width)

    handles: list[Patch] = [
        Patch(facecolor=colours[key], label=_label(key)) for key, _points in ordered
    ]
    handles += [
        Patch(facecolor="0.35", alpha=0.45, label="Stalled (Upper)"),
        Patch(facecolor="0.35", alpha=1.0, label="Recompute (Lower)"),
        Patch(facecolor="none", edgecolor="none", label="Makespan"),
        Patch(
            facecolor="none",
            edgecolor="black",
            linewidth=1.6,
            label="Minimum Makespan",
        ),
    ]
    legend = axes.legend(
        handles=handles,
        fontsize="x-small",
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        borderaxespad=0.0,
    )
    for entry in legend.get_texts():
        if entry.get_text() == "Makespan":
            entry.set_color("tab:red")
    figure.tight_layout()
    figure.savefig(path)
    return path


def _recompute_labels(levels: Sequence[int], groups: int) -> dict[int, str]:
    """Name each rung of the ladder by the share of groups it recomputes.

    Rounded to a quarter, because "50% of Groups" reads at a glance where
    "17 of 36" does not and the exact figure is not what a reader is
    comparing. Rounding falls back to the true share if it would give two
    rungs the same name, since a legend with a repeated key is worse than
    one with an awkward number.
    """

    if not groups:
        return {level: f"{level} Recomputing" for level in levels}
    rounded = {level: round(level / groups * 4) * 25 for level in levels}
    if len(set(rounded.values())) != len(levels):
        rounded = {level: round(level / groups * 100) for level in levels}
    return {
        level: f"{share}% of Groups Recomputing" for level, share in rounded.items()
    }


def _packed_width(extent: float, per_bar: float, cap: float) -> float:
    """How wide a packed chart needs to be, in inches.

    The extent is in bar pitches, so a budget where only one series planned
    costs one pitch rather than a full group's worth of blank.
    """

    return min(2.8 + per_bar * extent, cap)


def _packed_groups(
    occupied: dict[float, set[int]], gap: float
) -> tuple[dict[tuple[float, int], float], dict[float, float], float]:
    """Lay the groups out left to right, each only as wide as it needs.

    Reserving a slot per series would push a group of one bar to the edge of
    a grid sized for five and leave the rest of the category blank. Each
    group instead takes room for the bars it actually has, in series order,
    with a constant gap between groups. The budgets are categories rather
    than a numeric axis, so uneven pitch costs nothing and the whitespace
    goes away.

    Returns the centre of every bar, keyed by budget and series index, the
    centre of each group, which is where its tick belongs, and how far the
    whole layout reaches.
    """

    centres: dict[tuple[float, int], float] = {}
    ticks: dict[float, float] = {}
    cursor = 0.0
    for budget in sorted(occupied):
        offsets = sorted(occupied[budget])
        if not offsets:
            continue
        for position, offset in enumerate(offsets):
            centres[(budget, offset)] = cursor + position + 0.5
        ticks[budget] = cursor + len(offsets) / 2.0
        cursor += len(offsets) + gap
    return centres, ticks, max(cursor - gap, 1.0)


def _selection_waste(
    path: Path,
    key: tuple[int, int],
    points: tuple[_GeometryPoint, ...],
    *,
    share: bool,
) -> Path | None:
    """One geometry's waste, grouped by graph-pair selection.

    The other figures compare geometries under the plan the search answered
    with. This one opens that answer up: at each budget it shows every
    graph-pair selection the search evaluated, so a reader can see whether
    the winning level beat the others by a hair or by a factor, and where the
    rest stopped fitting at all.

    A selection with no plan leaves a gap, which is the useful negative
    result: at a tight budget only the most aggressive recomputation fits.
    """

    levels = sorted(
        {
            outcome.recompute_groups
            for item in points
            for outcome in item.graph_pair_selections
        }
    )
    if not levels:
        return None
    width = 0.88
    shades = matplotlib.colormaps["viridis"]
    occupied: dict[float, set[int]] = {}
    for item in points:
        for outcome in item.graph_pair_selections:
            if outcome.waiting_seconds is not None:
                occupied.setdefault(item.budget_gib, set()).add(
                    levels.index(outcome.recompute_groups)
                )
    centres, ticks, extent = _packed_groups(occupied, _GROUP_GAP)

    figure = Figure(figsize=(_packed_width(extent, 0.42, 30.0), 5.2), dpi=150)
    axes = figure.subplots()
    drawn: list[float] = []
    labels: list[tuple[float, float, float, float]] = []
    totals: list[tuple[float, float]] = []
    makespans: list[tuple[float, float, float]] = []
    for item in points:
        for outcome in item.graph_pair_selections:
            waiting = outcome.waiting_seconds
            if waiting is None:
                continue
            offset = levels.index(outcome.recompute_groups)
            centre = centres[(item.budget_gib, offset)]
            scale = outcome.makespan_seconds or 1.0
            recompute = outcome.recomputation_overhead_seconds / (
                scale if share else 1.0
            )
            wait = waiting / (scale if share else 1.0)
            drawn += [recompute, recompute + wait]
            colour = shades(offset / max(len(levels) - 1, 1))
            axes.bar(centre, recompute, width=width, color=colour, alpha=1.0)
            axes.bar(
                centre, wait, width=width, bottom=recompute, color=colour, alpha=0.45
            )
            labels.append(
                (centre, recompute, 0.0, outcome.recomputation_overhead_seconds)
            )
            labels.append((centre, recompute + wait, recompute, waiting))
            totals.append((centre, recompute + wait))
            if outcome.makespan_seconds is not None:
                makespans.append((centre, recompute + wait, outcome.makespan_seconds))

    if not drawn:
        return None
    microbatch, accumulation = key
    axes.set_title(
        "Recompute and Stall by Graph-Pair Selection"
        + (", Share of the Step" if share else "")
        + f": {microbatch} x {accumulation}"
    )
    axes.set_xlabel("Execution Budget (GiB)")
    axes.set_ylabel("Share of Makespan" if share else "Seconds")
    drawn_budgets = sorted(ticks)
    axes.set_xticks([ticks[budget] for budget in drawn_budgets])
    axes.set_xticklabels([f"{budget:g}" for budget in drawn_budgets])
    if share:
        axes.set_ylim(0.0, max(drawn) * 1.30)
        axes.yaxis.set_major_locator(MaxNLocator(nbins=12, steps=[1, 2, 2.5, 5, 10]))
        axes.yaxis.set_major_formatter(lambda item, _pos: f"{item * 100:g}%")
    else:
        _log_scale(axes, drawn, floor=0.0)
        axes.set_ylim(0.0, max(drawn) * 1.55)
    _mark_floor(axes, 0.0)
    axes.grid(True, axis="y", alpha=0.3, which="major")
    axes.set_axisbelow(True)
    _label_segments(axes, labels, width)
    for centre, total in totals:
        axes.annotate(
            f"{total * 100:.0f}%" if share else f"{total:.1f}",
            (centre, total),
            textcoords="offset points",
            xytext=(0, 4),
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="0.1",
        )
    _annotate_makespans(axes, makespans, width)

    groups = max(
        (
            outcome.group_count
            for item in points
            for outcome in item.graph_pair_selections
        ),
        default=0,
    )
    names = _recompute_labels(levels, groups)
    handles: list[Patch] = [
        Patch(
            facecolor=shades(index / max(len(levels) - 1, 1)),
            label=names[level],
        )
        for index, level in enumerate(levels)
    ]
    handles += [
        Patch(facecolor="0.35", alpha=0.45, label="Stalled (Upper)"),
        Patch(facecolor="0.35", alpha=1.0, label="Recompute (Lower)"),
        Patch(facecolor="none", edgecolor="none", label="Makespan"),
    ]
    legend = axes.legend(
        handles=handles,
        fontsize="x-small",
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        borderaxespad=0.0,
    )
    for entry in legend.get_texts():
        if entry.get_text() == "Makespan":
            entry.set_color("tab:red")
    figure.tight_layout()
    figure.savefig(path)
    return path


def _label_halves(axes: Axes) -> None:
    """Name the two halves of a mirrored chart on the axis itself.

    A legend entry says which opacity is which, but a reader looking at the
    lower half should not have to go and find it. The words sit outside the
    frame, centred on each half, where no bar can reach them however the
    data falls.
    """

    low, high = axes.get_ylim()
    for name, span in (("Fetch", (0.0, high)), ("Evict", (low, 0.0))):
        middle = (span[0] + span[1]) / 2.0
        axes.annotate(
            name,
            (0.0, middle),
            xycoords=("axes fraction", "data"),
            textcoords="offset points",
            xytext=(-38, 0),
            rotation=90,
            ha="center",
            va="center",
            fontsize=10.0,
            fontweight="bold",
            color="0.35",
            annotation_clip=False,
        )


def _transfer_bars(
    path: Path,
    series: _Series,
    colours: dict[tuple[int, int], tuple[float, float, float, float]],
    best_geometry: dict[float, tuple[int, int]],
    *,
    share: bool,
) -> Path | None:
    """What each geometry moves, fetch and evict, per budget.

    The winner-following transfer figures answer what the chosen plan moves.
    This one asks what the choice costs: a geometry that halves the
    microbatch fetches the same parameters for twice as many rounds, and the
    lane is what it pays with.

    Stacking would be wrong for utilization: fetch and evict are separate
    lanes running at the same time, so their shares are each out of one
    lane's seconds and a total of them is not a quantity. Mirroring keeps
    both readable against the same scale, spends one bar's width instead of
    two, and puts the comparison a reader wants -- how lopsided the two
    directions are -- on the axis itself.
    """

    ordered = tuple(reversed(series))
    # Fourteen bars to a group where the waste figure has seven, so the
    # group takes nearly the whole category and the pair nearly the whole
    # slot; anything less leaves bars too thin to carry their own numbers.
    width = 0.86
    occupied: dict[float, set[int]] = {}
    for offset, (_key, points) in enumerate(ordered):
        for item in points:
            occupied.setdefault(item.budget_gib, set()).add(offset)
    centres, ticks, extent = _packed_groups(occupied, _GROUP_GAP)

    # Fourteen bars to a group against the waste figure's seven, so the room
    # comes from height as well as width rather than a letterbox.
    figure = Figure(figsize=(_packed_width(extent, 0.42, 30.0), 7.4), dpi=150)
    axes = figure.subplots()
    above: list[float] = []
    below: list[float] = []
    totals: list[tuple[float, float]] = []
    makespans: list[tuple[float, float, float]] = []
    for offset, (key, points) in enumerate(ordered):
        for item in points:
            summary = item.summary
            if share:
                fetch = (
                    summary.transfer_bytes_fetched
                    / summary.fetch_bandwidth_bytes_per_second
                    / item.step_seconds
                )
                evict = (
                    summary.transfer_bytes_evicted
                    / summary.evict_bandwidth_bytes_per_second
                    / item.step_seconds
                )
            else:
                fetch = summary.transfer_bytes_fetched / _GIB
                evict = summary.transfer_bytes_evicted / _GIB
            centre = centres[(item.budget_gib, offset)]
            won = best_geometry.get(item.budget_gib) == key
            above.append(fetch)
            below.append(evict)
            for value, alpha in ((fetch, 1.0), (-evict, 0.45)):
                axes.bar(centre, value, width=width, color=colours[key], alpha=alpha)
                if won:
                    axes.bar(
                        centre,
                        value,
                        width=width,
                        fill=False,
                        edgecolor="black",
                        linewidth=1.6,
                        zorder=4,
                    )
                totals.append((centre, value))
            makespans.append((centre, fetch, item.step_seconds))

    axes.set_title(
        "Lane Utilization by Geometry" if share else "Transfer Traffic by Geometry"
    )
    axes.set_xlabel("Execution Budget (GiB)")
    axes.set_ylabel("Share of Lane-Seconds" if share else "GiB per Step", labelpad=26)
    drawn_budgets = sorted(ticks)
    axes.set_xticks([ticks[budget] for budget in drawn_budgets])
    axes.set_xticklabels([f"{budget:g}" for budget in drawn_budgets])
    # Half a gap at each end, so the outer budgets are spaced like the
    # inner ones rather than pinned to the frame.
    axes.set_xlim(-_GROUP_GAP / 2, extent + _GROUP_GAP / 2)
    # The two halves share a scale but not an extent: an evict lane that
    # never passes 40% should not leave the bottom half of the figure empty.
    # A lane cannot exceed its own seconds, so the ticks stop at 100%, and
    # the view reaches a little past so a full bar's number has room.
    if share:
        # Ticks first: a fixed locator carrying values past the limits pulls
        # the view out to reach them, so the limit has to be set last.
        axes.set_yticks([value / 10.0 for value in range(-10, 11)])
        axes.yaxis.set_major_formatter(lambda item, _pos: f"{abs(item) * 100:g}%")
        axes.set_ylim(-max(below) * 1.20, min(1.26, max(above) + 0.19))
    else:
        axes.set_ylim(-max(below) * 1.20, max(above) * 1.30)
        axes.yaxis.set_major_formatter(lambda item, _pos: f"{abs(item):g}")
    # What the lane cost bought: the step this geometry's plan reached, over
    # its fetch bar, in the one colour nothing else on the figure uses.
    _annotate_makespans(axes, makespans, width)
    axes.axhline(0.0, color="0.25", linewidth=1.4, zorder=3)
    _label_halves(axes)
    axes.grid(True, axis="y", alpha=0.3, which="major")
    axes.set_axisbelow(True)
    _annotate_bars(axes, totals, width, share=share)

    handles: list[Patch] = [
        Patch(facecolor=colours[key], label=_label(key)) for key, _points in ordered
    ]
    handles += [
        Patch(facecolor="0.35", alpha=1.0, label="Fetch (Above)"),
        Patch(facecolor="0.35", alpha=0.45, label="Evict (Below)"),
        Patch(facecolor="none", edgecolor="none", label="Makespan"),
        Patch(
            facecolor="none",
            edgecolor="black",
            linewidth=1.6,
            label="Minimum Makespan",
        ),
    ]
    legend = axes.legend(
        handles=handles,
        fontsize="x-small",
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        borderaxespad=0.0,
    )
    for entry in legend.get_texts():
        if entry.get_text() == "Makespan":
            entry.set_color("tab:red")
    figure.tight_layout()
    figure.savefig(path)
    return path


def _annotate_makespans(
    axes: Axes, entries: list[tuple[float, float, float]], bar_width: float
) -> None:
    """Write what each bar's plan cost in time, over the bar.

    Red because nothing else on these figures uses it: the palette is the
    series and the two opacities are the directions, so a third channel was
    free. The text turns upright when the bars are too narrow to hold it
    side by side, which is what a seven-series figure with three-digit
    seconds needs and a five-series one with two digits does not.
    """

    if not entries:
        return
    origin, step = axes.transData.transform([(0.0, 0.0), (bar_width, 0.0)])
    room = step[0] - origin[0]
    if room < _LABEL_WIDTH * 0.72:
        return
    widest = max(len(f"{seconds:.1f} s") for _centre, _top, seconds in entries)
    upright = room < widest * 6.2
    for centre, top, seconds in entries:
        axes.annotate(
            f"{seconds:.1f} s",
            (centre, top),
            textcoords="offset points",
            xytext=(0, 21),
            rotation=90 if upright else 0,
            ha="center",
            va="bottom",
            fontsize=9.0,
            color="tab:red",
        )


def _annotate_bars(
    axes: Axes,
    values: list[tuple[float, float]],
    bar_width: float,
    *,
    share: bool,
    inside: bool = False,
) -> None:
    """Write each bar's value on it, where the bar is wide enough.

    A bar narrower than the number would print over its neighbour, so it
    goes unlabelled rather than illegible. `inside` puts the number just
    under the bar's top instead of above it, for an axis that stops at a
    hard ceiling and has no room above a full bar.
    """

    origin, step = axes.transData.transform([(0.0, 0.0), (bar_width, 0.0)])
    if step[0] - origin[0] < _LABEL_WIDTH * 0.72:
        return
    for centre, value in values:
        text = f"{abs(value) * 100:.0f}%" if share else f"{abs(value):.0f}"
        if value < 0.0:
            axes.annotate(
                text,
                (centre, value),
                textcoords="offset points",
                xytext=(0, -4),
                ha="center",
                va="top",
                fontsize=9.5,
                color="0.1",
            )
        elif inside:
            axes.annotate(
                text,
                (centre, value),
                textcoords="offset points",
                xytext=(0, -4),
                ha="center",
                va="top",
                fontsize=9.5,
                color="0.1",
                path_effects=[withStroke(linewidth=1.8, foreground="white")],
                zorder=6,
            )
        else:
            axes.annotate(
                text,
                (centre, value),
                textcoords="offset points",
                xytext=(0, 4),
                ha="center",
                va="bottom",
                fontsize=9.5,
                color="0.1",
            )


def _selection_transfers(
    path: Path,
    key: tuple[int, int],
    points: tuple[_GeometryPoint, ...],
    *,
    share: bool,
) -> Path | None:
    """One geometry's lane cost, grouped by graph-pair selection.

    The same question the waste figure asks, on the other resource: a
    selection that recomputes less has to keep more, and keeping more is
    traffic. Colour is the selection level, and within it fetch is solid and
    evict is faded, the same convention the by-geometry transfer figures
    use.

    Both variants mirror the two directions about zero, the same way the
    by-geometry lane figures do: fetch and evict are separate lanes running
    at the same time, so a stack of them is not a quantity, and mirroring
    spends one bar's width instead of two.
    """

    levels = sorted(
        {
            outcome.recompute_groups
            for item in points
            for outcome in item.graph_pair_selections
            if outcome.fetched_bytes or outcome.evicted_bytes
        }
    )
    if not levels:
        return None
    width = 0.86
    shades = matplotlib.colormaps["viridis"]
    occupied: dict[float, set[int]] = {}
    for item in points:
        for outcome in item.graph_pair_selections:
            if outcome.recompute_groups in levels and outcome.makespan_seconds:
                occupied.setdefault(item.budget_gib, set()).add(
                    levels.index(outcome.recompute_groups)
                )
    centres, ticks, extent = _packed_groups(occupied, _GROUP_GAP)

    figure = Figure(figsize=(_packed_width(extent, 0.42, 30.0), 7.4), dpi=150)
    axes = figure.subplots()
    above: list[float] = []
    below: list[float] = []
    totals: list[tuple[float, float]] = []
    makespans: list[tuple[float, float, float]] = []
    for item in points:
        for outcome in item.graph_pair_selections:
            if outcome.recompute_groups not in levels:
                continue
            makespan = outcome.makespan_seconds
            if makespan is None or makespan <= 0.0:
                continue
            offset = levels.index(outcome.recompute_groups)
            colour = shades(offset / max(len(levels) - 1, 1))
            centre = centres[(item.budget_gib, offset)]
            summary = item.summary
            if share:
                fetch = (
                    outcome.fetched_bytes
                    / summary.fetch_bandwidth_bytes_per_second
                    / makespan
                )
                evict = (
                    outcome.evicted_bytes
                    / summary.evict_bandwidth_bytes_per_second
                    / makespan
                )
            else:
                fetch = outcome.fetched_bytes / _GIB
                evict = outcome.evicted_bytes / _GIB
            above.append(fetch)
            below.append(evict)
            for value, alpha in ((fetch, 1.0), (-evict, 0.45)):
                axes.bar(centre, value, width=width, color=colour, alpha=alpha)
                totals.append((centre, value))
            makespans.append((centre, fetch, makespan))

    if not above:
        return None
    microbatch, accumulation = key
    axes.set_title(
        (
            "Lane Utilization by Graph-Pair Selection"
            if share
            else "Transfer Traffic by Graph-Pair Selection"
        )
        + f": {microbatch} x {accumulation}"
    )
    axes.set_xlabel("Execution Budget (GiB)")
    axes.set_ylabel("Share of Lane-Seconds" if share else "GiB per Step", labelpad=26)
    drawn_budgets = sorted(ticks)
    axes.set_xticks([ticks[budget] for budget in drawn_budgets])
    axes.set_xticklabels([f"{budget:g}" for budget in drawn_budgets])
    axes.set_xlim(-_GROUP_GAP / 2, extent + _GROUP_GAP / 2)
    if share:
        # Ticks first: a fixed locator carrying values past the limits pulls
        # the view out to reach them, so the limit has to be set last.
        axes.set_yticks([value / 10.0 for value in range(-10, 11)])
        axes.yaxis.set_major_formatter(lambda item, _pos: f"{abs(item) * 100:g}%")
        axes.set_ylim(-max(below) * 1.20, min(1.26, max(above) + 0.19))
    else:
        axes.set_ylim(-max(below) * 1.20, max(above) * 1.30)
        axes.yaxis.set_major_formatter(lambda item, _pos: f"{abs(item):g}")
    # What the lane cost bought: the step this selection's plan reached,
    # over its fetch bar, in the one colour nothing else on the figure uses.
    _annotate_makespans(axes, makespans, width)
    axes.axhline(0.0, color="0.25", linewidth=1.4, zorder=3)
    _label_halves(axes)
    axes.grid(True, axis="y", alpha=0.3, which="major")
    axes.set_axisbelow(True)
    _annotate_bars(axes, totals, width, share=share)

    groups = max(
        (
            outcome.group_count
            for item in points
            for outcome in item.graph_pair_selections
        ),
        default=0,
    )
    names = _recompute_labels(levels, groups)
    handles: list[Patch] = [
        Patch(
            facecolor=shades(index / max(len(levels) - 1, 1)),
            label=names[level],
        )
        for index, level in enumerate(levels)
    ]
    handles += [
        Patch(facecolor="0.35", alpha=1.0, label="Fetch (Above)"),
        Patch(facecolor="0.35", alpha=0.45, label="Evict (Below)"),
        Patch(facecolor="none", edgecolor="none", label="Makespan"),
    ]
    legend = axes.legend(
        handles=handles,
        fontsize="x-small",
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        borderaxespad=0.0,
    )
    for entry in legend.get_texts():
        if entry.get_text() == "Makespan":
            entry.set_color("tab:red")
    figure.tight_layout()
    figure.savefig(path)
    return path


def _geometry_floor_ratio(
    path: Path,
    series: _Series,
    colours: dict[tuple[int, int], tuple[float, float, float, float]],
    winners: dict[float, tuple[int, int]],
) -> Path:
    """How close each geometry comes to its own compute floor.

    The floor charges every graph-pair group its cheapest option and no
    waiting, so it depends on the geometry and not on the budget. The legend
    carries it in seconds, because reaching 80% of a 14 s floor and 80% of an
    18 s floor are not the same achievement.
    """

    figure = Figure(figsize=(7.2, 4.4), dpi=150)
    axes = figure.subplots()
    for key, points in series:
        floor = min(item.summary.unconstrained_step_seconds for item in points)
        axes.plot(
            [item.budget_gib for item in points],
            [
                item.summary.unconstrained_step_seconds / item.step_seconds
                for item in points
            ],
            marker="o",
            markersize=4,
            color=colours[key],
            label=f"{_label(key)} -- Floor {floor:.2f} s",
        )
    _circle_winners(
        axes,
        series,
        winners,
        lambda item: item.summary.unconstrained_step_seconds / item.step_seconds,
    )
    _budget_ticks(axes, [item.budget_gib for _key, points in series for item in points])
    axes.set_title("Simulated Step Against Its Unconstrained Compute Floor")
    axes.set_xlabel("Execution Budget (GiB)")
    axes.set_ylabel("Floor / Simulated Step")
    axes.grid(True, alpha=0.3)
    axes.set_ylim(0.0, 1.05)
    axes.yaxis.set_major_formatter(lambda value, _pos: f"{value:.0%}")
    axes.legend(fontsize="small", ncols=2)
    figure.tight_layout()
    figure.savefig(path)
    return path


def _write_rows(
    path: Path, header: Sequence[str], rows: Sequence[Sequence[object]]
) -> Path:
    """Write one tidy table beside the figures it stands behind."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def _raw_data(
    root: Path, report: StepSearchReport, series: _Series
) -> tuple[Path, ...]:
    """Everything the figures were drawn from, so they can be drawn again.

    `search.json` is the report itself and is lossless: it is the same value
    `plot_step_search` was handed, so any figure here can be rebuilt from it
    exactly, in another style or another tool.

    The two CSVs are the tidy view of it. Two rather than one per figure: all
    but the recomputation ladder are projections of the same per-point row,
    and writing that row twenty times under different names would be twenty
    copies to disagree with each other. Every point is a row, including the
    ones that never planned, because a gap in a line is data too.
    """

    target = root / "raw_data"
    target.mkdir(parents=True, exist_ok=True)
    (target / "search.json").write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True)
    )
    points = []
    for point in report.points:
        summary = point.summary
        step = point.makespan_seconds
        points.append(
            [
                point.sequences_per_microbatch,
                point.accumulation_count,
                point.execution_budget_bytes / _GIB,
                point.spill_budget_bytes / _GIB,
                point.status,
                point.error or "",
                step,
                None if step is None else report.tokens_per_step / step,
                None if summary is None else summary.unconstrained_step_seconds,
                None if summary is None else summary.recomputation_overhead_seconds,
                None if summary is None else summary.idle_seconds,
                None if summary is None else summary.terminal_writeback_seconds,
                None
                if summary is None
                else summary.recomputation_overhead_seconds + summary.idle_seconds,
                None
                if summary is None or step is None
                else summary.unconstrained_step_seconds / step,
                None if summary is None else summary.recomputing_group_count,
                None if summary is None else summary.task_alternative_group_count,
                None if summary is None else summary.transfer_bytes_fetched,
                None if summary is None else summary.transfer_bytes_evicted,
                None if summary is None else summary.fetch_bandwidth_bytes_per_second,
                None if summary is None else summary.evict_bandwidth_bytes_per_second,
            ]
        )
    ladder = [
        [
            key[0],
            key[1],
            item.budget_gib,
            outcome.selection_id,
            outcome.recompute_groups,
            outcome.group_count,
            outcome.makespan_seconds,
            outcome.selected_compute_seconds,
            outcome.unconstrained_seconds,
            outcome.recomputation_overhead_seconds,
            outcome.waiting_seconds,
            outcome.valid_candidate_count,
            outcome.candidate_count,
            outcome.fetched_bytes,
            outcome.evicted_bytes,
        ]
        for key, item_points in series
        for item in item_points
        for outcome in item.graph_pair_selections
    ]
    written: tuple[Path, ...] = (
        _write_rows(
            target / "points.csv",
            (
                "sequences_per_microbatch",
                "accumulation_count",
                "execution_budget_gib",
                "spill_budget_gib",
                "status",
                "error",
                "simulated_step_seconds",
                "simulated_tokens_per_second",
                "unconstrained_seconds",
                "recomputation_overhead_seconds",
                "idle_seconds",
                "terminal_writeback_seconds",
                "wasted_seconds",
                "floor_over_step",
                "recomputing_group_count",
                "task_alternative_group_count",
                "transfer_bytes_fetched",
                "transfer_bytes_evicted",
                "fetch_bandwidth_bytes_per_second",
                "evict_bandwidth_bytes_per_second",
            ),
            points,
        ),
    )
    if ladder:
        written += (
            _write_rows(
                target / "graph_pair_selections.csv",
                (
                    "sequences_per_microbatch",
                    "accumulation_count",
                    "execution_budget_gib",
                    "selection_id",
                    "recompute_groups",
                    "group_count",
                    "makespan_seconds",
                    "selected_compute_seconds",
                    "unconstrained_seconds",
                    "recomputation_overhead_seconds",
                    "waiting_seconds",
                    "valid_candidate_count",
                    "candidate_count",
                    "fetched_bytes",
                    "evicted_bytes",
                ),
                ladder,
            ),
        )
    return written


def plot_step_search(
    report: StepSearchReport,
    directory: str | Path,
) -> tuple[Path, ...]:
    """Write the six figures and return their paths.

    Budgets without a winning geometry are omitted from every line. The
    report must hold a single spill budget; the execution budget is the
    x axis throughout.
    """

    winners = _winners(report)
    # The sequence length leads, so repeating a search at another length
    # writes beside the first rather than over it.
    root = Path(directory)
    # One directory per question the figures answer, under `sim` because
    # every one of them reads a plan rather than a run. `real` is its
    # counterpart, written by the run figures.
    target = root / "sim"
    throughput = target / "throughput"
    overheads = target / "overheads"
    transfers = target / "transfers"
    unconstrained = target / "vs_unconstrained"
    by_selection = overheads / "by_graph_pair_selection"
    lanes_by_selection = transfers / "by_graph_pair_selection"
    for directory_path in (
        target,
        throughput,
        overheads,
        transfers,
        unconstrained,
        by_selection,
        lanes_by_selection,
    ):
        directory_path.mkdir(parents=True, exist_ok=True)
    budgets = [point.execution_budget_bytes / _GIB for point in winners]
    steps = []
    summaries = []
    for point in winners:
        assert point.makespan_seconds is not None
        assert point.summary is not None
        steps.append(point.makespan_seconds)
        summaries.append(point.summary)
    table_path = target / "geometry_table.png"
    table_figure = Figure(figsize=(6.4, 0.6 + 0.4 * len(winners)), dpi=150)
    table_axes = table_figure.subplots()
    table_axes.axis("off")
    table = table_axes.table(
        cellText=[
            [
                f"{point.execution_budget_bytes / _GIB:.2f} GiB",
                f"{point.sequences_per_microbatch}",
                f"{point.accumulation_count}",
                f"{summary.recomputing_group_count}"
                f" / {summary.task_alternative_group_count}",
            ]
            for point, summary in zip(winners, summaries, strict=True)
        ],
        colLabels=[
            "Execution Budget",
            "Sequences / Microbatch",
            "Accumulation",
            "Groups Recomputing",
        ],
        loc="center",
        cellLoc="center",
    )
    table.scale(1.0, 1.4)
    table_axes.set_title("Chosen Geometry by Execution Budget")
    table_figure.tight_layout()
    table_figure.savefig(table_path)

    written: tuple[Path, ...] = (
        table_path,
        _figure(
            throughput / "winners.png",
            "Throughput",
            "Tokens per Second",
            budgets,
            {"Simulated": [report.tokens_per_step / value for value in steps]},
        ),
        _figure(
            throughput / "winners_step_time.png",
            "Simulated Step Time",
            "Seconds",
            budgets,
            {"Simulated": steps},
        ),
        _figure(
            overheads / "winners.png",
            "Recomputation and Stalls",
            "Seconds",
            budgets,
            {
                "Extra Recomputation": [
                    item.recomputation_overhead_seconds for item in summaries
                ],
                "Stalled Between Tasks": [item.idle_seconds for item in summaries],
                "Recomputation and Stalls (Sum)": [
                    item.recomputation_overhead_seconds + item.idle_seconds
                    for item in summaries
                ],
            },
        ),
        _figure(
            overheads / "winners_shares.png",
            "Recomputation and Stalls, Share of the Step",
            "Share of Simulated Step",
            budgets,
            {
                "Extra Recomputation": [
                    item.recomputation_overhead_seconds / step
                    for item, step in zip(summaries, steps, strict=True)
                ],
                "Stalled Between Tasks": [
                    item.idle_seconds / step
                    for item, step in zip(summaries, steps, strict=True)
                ],
                "Recomputation and Stalls (Sum)": [
                    (item.recomputation_overhead_seconds + item.idle_seconds) / step
                    for item, step in zip(summaries, steps, strict=True)
                ],
            },
            percent=True,
        ),
        _figure(
            transfers / "bytes.png",
            "Transfer Traffic per Step",
            "GiB per Step",
            budgets,
            {
                "Fetched": [item.transfer_bytes_fetched / _GIB for item in summaries],
                "Evicted": [item.transfer_bytes_evicted / _GIB for item in summaries],
            },
        ),
        _figure(
            transfers / "lane_utilization.png",
            "Simulated Lane Utilization",
            "Share of Lane-Seconds",
            budgets,
            {
                "Fetch Lane": [
                    item.transfer_bytes_fetched
                    / item.fetch_bandwidth_bytes_per_second
                    / step
                    for item, step in zip(summaries, steps, strict=True)
                ],
                "Evict Lane": [
                    item.transfer_bytes_evicted
                    / item.evict_bandwidth_bytes_per_second
                    / step
                    for item, step in zip(summaries, steps, strict=True)
                ],
            },
            percent=True,
        ),
    )

    # The second family: every geometry, not only the winner.
    series = _geometry_series(report)
    if series:
        colours = _geometry_colours(series)
        best_geometry = _winning_geometry(series)
        written += (
            _geometry_step_time(
                throughput / "by_geometry.png",
                report,
                series,
                colours,
                best_geometry,
            ),
            _geometry_waste_bars(
                overheads / "by_geometry.png",
                series,
                colours,
                best_geometry,
                share=False,
            ),
            _geometry_waste_bars(
                overheads / "by_geometry_shares.png",
                series,
                colours,
                best_geometry,
                share=True,
            ),
            _geometry_floor_ratio(
                unconstrained / "by_geometry.png",
                series,
                colours,
                best_geometry,
            ),
        )
        for share, name in (
            (True, "by_geometry.png"),
            (False, "by_geometry_bytes.png"),
        ):
            lanes = _transfer_bars(
                transfers / name, series, colours, best_geometry, share=share
            )
            if lanes is not None:
                written += (lanes,)
        written += _raw_data(root, report, series)
        for key, points in series:
            name = f"{key[0]}x{key[1]}"
            for share, suffix in ((False, ""), (True, "_shares")):
                figure = _selection_waste(
                    by_selection / f"{name}{suffix}.png", key, points, share=share
                )
                if figure is not None:
                    written += (figure,)
                lanes = _selection_transfers(
                    lanes_by_selection / f"{name}{suffix}.png",
                    key,
                    points,
                    share=share,
                )
                if lanes is not None:
                    written += (lanes,)
    return written
