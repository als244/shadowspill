"""Recomputation and stall: by geometry, and by graph-pair selection."""

from __future__ import annotations

from pathlib import Path

import matplotlib
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from shadowspill.plots._axis import budget_label

from .axes import (
    GROUP_GAP,
    annotate_makespans,
    label_segments,
    log_scale,
    mark_floor,
    packed_groups,
    packed_width,
    recompute_labels,
)
from .series import GeometryPoint, Series, geometry_label


def _waste_segments(
    axes: matplotlib.axes.Axes,
    ordered: tuple[tuple[tuple[int, int], tuple[GeometryPoint, ...]], ...],
    colours: dict[tuple[int, int], tuple[float, float, float, float]],
    best_geometry: dict[float, tuple[int, int]],
    centres: dict[tuple[float, int], float],
    *,
    share: bool,
    include_compute: bool,
    width: float,
) -> tuple[
    list[float],
    list[tuple[float, float, float, float]],
    list[tuple[float, float]],
    list[tuple[float, float, float]],
]:
    """Draw one stacked bar per geometry and budget, and say what was drawn."""

    drawn: list[float] = []
    labels: list[tuple[float, float, float, float]] = []
    totals: list[tuple[float, float]] = []
    makespans: list[tuple[float, float, float]] = []
    for offset, (key, points) in enumerate(ordered):
        for item in points:
            scale = item.step_seconds if share else 1.0
            # With the compute the three partition the step exactly: the
            # cheapest graphs with no waiting, the compute recomputation
            # adds, and everything the step was not computing -- stalls and
            # the terminal writeback, which the simulator prices inside the
            # same makespan. Without it the bar is the last two alone.
            stalled = (
                item.summary.idle_seconds + item.summary.terminal_writeback_seconds
            )
            parts = [
                (item.summary.recomputation_overhead_seconds, 1.0),
                (stalled, 0.45),
            ]
            if include_compute:
                parts = [
                    (item.summary.unconstrained_step_seconds, 1.0),
                    (item.summary.recomputation_overhead_seconds, 0.62),
                    (stalled, 0.30),
                ]
            centre = centres[(item.budget_gib, offset)]
            bottom = 0.0
            for seconds, alpha in parts:
                height = seconds / scale
                axes.bar(
                    centre,
                    height,
                    width=width,
                    bottom=bottom,
                    color=colours[key],
                    alpha=alpha,
                )
                labels.append((centre, bottom + height, bottom, seconds))
                bottom += height
                drawn.append(bottom)
            if best_geometry.get(item.budget_gib) == key:
                # One outline around the whole bar rather than around each
                # segment, which would draw a line along the boundary between
                # them and read as a third division.
                axes.bar(
                    centre,
                    bottom,
                    width=width,
                    fill=False,
                    edgecolor="black",
                    linewidth=1.6,
                    zorder=4,
                )
            totals.append((centre, bottom))
            if not include_compute:
                # With the compute, the bar's total is the makespan already.
                makespans.append((centre, bottom, item.step_seconds))
    return drawn, labels, totals, makespans


def _waste_axes(
    axes: matplotlib.axes.Axes,
    drawn: list[float],
    ticks: dict[float, float],
    *,
    share: bool,
    include_compute: bool,
) -> None:
    """Title the figure, label the budgets, and give the bars their scale."""

    title = "Where the Step Goes" if include_compute else "What the Step Wastes"
    axes.set_title(f"{title}, Share of the Step" if share else title)
    axes.set_xlabel("Execution Budget (GiB)")
    axes.set_ylabel("Share of Simulated Step" if share else "Seconds")
    drawn_budgets = sorted(ticks)
    axes.set_xticks([ticks[budget] for budget in drawn_budgets])
    axes.set_xticklabels([budget_label(budget) for budget in drawn_budgets])
    # Headroom is what the annotations need and nothing more: the total sits
    # just over the bar, and the makespan asks for its own room below.
    headroom = 1.12
    if share:
        axes.set_ylim(0.0, max(drawn) * min(headroom, 1.22))
        axes.yaxis.set_major_locator(MaxNLocator(nbins=12, steps=[1, 2, 2.5, 5, 10]))
        axes.yaxis.set_major_formatter(lambda item, _pos: f"{item * 100:g}%")
    else:
        # log_scale already labels a 1-2-5 ladder and keeps zero as a tick;
        # overriding its locator here would drop the zero and unclamp the
        # bottom, which is where every bar starts.
        log_scale(axes, drawn, floor=0.0)
        axes.set_ylim(0.0, max(drawn) * headroom)
    mark_floor(axes, 0.0)
    axes.grid(True, axis="y", alpha=0.3, which="major")
    axes.set_axisbelow(True)


def _waste_legend(
    axes: matplotlib.axes.Axes,
    ordered: tuple[tuple[tuple[int, int], tuple[GeometryPoint, ...]], ...],
    colours: dict[tuple[int, int], tuple[float, float, float, float]],
    *,
    include_compute: bool,
) -> None:
    """Name every geometry, then what each segment of a bar is."""

    handles: list[Patch] = [
        Patch(facecolor=colours[key], label=geometry_label(key))
        for key, _points in ordered
    ]
    handles += (
        [
            Patch(facecolor="0.35", alpha=1.0, label="Effective Compute (Lower)"),
            Patch(facecolor="0.35", alpha=0.62, label="Recomputation (Middle)"),
            Patch(facecolor="0.35", alpha=0.30, label="Stalled (Upper)"),
        ]
        if include_compute
        else [
            Patch(facecolor="0.35", alpha=1.0, label="Recomputation (Lower)"),
            Patch(facecolor="0.35", alpha=0.45, label="Stalled (Upper)"),
            Patch(facecolor="none", edgecolor="none", label="Makespan"),
        ]
    )
    handles += [
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


def geometry_waste_bars(
    path: Path,
    series: Series,
    colours: dict[tuple[int, int], tuple[float, float, float, float]],
    best_geometry: dict[float, tuple[int, int]],
    *,
    share: bool,
    include_compute: bool,
) -> Path:
    """Bars over budgets and geometries: what the step wastes, or all of it.

    Without the compute, a bar is the waste alone -- the recomputation the
    plan chose and the stall it could not avoid -- which is the comparison
    between geometries at its own scale, and the makespan is written above
    it because the bar no longer carries it. With the compute the bar is the
    whole step, the three parts partitioning it exactly, and the number above
    the bar is the makespan itself. Every segment carries its time in
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
    centres, ticks, extent = packed_groups(occupied, GROUP_GAP)

    # Wide enough that a segment's label clears its neighbour's, up to a
    # width a reader can still scan. Past that the bars narrow and the
    # labels drop out on their own rather than the file growing without
    # bound.
    figure = Figure(figsize=(packed_width(extent, 0.42, 30.0), 5.2), dpi=150)
    axes = figure.subplots()
    drawn, labels, totals, makespans = _waste_segments(
        axes,
        ordered,
        colours,
        best_geometry,
        centres,
        share=share,
        include_compute=include_compute,
        width=width,
    )
    _waste_axes(axes, drawn, ticks, share=share, include_compute=include_compute)
    label_segments(axes, labels, width)
    # With the compute, the total as a share is 100 % on every bar by
    # construction: the three parts partition the step. The waste's share is
    # the point of its own figure, so that one is annotated either way.
    if not share or not include_compute:
        for centre, total in totals:
            axes.annotate(
                f"{total * 100:.1f}%" if share else f"{total:.1f}",
                (centre, total),
                textcoords="offset points",
                xytext=(0, 4),
                ha="center",
                va="bottom",
                fontsize=10.0,
                color="0.1",
            )
    annotate_makespans(axes, makespans, width)
    _waste_legend(axes, ordered, colours, include_compute=include_compute)
    figure.tight_layout()
    figure.savefig(path)
    return path


def selection_waste(
    path: Path,
    key: tuple[int, int],
    points: tuple[GeometryPoint, ...],
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
    centres, ticks, extent = packed_groups(occupied, GROUP_GAP)

    figure = Figure(figsize=(packed_width(extent, 0.42, 30.0), 5.2), dpi=150)
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
    axes.set_xticklabels([budget_label(budget) for budget in drawn_budgets])
    if share:
        axes.set_ylim(0.0, max(drawn) * 1.30)
        axes.yaxis.set_major_locator(MaxNLocator(nbins=12, steps=[1, 2, 2.5, 5, 10]))
        axes.yaxis.set_major_formatter(lambda item, _pos: f"{item * 100:g}%")
    else:
        log_scale(axes, drawn, floor=0.0)
        axes.set_ylim(0.0, max(drawn) * 1.55)
    mark_floor(axes, 0.0)
    axes.grid(True, axis="y", alpha=0.3, which="major")
    axes.set_axisbelow(True)
    label_segments(axes, labels, width)
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
    annotate_makespans(axes, makespans, width)

    groups = max(
        (
            outcome.group_count
            for item in points
            for outcome in item.graph_pair_selections
        ),
        default=0,
    )
    names = recompute_labels(levels, groups)
    handles: list[Patch] = [
        Patch(
            facecolor=shades(index / max(len(levels) - 1, 1)),
            label=names[level],
        )
        for index, level in enumerate(levels)
    ]
    handles += [
        Patch(facecolor="0.35", alpha=1.0, label="Recomputation (Lower)"),
        Patch(facecolor="0.35", alpha=0.45, label="Waiting (Upper)"),
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
