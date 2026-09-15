"""Fetch and evict traffic: by geometry, and by graph-pair selection."""

from __future__ import annotations

from pathlib import Path

import matplotlib
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from shadowspill.plots._axis import budget_label

from .axes import (
    GROUP_GAP,
    annotate_bars,
    annotate_makespans,
    label_halves,
    packed_groups,
    packed_width,
    recompute_labels,
)
from .series import GIB, GeometryPoint, Series, geometry_label


def transfer_bars(
    path: Path,
    series: Series,
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
    centres, ticks, extent = packed_groups(occupied, GROUP_GAP)

    # Fourteen bars to a group against the waste figure's seven, so the room
    # comes from height as well as width rather than a letterbox.
    figure = Figure(figsize=(packed_width(extent, 0.42, 30.0), 7.4), dpi=150)
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
                fetch = summary.transfer_bytes_fetched / GIB
                evict = summary.transfer_bytes_evicted / GIB
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
    axes.set_xticklabels([budget_label(budget) for budget in drawn_budgets])
    # Half a gap at each end, so the outer budgets are spaced like the
    # inner ones rather than pinned to the frame.
    axes.set_xlim(-GROUP_GAP / 2, extent + GROUP_GAP / 2)
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
    annotate_makespans(axes, makespans, width)
    axes.axhline(0.0, color="0.25", linewidth=1.4, zorder=3)
    label_halves(axes)
    axes.grid(True, axis="y", alpha=0.3, which="major")
    axes.set_axisbelow(True)
    annotate_bars(axes, totals, width, share=share)

    handles: list[Patch] = [
        Patch(facecolor=colours[key], label=geometry_label(key))
        for key, _points in ordered
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


def selection_transfers(
    path: Path,
    key: tuple[int, int],
    points: tuple[GeometryPoint, ...],
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
    centres, ticks, extent = packed_groups(occupied, GROUP_GAP)

    figure = Figure(figsize=(packed_width(extent, 0.42, 30.0), 7.4), dpi=150)
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
                fetch = outcome.fetched_bytes / GIB
                evict = outcome.evicted_bytes / GIB
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
    axes.set_xticklabels([budget_label(budget) for budget in drawn_budgets])
    axes.set_xlim(-GROUP_GAP / 2, extent + GROUP_GAP / 2)
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
    annotate_makespans(axes, makespans, width)
    axes.axhline(0.0, color="0.25", linewidth=1.4, zorder=3)
    label_halves(axes)
    axes.grid(True, axis="y", alpha=0.3, which="major")
    axes.set_axisbelow(True)
    annotate_bars(axes, totals, width, share=share)

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
