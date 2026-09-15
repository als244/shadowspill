"""Axis, scale and annotation primitives the figures share.

The rules live here because they must agree across the family: the same
budget ticks, the same ring around the winner, the same decision about when
a linear axis would hide most of the range, and the same refusal to write a
label that would not fit.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from pathlib import Path

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patheffects import withStroke
from matplotlib.ticker import LogLocator, NullFormatter, NullLocator

from shadowspill.plots._axis import budget_label

from .series import GeometryPoint, Series


def budget_ticks(axes: Axes, budgets: Sequence[float]) -> None:
    """Tick the budgets that were searched, not a round interpolation of them.

    A budget is a value someone chose, so a tick between two of them names a
    point the search never visited. Crowded ladders lean their labels rather
    than dropping them.
    """

    values = sorted(set(budgets))
    axes.set_xticks(values)
    axes.set_xticklabels(
        [budget_label(value) for value in values],
        rotation=45 if len(values) > 8 else 0,
        ha="right" if len(values) > 8 else "center",
    )


def line_figure(
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
    budget_ticks(axes, budgets_gib)
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


def circle_winners(
    axes: Axes,
    series: Series,
    winners: dict[float, tuple[int, int]],
    measure: Callable[[GeometryPoint], float],
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
GROUP_GAP = 0.8

#: Spread past which a linear axis flattens the fast geometries into the floor.
LOG_SPREAD = 4.0


def mark_floor(axes: Axes, value: float) -> None:
    """Draw the best attainable value as a heavier line than the grid.

    Every one of these figures has an ideal its lines approach from one side,
    and reading how close a geometry is to it is the point. A gridline of the
    same weight as the others does not say which one that is.
    """

    axes.axhline(value, color="0.25", linewidth=1.8, zorder=1.5)


def log_scale(axes: Axes, values: list[float], *, floor: float | None) -> bool:
    """Spread the axis logarithmically when a linear one would hide most of it.

    One slow geometry can be a thousand times another, which presses every
    fast line onto the axis floor. Where the ideal is zero the scale is
    symmetric-log, whose linear region below the smallest measured value is
    what lets zero be a tick at all: a plain log axis puts it at negative
    infinity and cannot draw it.
    """

    if not values or any(value <= 0.0 for value in values):
        return False
    if max(values) / min(values) < LOG_SPREAD:
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


#: Room one label needs, in display points: comfortably more than the text's
#: own height, so that two segments of nearly the same size are either both
#: labelled or both not, and the width of the widest number these charts
#: print.
LABEL_HEIGHT = 24.0

LABEL_WIDTH = 32.0


def label_segments(
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
    if step[0] - origin[0] < LABEL_WIDTH:
        return
    for centre, top, bottom, value in segments:
        low, high = axes.transData.transform([(centre, bottom), (centre, top)])
        if high[1] - low[1] < LABEL_HEIGHT:
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


def recompute_labels(levels: Sequence[int], groups: int) -> dict[int, str]:
    """Name each rung of the ladder by the share of groups it recomputes.

    Rounded to an eighth, the ladder's own step, because "50% of Groups"
    reads at a glance where "17 of 36" does not and the exact figure is not
    what a reader is comparing. Rounding falls back to the true share if it
    would give two rungs the same name, since a legend with a repeated key is
    worse than one with an awkward number.
    """

    if not groups:
        return {level: f"{level} Recomputing" for level in levels}
    rounded = {level: round(level / groups * 8) * 12.5 for level in levels}
    if len(set(rounded.values())) != len(levels):
        rounded = {level: float(round(level / groups * 100)) for level in levels}
    return {
        level: f"{share:g}% of Groups Recomputing" for level, share in rounded.items()
    }


def packed_width(extent: float, per_bar: float, cap: float) -> float:
    """How wide a packed chart needs to be, in inches.

    The extent is in bar pitches, so a budget where only one series planned
    costs one pitch rather than a full group's worth of blank.
    """

    return min(2.8 + per_bar * extent, cap)


def packed_groups(
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


def label_halves(axes: Axes) -> None:
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


#: The makespan text over a bar: its size and how far above the bar it sits,
#: both in points.
MAKESPAN_FONT_SIZE = 9.0
MAKESPAN_OFFSET = 21.0


def annotate_makespans(
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
    if room < LABEL_WIDTH * 0.72:
        return
    # What the text actually needs, in the same display units as the bar:
    # a digit is about 0.62 em wide, and an em is the font size in points.
    # Estimated rather than measured because measuring needs a renderer, and
    # the estimate only has to decide which way the text runs.
    widest = max(len(f"{seconds:.1f} s") for _centre, _top, seconds in entries)
    character = MAKESPAN_FONT_SIZE * 0.62
    upright = room < widest * character * axes.figure.dpi / 72.0
    for centre, top, seconds in entries:
        axes.annotate(
            f"{seconds:.1f} s",
            (centre, top),
            textcoords="offset points",
            xytext=(0, MAKESPAN_OFFSET),
            rotation=90 if upright else 0,
            ha="center",
            va="bottom",
            fontsize=MAKESPAN_FONT_SIZE,
            color="tab:red",
        )
    # Upright text is as tall as the label is long, so the axis has to be
    # told: the caller cannot know which way the text ran. Asked for in
    # pixels over the tallest bar and converted back, which is the only way
    # that holds on the log axis these bars are usually drawn on.
    text = widest * character if upright else MAKESPAN_FONT_SIZE * 1.4
    tallest = max(top for _centre, top, _seconds in entries)
    wanted = (
        axes.transData.transform((0.0, tallest))[1]
        + (MAKESPAN_OFFSET + text + 4.0) * axes.figure.dpi / 72.0
    )
    low, high = axes.get_ylim()
    needed = float(axes.transData.inverted().transform((0.0, wanted))[1])
    if needed > high:
        axes.set_ylim(low, needed)


def annotate_bars(
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
    if step[0] - origin[0] < LABEL_WIDTH * 0.72:
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
