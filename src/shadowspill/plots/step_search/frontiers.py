"""Throughput against execution budget, one line per transfer calibration.

A search answers what a budget buys under one interconnect. Running it again
under another, with nothing else changed, says what the interconnect is worth:
the same programs, the same geometries and the same budgets, planned as though
the lanes were faster. The lines here are those answers overlaid, so the
distance between them is the transfer calibration and nothing else.

Nothing here is measured. A line is what the simulator predicts for the plan
the search chose, which is the quantity the winners figure plots for a single
calibration. Measured points may be drawn on the line they were run under,
in its colour and with a distinct marker, so the two cannot be confused.

The axis begins at the first budget a plan exists for: below it no residency
fits at any bandwidth, which is a fact about the program rather than one of
the calibrations this figure compares.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from matplotlib.figure import Figure

from shadowspill.plots._axis import budget_label

#: One per line, so lines that converge are still told apart.
_MARKERS = ("o", "s", "^", "D", "v", "P", "*", "X")


@dataclass(frozen=True, slots=True)
class FrontierLine:
    """One calibration's answer at every budget that was searched.

    ``tokens_per_second`` holds ``None`` where the search found no plan, so a
    line keeps its place on the budget axis rather than sliding left.
    """

    label: str
    fetch_bytes_per_second: int
    evict_bytes_per_second: int
    tokens_per_second: Mapping[float, float | None]

    def at(self, budget_gib: float) -> float | None:
        return self.tokens_per_second.get(budget_gib)


@dataclass(frozen=True, slots=True)
class MeasuredPoints:
    """What a run measured, drawn on the line it was run under.

    ``line`` names that line, so the markers take its colour and the eye
    pairs them without reading the legend. A measured set whose line was not
    searched is still drawn, in its own colour, rather than dropped.
    """

    label: str
    tokens_per_second: Mapping[float, float]
    line: str | None = None


def plot_bandwidth_frontier(
    path: str | Path,
    budgets_gib: Sequence[float],
    lines: Sequence[FrontierLine],
    *,
    title: str,
    subtitle: str | None = None,
    measured: Sequence[MeasuredPoints] = (),
    unconstrained_tokens_per_second: float | None = None,
    faded_lines: bool = False,
) -> Path:
    """Draw one line per calibration and return where it was written.

    The axis begins where a plan first exists. Below that no residency fits,
    whatever the lanes can carry, and a budget no line can answer says
    nothing a reader of this figure came for.

    ``unconstrained_tokens_per_second`` draws the ceiling: every
    task-alternative group charged its cheapest option and nothing waiting,
    which is the step a machine with no transfers at all would run. It is
    annotated on the line rather than listed in the legend, because it is a
    property of the program and not one of the calibrations being compared.

    ``faded_lines`` puts the simulated lines into the background, dashed and
    faint, and joins the measured points instead. It is for the figure whose
    subject is what a machine did, where the predictions are the context
    rather than the claim.
    """

    feasible = [
        budget
        for budget in sorted(budgets_gib)
        if any(line.at(budget) is not None for line in lines)
    ]
    if not feasible:
        feasible = sorted(budgets_gib)
    # A legend of more than about seven entries no longer fits in whatever
    # corner the data leaves empty, so it goes under the axes instead and the
    # figure grows to hold it. Below that it sits inside, where it costs no
    # width and the eye stays on one rectangle.
    crowded = len(lines) + len(measured) > 7
    figure = Figure(
        figsize=(7.6, 5.4 if crowded else 4.6), dpi=150, layout="constrained"
    )
    axes = figure.subplots()
    colours: dict[str, str] = {}
    for index, line in enumerate(lines):
        values = [line.at(budget) for budget in feasible]
        drawn = axes.plot(
            feasible,
            [float("nan") if value is None else value for value in values],
            # A distinct marker per line, because two calibrations fast
            # enough to converge draw the same curve and the upper one would
            # otherwise erase the other.
            marker=_MARKERS[index % len(_MARKERS)],
            markersize=4.5 if faded_lines else 5.5,
            linewidth=1.3 if faded_lines else 1.8,
            linestyle=(0, (5, 3)) if faded_lines else "-",
            alpha=0.4 if faded_lines else 1.0,
            zorder=2,
            label=line.label,
        )
        colours[line.label] = drawn[0].get_color()
    drawn_budgets = set(feasible)
    for series in measured:
        points = sorted(
            (budget, value)
            for budget, value in series.tokens_per_second.items()
            if budget >= feasible[0]
        )
        if not points:
            continue
        drawn_budgets.update(budget for budget, _ in points)
        axes.plot(
            [budget for budget, _ in points],
            [value for _, value in points],
            # Joined, because two measurements of one machine at two budgets
            # are a curve the audience is being shown, not two facts.
            linestyle="-",
            linewidth=2.2,
            marker="X",
            markersize=9,
            markeredgecolor="white",
            markeredgewidth=1.0,
            color=colours.get(series.line or "", "0.2"),
            zorder=5,
            label=series.label,
        )
    if unconstrained_tokens_per_second is not None:
        axes.axhline(
            unconstrained_tokens_per_second,
            linestyle=(0, (6, 4)),
            linewidth=1.1,
            color="0.45",
            zorder=1,
        )
        axes.annotate(
            "compute ceiling (full save, no transfers)",
            xy=(feasible[0], unconstrained_tokens_per_second),
            xytext=(2, 5),
            textcoords="offset points",
            fontsize=8.5,
            color="0.35",
        )
    # Set here rather than through `budget_ticks`, which leans its labels once
    # a ladder passes eight: these are two digits and sit a budget apart, so
    # they stay upright and the axis stays quiet.
    ticks = sorted(drawn_budgets)
    axes.set_xticks(ticks)
    axes.set_xticklabels([budget_label(value) for value in ticks])
    axes.set_xlabel("Execution budget (GiB)", labelpad=6)
    axes.set_ylabel("Throughput (tokens/s)", labelpad=6)
    axes.set_title(title, fontsize=11.5, loc="left", pad=22 if subtitle else 8)
    if subtitle:
        axes.text(
            0.0,
            1.015,
            subtitle,
            transform=axes.transAxes,
            ha="left",
            va="bottom",
            fontsize=9.5,
            color="0.4",
        )
    axes.set_ylim(bottom=0.0)
    if unconstrained_tokens_per_second is not None:
        axes.set_ylim(top=unconstrained_tokens_per_second * 1.12)
    axes.margins(x=0.04)
    axes.yaxis.set_major_formatter(lambda value, _pos: f"{value:,.0f}")
    axes.grid(True, axis="y", alpha=0.25, linewidth=0.7)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    if crowded:
        axes.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.14),
            ncol=3,
            frameon=False,
            fontsize=9,
            handlelength=2.0,
            columnspacing=1.6,
        )
    else:
        # Inside the axes, wherever the data is not: a figure whose legend
        # takes a third of its width has that much less room for the lines it
        # is about.
        axes.legend(
            loc="best",
            frameon=True,
            framealpha=0.92,
            edgecolor="0.85",
            fontsize=9,
            handlelength=2.0,
            borderpad=0.6,
            labelspacing=0.4,
        )
    written = Path(path)
    figure.savefig(written)
    return written
