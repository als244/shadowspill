"""Step time and throughput: by geometry, by ordering, against the floor."""

from __future__ import annotations

from pathlib import Path

from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator, NullFormatter, NullLocator

from shadowspill.search import StepSearchReport

from .axes import budget_ticks, circle_winners, log_scale
from .series import GeometryPoint, Series, geometry_label, ordering_series

#: How far above the fastest step the detail panel reaches. Wide enough to
#: hold every geometry worth choosing, narrow enough to separate them.
DETAIL_SPAN = 1.25


def geometry_step_time(
    path: Path,
    report: StepSearchReport,
    series: Series,
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

    def rate(item: GeometryPoint) -> float:
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
                label=geometry_label(key) if axes is overview else None,
            )
        circle_winners(axes, series, best_geometry, rate)

    rates = [rate(item) for _key, points in series for item in points]
    overview.set_title("Simulated Throughput by Geometry")
    overview.set_ylabel("Tokens per Second")
    logarithmic = log_scale(overview, rates, floor=None)
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
    detail.set_ylim(fastest / DETAIL_SPAN, fastest * 1.02)
    budget_ticks(
        detail, [item.budget_gib for _key, points in series for item in points]
    )
    detail.set_xlabel("Execution Budget (GiB)")
    detail.set_ylabel("Tokens per Second (Detail)")
    detail.yaxis.set_major_locator(MaxNLocator(nbins=8, steps=[1, 2, 2.5, 5, 10]))
    detail.grid(True, alpha=0.3)
    detail.set_title(
        f"Within {DETAIL_SPAN:g}x of the Best Throughput", fontsize="small"
    )
    figure.tight_layout()
    figure.savefig(path)
    return path


def ordering_step_time(
    path: Path, report: StepSearchReport, key: tuple[int, int]
) -> Path:
    """One geometry's step time under every ordering the search tried.

    The geometry figures above show each geometry at its best ordering; this
    is the ladder behind one of those lines, so a reader can see how much
    the walk itself was worth at each budget and which walk it was.
    """
    figure = Figure(figsize=(7.6, 4.4), dpi=150)
    axes = figure.subplots()
    series = ordering_series(report, key)
    for label, points in series:
        axes.plot(
            [budget for budget, _step in points],
            [step for _budget, step in points],
            marker="o",
            markersize=4,
            label=label,
        )
    best: dict[float, tuple[float, str]] = {}
    for label, points in series:
        for budget, step in points:
            if budget not in best or step < best[budget][0]:
                best[budget] = (step, label)
    if best:
        axes.scatter(
            list(best),
            [step for step, geometry_label in best.values()],
            s=110,
            facecolors="none",
            edgecolors="black",
            linewidths=1.2,
            zorder=5,
            label="Best at This Budget",
        )
    axes.set_title(f"Simulated Step Time by Ordering, {geometry_label(key)}")
    axes.set_xlabel("Execution Budget (GiB)")
    axes.set_ylabel("Seconds")
    axes.grid(True, alpha=0.3)
    axes.legend(title="depth x breadth (r: reversed, p: paired loss)", fontsize=7)
    figure.tight_layout()
    figure.savefig(path)
    return path


def geometry_floor_ratio(
    path: Path,
    series: Series,
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
            label=f"{geometry_label(key)} -- Floor {floor:.2f} s",
        )
    circle_winners(
        axes,
        series,
        winners,
        lambda item: item.summary.unconstrained_step_seconds / item.step_seconds,
    )
    budget_ticks(axes, [item.budget_gib for _key, points in series for item in points])
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
