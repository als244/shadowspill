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

from pathlib import Path

from shadowspill.search import StepSearchReport

from .frontiers import FrontierLine, MeasuredPoints, plot_bandwidth_frontier
from .layout import FigureTree
from .series import geometry_colours, geometry_series, winning_geometry, winning_points
from .step_time import geometry_floor_ratio, geometry_step_time, ordering_step_time
from .tables import raw_data
from .transfers import selection_transfers, transfer_bars
from .waste import geometry_waste_bars, selection_waste
from .winners import winner_figures


def plot_step_search(
    report: StepSearchReport,
    directory: str | Path,
) -> tuple[Path, ...]:
    """Write the figures and the tables behind them, and return their paths.

    Budgets without a winning geometry are omitted from every line. The
    report must hold a single spill budget; the execution budget is the
    x axis throughout.
    """

    winners = winning_points(report)
    tree = FigureTree.under(directory)
    written = winner_figures(report, winners, tree)
    # The second family: every geometry, not only the winner.
    series = geometry_series(report)
    if series:
        colours = geometry_colours(series)
        best_geometry = winning_geometry(series)
        written += (
            geometry_step_time(
                tree.throughput / "by_geometry.png",
                report,
                series,
                colours,
                best_geometry,
            ),
            *(
                geometry_waste_bars(
                    tree.overheads / name,
                    series,
                    colours,
                    best_geometry,
                    share=share,
                    include_compute=include_compute,
                )
                # The waste alone is the comparison between geometries; the
                # whole step is the context it sits in, which is a second
                # figure rather than a second axis.
                for name, share, include_compute in (
                    ("by_geometry.png", False, False),
                    ("by_geometry_shares.png", True, False),
                    ("by_geometry_with_compute.png", False, True),
                    ("by_geometry_with_compute_shares.png", True, True),
                )
            ),
            geometry_floor_ratio(
                tree.unconstrained / "by_geometry.png",
                series,
                colours,
                best_geometry,
            ),
        )
        for share, name in (
            (True, "by_geometry.png"),
            (False, "by_geometry_bytes.png"),
        ):
            lanes = transfer_bars(
                tree.transfers / name, series, colours, best_geometry, share=share
            )
            if lanes is not None:
                written += (lanes,)
        written += tuple(
            ordering_step_time(tree.orderings / f"{key[0]}x{key[1]}.png", report, key)
            for key, _points in series
        )
        written += raw_data(tree.root, report, series)
        for key, points in series:
            name = f"{key[0]}x{key[1]}"
            for share, suffix in ((False, ""), (True, "_shares")):
                figure = selection_waste(
                    tree.by_selection / f"{name}{suffix}.png", key, points, share=share
                )
                if figure is not None:
                    written += (figure,)
                lanes = selection_transfers(
                    tree.lanes_by_selection / f"{name}{suffix}.png",
                    key,
                    points,
                    share=share,
                )
                if lanes is not None:
                    written += (lanes,)
    return written


__all__ = [
    "FrontierLine",
    "MeasuredPoints",
    "plot_bandwidth_frontier",
    "plot_step_search",
]
