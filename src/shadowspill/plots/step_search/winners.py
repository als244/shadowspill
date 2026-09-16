"""The winning geometry at each budget: what was chosen, and where its step went.

One line per measure over the budgets that produced a winner, plus the table
naming the geometry and ordering behind each of those points. These are the
figures a reader starts from; the per-geometry families say what the winner
was chosen against.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from matplotlib.figure import Figure

from shadowspill.search import StepSearchPoint, StepSearchReport

from .axes import line_figure
from .layout import FigureTree
from .series import GIB


def winner_figures(
    report: StepSearchReport,
    winners: Sequence[StepSearchPoint],
    tree: FigureTree,
) -> tuple[Path, ...]:
    """Write the geometry table and the six lines over the winning points."""

    budgets = [point.execution_budget_bytes / GIB for point in winners]
    steps = []
    summaries = []
    for point in winners:
        assert point.makespan_seconds is not None
        assert point.summary is not None
        steps.append(point.makespan_seconds)
        summaries.append(point.summary)
    table_path = tree.sim / "geometry_table.png"
    table_figure = Figure(figsize=(6.4, 0.6 + 0.4 * len(winners)), dpi=150)
    table_axes = table_figure.subplots()
    table_axes.axis("off")
    table = table_axes.table(
        cellText=[
            [
                f"{point.execution_budget_bytes / GIB:.2f} GiB",
                f"{point.sequences_per_microbatch}",
                f"{point.accumulation_count}",
                point.ordering.label,
                f"{summary.recomputing_group_count}"
                f" / {summary.task_alternative_group_count}",
            ]
            for point, summary in zip(winners, summaries, strict=True)
        ],
        colLabels=[
            "Execution Budget",
            "Sequences / Microbatch",
            "Accumulation",
            "Ordering",
            "Groups Recomputing",
        ],
        loc="center",
        cellLoc="center",
    )
    table.scale(1.0, 1.4)
    table_axes.set_title("Chosen Geometry by Execution Budget")
    table_figure.tight_layout()
    table_figure.savefig(table_path)

    return (
        table_path,
        line_figure(
            tree.throughput / "winners.png",
            "Throughput",
            "Tokens per Second",
            budgets,
            {"Simulated": [report.tokens_per_step / value for value in steps]},
        ),
        line_figure(
            tree.throughput / "winners_step_time.png",
            "Simulated Step Time",
            "Seconds",
            budgets,
            {"Simulated": steps},
        ),
        line_figure(
            tree.overheads / "winners.png",
            "Where the Step Goes",
            "Seconds",
            budgets,
            {
                "Effective Compute": [
                    item.unconstrained_step_seconds for item in summaries
                ],
                "Recomputation": [
                    item.recomputation_overhead_seconds for item in summaries
                ],
                "Stalled": [
                    item.idle_seconds + item.terminal_writeback_seconds
                    for item in summaries
                ],
            },
        ),
        line_figure(
            tree.overheads / "winners_shares.png",
            "Where the Step Goes, Share of the Step",
            "Share of Simulated Step",
            budgets,
            {
                "Effective Compute": [
                    item.unconstrained_step_seconds / step
                    for item, step in zip(summaries, steps, strict=True)
                ],
                "Recomputation": [
                    item.recomputation_overhead_seconds / step
                    for item, step in zip(summaries, steps, strict=True)
                ],
                "Stalled": [
                    (item.idle_seconds + item.terminal_writeback_seconds) / step
                    for item, step in zip(summaries, steps, strict=True)
                ],
            },
            percent=True,
        ),
        line_figure(
            tree.transfers / "bytes.png",
            "Transfer Traffic per Step",
            "GiB per Step",
            budgets,
            {
                "Fetched": [item.transfer_bytes_fetched / GIB for item in summaries],
                "Evicted": [item.transfer_bytes_evicted / GIB for item in summaries],
            },
        ),
        line_figure(
            tree.transfers / "lane_utilization.png",
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
