"""Figures over a geometry search's winners, one per budget on the x axis.

Six figures, written as PNG files into a directory: throughput and raw
step time of each budget's winning geometry; the winners' recomputation
and waiting overheads, raw and as shares of the simulated step; and the
winners' fetch/evict traffic, raw and as simulated lane utilization
(bytes over assumed bandwidth over step time). Every value is plan-side,
read from each winning point's :class:`PlanSummary`.
"""

from __future__ import annotations

from pathlib import Path

from matplotlib.figure import Figure

from shadowspill.pytorch.step_search import StepSearchPoint, StepSearchReport

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
    axes.set_title(title)
    axes.set_xlabel("Execution budget (GiB)")
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
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
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
                f"{summary.recompute_selection_count} / {summary.selection_count}",
            ]
            for point, summary in zip(winners, summaries, strict=True)
        ],
        colLabels=[
            "Execution budget",
            "Sequences / microbatch",
            "Accumulation",
            "Recompute selections",
        ],
        loc="center",
        cellLoc="center",
    )
    table.scale(1.0, 1.4)
    table_axes.set_title("Chosen geometry by execution budget")
    table_figure.tight_layout()
    table_figure.savefig(table_path)

    written = (
        table_path,
        _figure(
            target / "throughput.png",
            "Throughput",
            "Tokens per second",
            budgets,
            {"Simulated": [report.tokens_per_step / value for value in steps]},
        ),
        _figure(
            target / "step_time.png",
            "Simulated step time",
            "Seconds",
            budgets,
            {"Simulated": steps},
        ),
        _figure(
            target / "overheads.png",
            "Recomputation and waiting",
            "Seconds",
            budgets,
            {
                "Extra recomputation": [
                    item.recomputation_overhead_seconds for item in summaries
                ],
                "Waiting between tasks": [item.idle_seconds for item in summaries],
                "Wasted compute (sum)": [
                    item.recomputation_overhead_seconds + item.idle_seconds
                    for item in summaries
                ],
            },
        ),
        _figure(
            target / "overhead_shares.png",
            "Recomputation and waiting, share of the step",
            "Share of simulated step",
            budgets,
            {
                "Extra recomputation": [
                    item.recomputation_overhead_seconds / step
                    for item, step in zip(summaries, steps, strict=True)
                ],
                "Waiting between tasks": [
                    item.idle_seconds / step
                    for item, step in zip(summaries, steps, strict=True)
                ],
                "Wasted compute (sum)": [
                    (item.recomputation_overhead_seconds + item.idle_seconds) / step
                    for item, step in zip(summaries, steps, strict=True)
                ],
            },
            percent=True,
        ),
        _figure(
            target / "transfer_bytes.png",
            "Transfer traffic per step",
            "GiB per step",
            budgets,
            {
                "Fetched": [item.transfer_bytes_fetched / _GIB for item in summaries],
                "Evicted": [item.transfer_bytes_evicted / _GIB for item in summaries],
            },
        ),
        _figure(
            target / "lane_utilization.png",
            "Simulated lane utilization",
            "Share of lane-seconds",
            budgets,
            {
                "Fetch lane": [
                    item.transfer_bytes_fetched
                    / item.fetch_bandwidth_bytes_per_second
                    / step
                    for item, step in zip(summaries, steps, strict=True)
                ],
                "Evict lane": [
                    item.transfer_bytes_evicted
                    / item.evict_bandwidth_bytes_per_second
                    / step
                    for item, step in zip(summaries, steps, strict=True)
                ],
            },
            percent=True,
        ),
    )
    return written
