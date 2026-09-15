"""The report and its tidy tables, written beside the figures they stand behind."""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path

from shadowspill.pytorch.step_search import StepSearchReport

from .series import GIB, Series


def write_rows(
    path: Path, header: Sequence[str], rows: Sequence[Sequence[object]]
) -> Path:
    """Write one tidy table beside the figures it stands behind."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def raw_data(root: Path, report: StepSearchReport, series: Series) -> tuple[Path, ...]:
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
                point.ordering.label,
                point.execution_budget_bytes / GIB,
                point.spill_budget_bytes / GIB,
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
            item.ordering_label,
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
        write_rows(
            target / "points.csv",
            (
                "sequences_per_microbatch",
                "accumulation_count",
                "ordering",
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
            write_rows(
                target / "graph_pair_selections.csv",
                (
                    "sequences_per_microbatch",
                    "accumulation_count",
                    "ordering",
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
