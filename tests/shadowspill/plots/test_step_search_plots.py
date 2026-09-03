"""The six step-search figures render from a fabricated report."""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType

import pytest

from shadowspill.planner.diagnostics.plan import PlanSummary
from shadowspill.plots import plot_step_search
from shadowspill.pytorch import StepSearchPoint, StepSearchReport


def _summary(step: float) -> PlanSummary:
    return PlanSummary(
        simulated_step_seconds=step,
        unconstrained_step_seconds=step * 0.5,
        recomputation_overhead_seconds=step * 0.2,
        idle_seconds=step * 0.25,
        terminal_writeback_seconds=step * 0.05,
        recompute_selection_count=2,
        selection_count=4,
        transfer_bytes_fetched=int(4e9),
        transfer_bytes_evicted=int(3e9),
        fetch_bandwidth_bytes_per_second=int(20e9),
        evict_bandwidth_bytes_per_second=int(20e9),
        planning_phase_seconds=MappingProxyType({}),
    )


def _point(execution: int, spill: int, step: float) -> StepSearchPoint:
    return StepSearchPoint(
        sequences_per_microbatch=8,
        accumulation_count=4,
        execution_budget_bytes=execution,
        spill_budget_bytes=spill,
        status="succeeded",
        makespan_seconds=step,
        summary=_summary(step),
        error=None,
        search_seconds=0.1,
    )


def test_all_six_figures_render(tmp_path: Path) -> None:
    budgets = ((8 << 30, 64 << 30), (16 << 30, 64 << 30))
    report = StepSearchReport(
        total_sequences_per_step=32,
        sequence_length=1024,
        budgets=budgets,
        geometries=(),
        points=(_point(8 << 30, 64 << 30, 6.0), _point(16 << 30, 64 << 30, 4.0)),
        skipped=(),
    )
    written = plot_step_search(report, tmp_path)
    assert len(written) == 7
    for path in written:
        assert path.exists() and path.stat().st_size > 0


def test_run_throughput_figure_renders(tmp_path: Path) -> None:
    from shadowspill.plots import plot_step_run

    path = plot_step_run(
        [(16 << 30, 4.0, 4.4), (8 << 30, 6.0, 6.9)],
        tmp_path,
        tokens_per_step=32 * 1024,
    )
    assert path.exists() and path.stat().st_size > 0


def test_mixed_spill_budgets_are_rejected() -> None:
    report = StepSearchReport(
        total_sequences_per_step=32,
        sequence_length=1024,
        budgets=((8 << 30, 32 << 30), (16 << 30, 64 << 30)),
        geometries=(),
        points=(),
        skipped=(),
    )
    with pytest.raises(ValueError, match="one spill budget"):
        plot_step_search(report, ".")
