"""The step-search figures render from a fabricated report."""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType

import pytest

from shadowspill.planner import StepDataOrdering
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
        recomputing_group_count=2,
        task_alternative_group_count=4,
        flexible_group_count=4,
        transfer_bytes_fetched=int(4e9),
        transfer_bytes_evicted=int(3e9),
        fetch_busy_ns=int(0.3e9),
        evict_busy_ns=int(0.2e9),
        fetch_solo_bandwidth_bytes_per_second=int(20e9),
        fetch_concurrent_bandwidth_bytes_per_second=(int(20e9)),
        evict_solo_bandwidth_bytes_per_second=int(20e9),
        evict_concurrent_bandwidth_bytes_per_second=(int(20e9)),
        planning_phase_seconds=MappingProxyType({}),
    )


def _point(execution: int, spill: int, step: float) -> StepSearchPoint:
    return StepSearchPoint(
        candidate="8",
        accumulation_count=4,
        ordering=StepDataOrdering.depth_first(4),
        execution_budget_bytes=execution,
        spill_budget_bytes=spill,
        status="succeeded",
        makespan_seconds=step,
        summary=_summary(step),
        error=None,
        search_seconds=0.1,
    )


def test_the_ordering_ladder_renders_beside_the_geometry_figures(
    tmp_path: Path,
) -> None:
    from dataclasses import replace

    slow = _point(10 << 30, 1 << 30, 20.0)
    fast = replace(slow, ordering=StepDataOrdering(2, 2), makespan_seconds=18.0)
    report = StepSearchReport(
        metadata={"units_per_step": 32 * 1, "unit_label": "tokens"},
        budgets=((10 << 30, 1 << 30),),
        geometries=(),
        points=(slow, fast),
    )
    written = plot_step_search(report, tmp_path)
    ladder = tmp_path / "sim" / "orderings" / "8x4.png"
    assert ladder in written and ladder.exists()
    rows = (tmp_path / "raw_data" / "points.csv").read_text().splitlines()
    assert rows[0].split(",")[2] == "ordering"
    assert {row.split(",")[2] for row in rows[1:]} == {"4x1rp", "2x2rp"}


def test_every_figure_renders(tmp_path: Path) -> None:
    budgets = ((8 << 30, 64 << 30), (16 << 30, 64 << 30))
    report = StepSearchReport(
        metadata={"units_per_step": 32 * 1024, "unit_label": "tokens"},
        budgets=budgets,
        geometries=(),
        points=(_point(8 << 30, 64 << 30, 6.0), _point(16 << 30, 64 << 30, 4.0)),
    )
    written = plot_step_search(report, tmp_path)
    assert written
    for path in written:
        assert path.exists() and path.stat().st_size > 0


def test_run_figures_render(tmp_path: Path) -> None:
    from shadowspill.plots import RunBudgetOutcome, plot_step_run

    written = plot_step_run(
        [
            RunBudgetOutcome(
                execution_budget_bytes=budget,
                simulated_step_seconds=simulated,
                measured_step_seconds=measured,
                profiled_task_seconds=simulated * 0.8,
                real_task_seconds=measured * 0.8,
                simulated_idle_seconds=simulated * 0.15,
                real_idle_seconds=measured * 0.13,
                simulated_entry_delay_seconds=simulated * 0.03,
                real_entry_delay_seconds=measured * 0.05,
                real_terminal_tail_seconds=measured * 0.02,
                terminal_tail_seconds=simulated * 0.02,
            )
            for budget, simulated, measured in (
                (16 << 30, 4.0, 4.4),
                (8 << 30, 6.0, 6.9),
            )
        ],
        tmp_path,
        units_per_step=32 * 1024,
    )
    for path in written:
        assert path.exists() and path.stat().st_size > 0


def test_mixed_spill_budgets_are_rejected() -> None:
    report = StepSearchReport(
        metadata={"units_per_step": 32 * 1024, "unit_label": "tokens"},
        budgets=((8 << 30, 32 << 30), (16 << 30, 64 << 30)),
        geometries=(),
        points=(),
    )
    with pytest.raises(ValueError, match="one spill budget"):
        plot_step_search(report, ".")


def test_fidelity_uses_complete_invocation_and_keeps_cycle_separate() -> None:
    from shadowspill.plots import RunBudgetOutcome

    result = RunBudgetOutcome(
        execution_budget_bytes=8 << 30,
        simulated_step_seconds=4.0,
        measured_step_seconds=6.0,
        profiled_task_seconds=3.0,
        real_task_seconds=3.5,
        simulated_idle_seconds=0.5,
        real_idle_seconds=0.5,
        simulated_entry_delay_seconds=0.2,
        real_entry_delay_seconds=0.4,
        terminal_tail_seconds=0.3,
        real_terminal_tail_seconds=0.6,
        recomputation_seconds=0.5,
    )
    assert sum(result.components(measured=False)) == pytest.approx(4.0)
    assert sum(result.components(measured=True)) == pytest.approx(5.0)
    assert result.traced_step_seconds == pytest.approx(5.0)
    assert result.trace_relative_error == pytest.approx(0.25)
    assert result.relative_error == pytest.approx(0.50)  # whole-cycle throughput


@pytest.mark.parametrize("include_complete", [False, True])
def test_missing_transfer_times_preserve_throughput_and_replot(
    tmp_path: Path, include_complete: bool
) -> None:
    import csv
    from dataclasses import replace

    from benchmarking.replot import _run_entries
    from shadowspill.plots import RunBudgetOutcome, plot_step_run

    incomplete = RunBudgetOutcome(
        execution_budget_bytes=8 << 30,
        simulated_step_seconds=4.0,
        measured_step_seconds=6.0,
        profiled_task_seconds=3.0,
        real_task_seconds=3.5,
        simulated_idle_seconds=0.5,
        real_idle_seconds=0.5,
        simulated_entry_delay_seconds=0.2,
        real_entry_delay_seconds=0.4,
        terminal_tail_seconds=0.3,
        real_terminal_tail_seconds=None,
        step_seconds=(6.0, 6.1),
    )
    assert incomplete.traced_step_seconds is None
    assert incomplete.trace_relative_error is None
    assert incomplete.relative_error == pytest.approx(0.5)
    entries = [incomplete]
    if include_complete:
        entries.append(
            replace(
                incomplete,
                execution_budget_bytes=10 << 30,
                real_terminal_tail_seconds=0.6,
            )
        )
    written = plot_step_run(entries, tmp_path, units_per_step=8192)
    assert all(path.stat().st_size > 0 for path in written)
    table = tmp_path / "raw_data" / "run_budgets.csv"
    with table.open(newline="") as handle:
        first = next(csv.DictReader(handle))
    assert float(first["measured_units_per_second"]) == pytest.approx(8192 / 6)
    assert first["real_terminal_tail_seconds"] == ""
    assert first["traced_step_seconds"] == ""
    assert first["trace_relative_error"] == ""
    restored = _run_entries(table.parent, ())
    assert list(restored) == entries
    assert all(
        path.stat().st_size > 0
        for path in plot_step_run(restored, tmp_path / "replotted", units_per_step=8192)
    )


def test_lane_csv_exports_actual_busy_time_and_blended_rate(tmp_path):
    import csv

    point = _point(8 << 30, 64 << 30, 6.0)
    report = StepSearchReport(
        metadata={"units_per_step": 32 * 1024, "unit_label": "tokens"},
        budgets=((8 << 30, 64 << 30),),
        geometries=(),
        points=(point,),
    )
    written = plot_step_search(report, tmp_path)
    assert any(p.name == "blended_bandwidth.png" for p in written)
    with (tmp_path / "raw_data/points.csv").open() as f:
        row = next(csv.DictReader(f))
    assert float(row["fetch_utilization"]) == pytest.approx(0.3 / 6)
    assert float(row["fetch_blended_bandwidth_bytes_per_second"]) == pytest.approx(
        4e9 / 0.3
    )
