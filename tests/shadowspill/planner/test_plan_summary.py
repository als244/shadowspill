"""The plan summary's accounting identity on a fabricated selected plan."""

from __future__ import annotations

from shadowspill.planner.diagnostics.plan_summary import (
    PlanSummary,
    summarize_selected_plan,
)
from tests.shadowspill.ir._examples import SAVE_SELECTION

from ._examples import representative_result


def test_summary_parts_identify_to_the_simulated_step() -> None:
    summary = summarize_selected_plan(
        representative_result(),
        phase_timings_ns=(("capture_lowering", 2_000_000_000), ("selection", 500)),
    )
    assert dict(summary.planning_phase_seconds) == {
        "capture_lowering": 2.0,
        "selection": 5e-7,
    }
    result = representative_result()
    fetched = sum(
        item.bytes
        for item in result.simulation.transfer_intervals
        if item.direction.value == "fetch"
    )
    evicted = sum(
        item.bytes
        for item in result.simulation.transfer_intervals
        if item.direction.value != "fetch"
    )
    assert summary.spill_peak_bytes == result.simulation.spill_peak_bytes
    assert summary.transfer_bytes_fetched == fetched
    assert summary.transfer_bytes_evicted == evicted
    assert summary.fetch_solo_bandwidth_bytes_per_second == 1 << 30
    assert summary.evict_solo_bandwidth_bytes_per_second == 1 << 30
    assert dict(summary.selected_candidate) == {
        "residency_strategy": summary.selected_candidate["residency_strategy"],
        "fetch_rule": summary.selected_candidate["fetch_rule"],
        "repairs_at_best": None,
        "best_unplaced_makespan_ns": None,
        "unplaced_plans": 0,
        "placement_gap": None,
    }
    assert summary.as_dict()["selected_candidate"] == dict(summary.selected_candidate)
    assert list(summary.planning_phase_seconds) == ["capture_lowering", "selection"]
    reassembled = (
        summary.unconstrained_step_seconds
        + summary.recomputation_overhead_seconds
        + summary.task_contention_seconds
        + summary.idle_seconds
        + summary.terminal_writeback_seconds
    )
    assert abs(reassembled - summary.simulated_step_seconds) < 1e-12
    assert summary.task_alternative_group_count == len(SAVE_SELECTION)
    # Save and recompute share one profile in the representative program, so
    # the chosen option is never strictly costlier than the cheapest.
    assert summary.recomputing_group_count == 0
    assert summary.recomputation_overhead_seconds == 0.0


def test_contention_inside_a_task_is_not_reported_as_idle() -> None:
    from dataclasses import replace

    import pytest

    result = representative_result()
    baseline = summarize_selected_plan(result)
    *earlier, last = result.simulation.task_intervals
    # Shared-resource contention extends the final task without changing
    # its input-readiness wait or the isolated profile's work.
    simulation = replace(
        result.simulation,
        task_intervals=(*earlier, replace(last, end_ns=last.end_ns + 100)),
        makespan_ns=result.simulation.makespan_ns + 100,
    )
    summary = summarize_selected_plan(replace(result, simulation=simulation))
    assert summary.idle_seconds == baseline.idle_seconds
    assert summary.task_contention_seconds == pytest.approx(100e-9)
    assert summary.simulated_step_seconds == pytest.approx(
        summary.unconstrained_step_seconds
        + summary.recomputation_overhead_seconds
        + summary.task_contention_seconds
        + summary.idle_seconds
        + summary.terminal_writeback_seconds
    )
    assert PlanSummary.from_dict(summary.as_dict()) == summary


def test_recompute_fraction_is_guarded_against_empty_selections() -> None:
    empty = PlanSummary(
        simulated_step_seconds=1.0,
        unconstrained_step_seconds=1.0,
        recomputation_overhead_seconds=0.0,
        idle_seconds=0.0,
        terminal_writeback_seconds=0.0,
        recomputing_group_count=0,
        task_alternative_group_count=0,
        flexible_group_count=0,
    )
    assert empty.recomputing_group_fraction == 0.0
    assert dict(empty.planning_phase_seconds) == {}


def test_selected_candidate_is_read_from_the_selected_program() -> None:
    """A policy is evaluated once per resolved program; the summary describes
    the evaluation in the program the search selected, not the last one."""

    from dataclasses import replace

    from shadowspill.planner.diagnostics import (
        ResolvedProgramDiagnostics,
    )

    result = representative_result()
    diagnostics = result.diagnostics
    (selected_program,) = diagnostics.resolved_programs
    (candidate,) = selected_program.candidate_evaluations
    chosen = replace(
        candidate,
        repairs_at_best=2,
        best_unplaced_makespan_ns=candidate.makespan_ns - 1,
        unplaced_plans=3,
    )
    # the same policy, evaluated in another resolved program, never placed
    other = ResolvedProgramDiagnostics(
        selection_id="other",
        choices=(),
        selected_candidate_id=None,
        selected_makespan_ns=None,
        candidate_evaluations=(
            replace(
                candidate,
                selection_id="other",
                status="infeasible",
                failure_kind="unplaceable",
                makespan_ns=None,
            ),
        ),
    )
    diagnostics = replace(
        diagnostics,
        resolved_programs=(
            replace(selected_program, candidate_evaluations=(chosen,)),
            other,
        ),
    )
    summary = summarize_selected_plan(
        replace(result, diagnostics=diagnostics), phase_timings_ns=()
    )
    selected = dict(summary.selected_candidate)
    assert selected["repairs_at_best"] == 2
    assert selected["unplaced_plans"] == 3
    assert selected["best_unplaced_makespan_ns"] == candidate.makespan_ns - 1
    assert selected["placement_gap"] == round(
        candidate.makespan_ns / (candidate.makespan_ns - 1), 4
    )
