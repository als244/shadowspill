"""The plan summary's accounting identity on a fabricated selected plan."""

from __future__ import annotations

from shadowspill.planner import PressureFitOptions
from shadowspill.planner.diagnostics import (
    CandidateDiagnostic,
    PressureFitDiagnostics,
    RecomputationProblemDiagnostics,
)
from shadowspill.planner.diagnostics.plan import PlanSummary, summarize_selected_plan
from shadowspill.planner.result import PressureFitResult
from shadowspill.simulator import (
    DeviceSimulationConfig,
    SimulationConfig,
    simulate,
)
from tests.shadowspill.ir._examples import (
    SAVE_SELECTION,
    representative_plan,
    representative_program,
)


def _result() -> PressureFitResult:
    program = representative_program()
    plan = representative_plan()
    config = SimulationConfig(
        devices=(
            DeviceSimulationConfig(
                device_id="cuda_0",
                capacity_bytes=1 << 20,
                fetch_bandwidth_bytes_per_second=1 << 30,
                evict_bandwidth_bytes_per_second=1 << 30,
                fetch_latency_ns=0,
                evict_latency_ns=0,
            ),
        ),
        spill_capacity_bytes=1 << 20,
    )
    simulation = simulate(
        program, plan.schedule, selections=SAVE_SELECTION, config=config
    )
    return PressureFitResult(
        program=program,
        options=PressureFitOptions(workers=1),
        initial_residency=plan.schedule.initial_residency,
        final_residency=plan.schedule.final_residency,
        simulation_config=config,
        schedule=plan.schedule,
        selections=SAVE_SELECTION,
        simulation=simulation,
        diagnostics=PressureFitDiagnostics(
            selected_candidate_id="fixture",
            selected_selection_id="fixture",
            selected_makespan_ns=simulation.makespan_ns,
            recomputation_problems=(
                RecomputationProblemDiagnostics(
                    selection_id="fixture",
                    choices=(),
                    selected_candidate_id="fixture",
                    selected_makespan_ns=simulation.makespan_ns,
                    candidate_evaluations=(
                        CandidateDiagnostic(
                            candidate_id="fixture",
                            selection_id="fixture",
                            status="valid",
                            makespan_ns=simulation.makespan_ns,
                        ),
                    ),
                ),
            ),
        ),
        admission_facts=None,
    )


def test_summary_parts_identify_to_the_simulated_step() -> None:
    summary = summarize_selected_plan(
        _result(),
        phase_timings_ns=(("capture_lowering", 2_000_000_000), ("selection", 500)),
    )
    assert dict(summary.planning_phase_seconds) == {
        "capture_lowering": 2.0,
        "selection": 5e-7,
    }
    result = _result()
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
    assert summary.transfer_bytes_fetched == fetched
    assert summary.transfer_bytes_evicted == evicted
    assert summary.fetch_bandwidth_bytes_per_second == 1 << 30
    assert summary.evict_bandwidth_bytes_per_second == 1 << 30
    assert dict(summary.selected_candidate) == {
        "residency_strategy": summary.selected_candidate["residency_strategy"],
        "fetch_rule": summary.selected_candidate["fetch_rule"],
        "coalesced": summary.selected_candidate["coalesced"],
        "repairs_at_best": None,
    }
    assert summary.as_dict()["selected_candidate"] == dict(summary.selected_candidate)
    assert list(summary.planning_phase_seconds) == ["capture_lowering", "selection"]
    reassembled = (
        summary.unconstrained_step_seconds
        + summary.recomputation_overhead_seconds
        + summary.idle_seconds
        + summary.terminal_writeback_seconds
    )
    assert abs(reassembled - summary.simulated_step_seconds) < 1e-12
    assert summary.selection_count == len(SAVE_SELECTION)
    # Save and recompute share one profile in the representative program, so
    # the chosen option is never strictly costlier than the cheapest.
    assert summary.recompute_selection_count == 0
    assert summary.recomputation_overhead_seconds == 0.0


def test_recompute_fraction_is_guarded_against_empty_selections() -> None:
    empty = PlanSummary(
        simulated_step_seconds=1.0,
        unconstrained_step_seconds=1.0,
        recomputation_overhead_seconds=0.0,
        idle_seconds=0.0,
        terminal_writeback_seconds=0.0,
        recompute_selection_count=0,
        selection_count=0,
    )
    assert empty.recompute_selection_fraction == 0.0
    assert dict(empty.planning_phase_seconds) == {}
