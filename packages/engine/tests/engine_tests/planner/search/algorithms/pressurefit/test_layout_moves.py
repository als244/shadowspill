"""A layout that overran the pool is answered by moving a fetch, and the
record of it travels."""

from __future__ import annotations

from shadowspill.planner import GenericPlanningOptions, pressurefit
from shadowspill.planner.diagnostics import (
    CandidateDiagnostic,
    PlanningRepairDiagnostics,
    ReductionStep,
)

from ...._examples import (
    training_chain_config,
    training_chain_initial,
    training_chain_program,
)


def test_a_candidate_without_a_pool_never_moves_for_a_layout() -> None:
    result = pressurefit(
        training_chain_program(10),
        initial_residency=training_chain_initial(10),
        config=training_chain_config(500),
        generic=GenericPlanningOptions(minimum_object_bytes_evict_eligible=0),
    )
    candidates = [
        candidate
        for problem in result.diagnostics.resolved_programs
        for candidate in problem.candidate_evaluations
    ]
    assert candidates
    # nothing is measured without a pool, so nothing is moved for a layout
    assert all(c.repairs.layout_fetch_delay_attempts == 0 for c in candidates)
    assert all(not step.moved for c in candidates for step in c.steps)


def test_the_move_counts_toward_the_total_and_travels() -> None:
    repairs = PlanningRepairDiagnostics(
        simulation_fetch_delay_attempts=2, layout_fetch_delay_attempts=3
    )
    assert repairs.total_attempts == 5
    restored = PlanningRepairDiagnostics.from_value(repairs.to_dict(), "r")
    assert restored == repairs
    assert (repairs + repairs).layout_fetch_delay_attempts == 6


def test_records_written_before_layout_moves_read_as_none() -> None:
    value = PlanningRepairDiagnostics(simulation_fetch_delay_attempts=1).to_dict()
    del value["layout_miss"]
    value["total_attempts"] = 1
    assert (
        PlanningRepairDiagnostics.from_value(value, "r").layout_fetch_delay_attempts
        == 0
    )
    step = ReductionStep(
        makespan_ns=1,
        required_bytes=0,
        capacity_bytes=1,
        cut_aliases=(),
        repairs=0,
        simulation_status=0,
        capacity_violations=0,
        simulated=True,
        measured=False,
        placed=False,
        refined=False,
        best=False,
        answer=False,
    ).to_dict()
    del step["outcome"]["moved"]
    assert ReductionStep.from_value(step, "s").moved is False
    candidate = CandidateDiagnostic(
        candidate_id="tight-stall/packed-fit",
        selection_id="none",
        status="valid",
        makespan_ns=1,
    )
    assert candidate.repairs.layout_fetch_delay_attempts == 0
