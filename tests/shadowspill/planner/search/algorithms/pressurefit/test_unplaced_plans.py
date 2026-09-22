"""A candidate reports the fastest plan it could not place, and how many it measured."""

from __future__ import annotations

from shadowspill.planner import GenericPlanningOptions, pressurefit
from shadowspill.planner.diagnostics import CandidateDiagnostic, PlanningDiagnostics

from ...._examples import (
    training_chain_config,
    training_chain_initial,
    training_chain_program,
)


def _candidates(diagnostics: PlanningDiagnostics) -> list[CandidateDiagnostic]:
    return [
        candidate
        for problem in diagnostics.resolved_programs
        for candidate in problem.candidate_evaluations
    ]


def test_unplaced_record_is_consistent_and_travels() -> None:
    result = pressurefit(
        training_chain_program(10),
        initial_residency=training_chain_initial(10),
        config=training_chain_config(500),
        generic=GenericPlanningOptions(minimum_object_bytes_evict_eligible=0),
    )
    candidates = _candidates(result.diagnostics)
    assert candidates
    for candidate in candidates:
        # a best exists exactly when a plan was measured and missed
        assert (candidate.best_unplaced_makespan_ns is None) == (
            candidate.unplaced_plans == 0
        )
        # every unplaced plan was a measurement that did not fit
        assert candidate.unplaced_plans <= (
            candidate.placements_attempted - candidate.placements_admitted
        )
    restored = PlanningDiagnostics.from_value(result.diagnostics.to_dict(), "d")
    assert [
        (c.best_unplaced_makespan_ns, c.unplaced_plans) for c in _candidates(restored)
    ] == [(c.best_unplaced_makespan_ns, c.unplaced_plans) for c in candidates]


def test_records_written_before_the_unplaced_record_read_as_none() -> None:
    value = CandidateDiagnostic(
        candidate_id="tight-stall/packed-fit",
        selection_id="none",
        status="valid",
        makespan_ns=1,
    ).to_dict()
    del value["outcome"]["best_unplaced_makespan_ns"]
    del value["outcome"]["unplaced_plans"]
    restored = CandidateDiagnostic.from_value(value, "c", "none")
    assert restored.best_unplaced_makespan_ns is None
    assert restored.unplaced_plans == 0
