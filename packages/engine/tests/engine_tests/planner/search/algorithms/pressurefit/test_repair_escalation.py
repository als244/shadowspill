"""A failure that repeats asks for more; an ask that cannot be met is taken back."""

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


def test_escalations_are_counted_and_never_exceed_what_was_asked() -> None:
    result = pressurefit(
        training_chain_program(10),
        initial_residency=training_chain_initial(10),
        config=training_chain_config(500),
        generic=GenericPlanningOptions(minimum_object_bytes_evict_eligible=0),
    )

    candidates = _candidates(result.diagnostics)
    assert candidates
    for candidate in candidates:
        # an escalation is a pressure repair, so it never outnumbers them
        assert candidate.pressure_escalations <= (
            candidate.repairs.simulation_pressure_boundary_attempts
        )
        assert candidate.escalations_taken_back <= candidate.pressure_escalations
    # the counters travel with the record
    restored = PlanningDiagnostics.from_value(result.diagnostics.to_dict(), "d")
    assert [
        (c.pressure_escalations, c.escalations_taken_back)
        for c in _candidates(restored)
    ] == [(c.pressure_escalations, c.escalations_taken_back) for c in candidates]


def test_records_written_before_escalation_read_as_none() -> None:
    value = CandidateDiagnostic(
        candidate_id="tight-stall/packed-fit",
        selection_id="none",
        status="valid",
        makespan_ns=1,
    ).to_dict()
    del value["outcome"]["pressure_escalations"]
    del value["outcome"]["escalations_taken_back"]
    restored = CandidateDiagnostic.from_value(value, "c", "none")
    assert restored.pressure_escalations == 0
    assert restored.escalations_taken_back == 0
