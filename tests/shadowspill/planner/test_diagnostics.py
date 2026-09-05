from __future__ import annotations

import pytest

from shadowspill.planner import (
    CandidateDiagnostic,
    PressureFitDiagnostics,
    PressureFitRepairDiagnostics,
    PressureFitSectionTiming,
    PressureFitWorkDiagnostics,
    ReductionStep,
    ResolvedProgramDiagnostics,
    TaskAlternativeChoiceDiagnostic,
)


def _diagnostics() -> PressureFitDiagnostics:
    repairs = PressureFitRepairDiagnostics(
        admission_fetch_advance_attempts=1,
        simulation_pressure_boundary_attempts=2,
    )
    candidate_work = PressureFitWorkDiagnostics(
        schedule_emissions=1,
        simulation_calls=3,
        admission_calls=2,
        sections=PressureFitSectionTiming(
            total_ns=80,
            reduce_ns=10,
            emit_ns=20,
            simulate_ns=30,
            digest_ns=5,
            admit_ns=15,
            residual_ns=15,
        ),
    )
    step = ReductionStep(
        makespan_ns=1_000,
        required_bytes=2_048,
        capacity_bytes=4_096,
        cut_aliases=(3, 7),
        repairs=1,
        simulation_status=0,
        capacity_violations=0,
        simulated=True,
        measured=True,
        placed=True,
        refined=False,
        best=True,
        answer=True,
    )
    selection_id = "stage_0=recompute"
    candidate = CandidateDiagnostic(
        candidate_id="tight-transfer/latest-safe-coalesced",
        selection_id=selection_id,
        status="valid",
        makespan_ns=1_000,
        schedule_digest="a" * 64,
        repairs=repairs,
        work=candidate_work,
        steps=(step,),
    )
    problem = ResolvedProgramDiagnostics(
        selection_id=selection_id,
        choices=(TaskAlternativeChoiceDiagnostic("stage_0", "recompute"),),
        selected_candidate_id=candidate.candidate_id,
        selected_makespan_ns=1_000,
        candidate_evaluations=(candidate,),
        work=PressureFitWorkDiagnostics(
            schedule_emissions=1,
            simulation_calls=3,
            admission_calls=2,
            sections=PressureFitSectionTiming(
                total_ns=90,
                reduce_ns=10,
                emit_ns=20,
                simulate_ns=30,
                digest_ns=5,
                admit_ns=15,
                residual_ns=25,
            ),
        ),
    )
    return PressureFitDiagnostics(
        selected_candidate_id=candidate.candidate_id,
        selected_selection_id=selection_id,
        selected_makespan_ns=1_000,
        resolved_programs=(problem,),
        work=PressureFitWorkDiagnostics(
            schedule_emissions=1,
            simulation_calls=4,
            admission_calls=3,
            sections=PressureFitSectionTiming(
                total_ns=97,
                reduce_ns=10,
                emit_ns=20,
                simulate_ns=30,
                digest_ns=5,
                select_ns=7,
                admit_ns=21,
                residual_ns=25,
            ),
        ),
    )


def test_pressurefit_diagnostics_round_trip_preserves_hierarchy() -> None:
    source = _diagnostics()
    restored = PressureFitDiagnostics.from_value(source.to_dict(), "diagnostics")

    assert restored == source
    assert restored.resolved_program_count == 1
    assert restored.candidate_policy_count == 1
    assert restored.candidate_evaluation_count == 1
    assert restored.repairs.total_attempts == 3
    assert restored.repairs.pressure_boundary_attempts == 2
    candidate = restored.resolved_programs[0].candidate_evaluations[0]
    assert candidate.residency_strategy == "tight-transfer"
    assert candidate.fetch_rule == "latest-safe"
    assert candidate.coalesced
    assert candidate.steps[0].cut_aliases == (3, 7)
    assert candidate.steps[0].answer
    sections = restored.work.sections
    assert sections.total_ns == sections.named_ns + sections.residual_ns


def test_pressurefit_diagnostics_rejects_old_flat_schema() -> None:
    source = _diagnostics()
    flat = {
        "selected_candidate_id": source.selected_candidate_id,
        "selected_selection_id": source.selected_selection_id,
        "selected_makespan_ns": source.selected_makespan_ns,
        "candidate_count": source.candidate_evaluation_count,
        "valid_candidate_count": source.valid_candidate_evaluation_count,
        "candidates": [],
    }

    with pytest.raises(ValueError, match="unsupported schema"):
        PressureFitDiagnostics.from_value(flat, "diagnostics")


def test_candidate_diagnostic_rejects_old_flat_schema() -> None:
    with pytest.raises(ValueError, match="candidate_policy must be an object"):
        CandidateDiagnostic.from_value(
            {
                "candidate_id": "tight-transfer/latest-safe",
                "selection_id": "none",
                "status": "valid",
            },
            "candidate",
            "none",
        )


def test_physical_prediction_updates_nested_selected_candidate() -> None:
    source = _diagnostics()
    updated = source.replace_selected_makespan(1_250)

    assert source.selected_makespan_ns == 1_000
    assert updated.selected_makespan_ns == 1_250
    problem = updated.resolved_programs[0]
    assert problem.selected_makespan_ns == 1_250
    assert problem.candidate_evaluations[0].makespan_ns == 1_250
    assert problem.candidate_evaluations[0].work == (
        source.resolved_programs[0].candidate_evaluations[0].work
    )
