from __future__ import annotations

import pytest

from shadowspill.planner import (
    CandidateDiagnostic,
    PressureFitDiagnostics,
    PressureFitRepairDiagnostics,
    PressureFitWorkDiagnostics,
    RecomputationChoiceDiagnostic,
    RecomputationProblemDiagnostics,
)


def _diagnostics() -> PressureFitDiagnostics:
    repairs = PressureFitRepairDiagnostics(
        admission_prefetch_advance_attempts=1,
        simulation_pressure_boundary_attempts=2,
    )
    candidate_work = PressureFitWorkDiagnostics(
        evaluation_time_ns=80,
        residency_cache_misses=1,
        schedule_emissions=1,
        simulation_calls=3,
        admission_calls=2,
        residency_time_ns=10,
        schedule_time_ns=20,
        simulation_time_ns=30,
        admission_time_ns=15,
        digest_time_ns=5,
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
    )
    problem = RecomputationProblemDiagnostics(
        selection_id=selection_id,
        choices=(RecomputationChoiceDiagnostic("stage_0", "recompute"),),
        selected_candidate_id=candidate.candidate_id,
        selected_makespan_ns=1_000,
        candidate_evaluations=(candidate,),
        work=PressureFitWorkDiagnostics(
            evaluation_time_ns=90,
            residency_cache_misses=1,
            schedule_emissions=1,
            simulation_calls=3,
            admission_calls=2,
            residency_time_ns=10,
            schedule_time_ns=20,
            simulation_time_ns=30,
            admission_time_ns=15,
            digest_time_ns=5,
        ),
    )
    return PressureFitDiagnostics(
        selected_candidate_id=candidate.candidate_id,
        selected_selection_id=selection_id,
        selected_makespan_ns=1_000,
        recomputation_problems=(problem,),
        work=PressureFitWorkDiagnostics(
            evaluation_time_ns=90,
            residency_cache_misses=1,
            schedule_emissions=1,
            simulation_calls=3,
            result_simulation_calls=1,
            admission_calls=2,
            result_admission_calls=1,
            residency_time_ns=10,
            schedule_time_ns=20,
            simulation_time_ns=30,
            result_simulation_time_ns=7,
            admission_time_ns=15,
            result_admission_time_ns=6,
            digest_time_ns=5,
        ),
    )


def test_pressurefit_diagnostics_round_trip_preserves_hierarchy() -> None:
    source = _diagnostics()
    restored = PressureFitDiagnostics.from_value(source.to_dict(), "diagnostics")

    assert restored == source
    assert restored.recomputation_problem_count == 1
    assert restored.candidate_policy_count == 1
    assert restored.candidate_evaluation_count == 1
    assert restored.repairs.total_attempts == 3
    assert restored.repairs.pressure_boundary_attempts == 2
    candidate = restored.recomputation_problems[0].candidate_evaluations[0]
    assert candidate.residency_strategy == "tight-transfer"
    assert candidate.prefetch_rule == "latest-safe"
    assert candidate.coalesced


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
        )


def test_physical_prediction_updates_nested_selected_candidate() -> None:
    source = _diagnostics()
    updated = source.with_selected_makespan(1_250)

    assert source.selected_makespan_ns == 1_000
    assert updated.selected_makespan_ns == 1_250
    problem = updated.recomputation_problems[0]
    assert problem.selected_makespan_ns == 1_250
    assert problem.candidate_evaluations[0].makespan_ns == 1_250
    assert problem.candidate_evaluations[0].work == (
        source.recomputation_problems[0].candidate_evaluations[0].work
    )
