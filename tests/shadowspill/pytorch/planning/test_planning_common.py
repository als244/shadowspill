from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from shadowspill.errors import (
    AdmissionError,
    PlanInfeasibleError,
    PlanningError,
    PlanSearchExhaustedError,
)
from shadowspill.planner import (
    CandidateDiagnostic,
    PressureFitInfeasibleError,
    PressureFitRepairDiagnostics,
    PressureFitSearchExhaustedError,
)
from shadowspill.pytorch import (
    TensorSpec,
)
from shadowspill.pytorch.materialization import representative_cpu_inputs
from shadowspill.pytorch.planning.admission.physical import physical_admission
from shadowspill.pytorch.planning.common import (
    PlanningTimer,
    estimate_spill_reservation,
    public_infeasible_plan_error,
    public_search_exhausted_error,
    simulation_capacity,
    validate_budgets,
    validate_cpu_model,
    workspace_reserve,
)
from shadowspill.pytorch.profiling import TaskMeasurement
from shadowspill.pytorch.runtime_adapter.runtime import _adapter_path


def test_phase_timer_attributes_compilation_and_profiling_without_overlap() -> None:
    timer = PlanningTimer(verbose=False)
    timer.values = [
        ("capture_lowering", 11),
        ("compiler_manifest", 30),
        ("structural_profiling", 100),
        ("compilation", 20),
        ("program_lowering", 13),
    ]
    profiler = SimpleNamespace(
        compilation_wall_time_ns=40,
        saved_control_compilation_wall_time_ns=0,
        profiling_wall_time_ns=60,
        entrypoint_warmup_wall_time_ns=5,
    )

    timer.attribute_compilation_and_profiling(profiler)  # type: ignore[arg-type]

    assert timer.values == [
        ("capture_lowering", 11),
        ("compiled_entrypoint_construction", 40),
        ("unique_stage_warmup_profiling", 60),
        ("cached_entrypoint_warmup", 5),
        ("profile_cache_and_entrypoint_orchestration", 45),
        ("program_lowering", 13),
    ]


def test_phase_timer_annotates_nested_planning_failures() -> None:
    timer = PlanningTimer(verbose=False)
    with (
        pytest.raises(RuntimeError) as captured,
        timer.measure("outer"),
        timer.measure("inner"),
    ):
        raise RuntimeError("broken")

    assert captured.value.__notes__ == [
        "ShadowSpill planning phase 'inner' failed after 0.000 seconds",
        "ShadowSpill planning phase 'outer' failed after 0.000 seconds",
    ]


def test_planning_admission_helpers_reject_invalid_values() -> None:
    model = nn.Linear(2, 2)
    with pytest.raises(TypeError, match="model"):
        validate_cpu_model(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="execution_budget"):
        validate_budgets(True, 1 << 30)
    with pytest.raises(TypeError, match="spill_budget"):
        validate_budgets(1 << 30, True)
    with pytest.raises(AdmissionError, match="positive"):
        validate_budgets(0, 1 << 30)
    with pytest.raises(AdmissionError, match="spill_budget"):
        validate_budgets(1 << 30, 0)
    with pytest.raises(PlanningError, match="CPU resident"):
        validate_cpu_model(nn.Linear(2, 2, device="meta"))
    with pytest.raises(AdmissionError, match="spill-pool budget"):
        estimate_spill_reservation(model, [torch.ones(2)], 1)


def test_pressurefit_infeasibility_is_structured_for_plan_callers() -> None:
    internal = PressureFitInfeasibleError(
        "task cannot fit",
        kind="task_footprint",
        device_id="cuda_0",
        boundary_task_id="task_000017",
        required_bytes=123,
        capacity_bytes=100,
    )

    public = public_infeasible_plan_error(internal)

    assert isinstance(public, PlanInfeasibleError)
    assert isinstance(public, AdmissionError)
    assert public.kind == "task_footprint"
    assert public.boundary_task_id == "task_000017"
    assert public.required_bytes == 123
    assert public.capacity_bytes == 100
    assert "could not construct a feasible memory schedule" in str(public)
    assert "boundary_task: task_000017" in str(public)


def test_pressurefit_search_exhaustion_is_not_reported_as_infeasibility() -> None:
    internal = PressureFitSearchExhaustedError(
        "bounded search stopped",
        diagnostics=(
            CandidateDiagnostic(
                candidate_id="tight/latest-safe",
                selection_id="selection_0",
                status="exhausted",
                failure_kind="repair_budget_exhausted",
                repairs=PressureFitRepairDiagnostics(
                    unclassified_attempts=25
                ),
            ),
        ),
    )

    public = public_search_exhausted_error(internal)

    assert isinstance(public, PlanSearchExhaustedError)
    assert not isinstance(public, AdmissionError)
    assert "exhausted_candidates: 1" in str(public)
    assert "largest_repair_count: 25" in str(public)


def test_workspace_and_capacity_helpers_are_explicit() -> None:
    measurement = TaskMeasurement(1, 3, 100, (100,), (1,), "test")
    assert workspace_reserve((measurement,)) == 512 << 20
    assert simulation_capacity(1 << 30, 512 << 20, (measurement,)) == (
        (512 << 20) + 100
    )
    assert (
        simulation_capacity(
            1 << 30,
            512 << 20,
            (measurement,),
            fixed_slab_bytes=32,
        )
        == (512 << 20) + 68
    )
    with pytest.raises(AdmissionError, match="smaller"):
        simulation_capacity(1, 2, ())
    with pytest.raises(PlanInfeasibleError, match="no capacity") as captured:
        simulation_capacity(512 << 20, 512 << 20, ())
    assert captured.value.kind == "analytic_capacity"
    assert captured.value.required_bytes == 1
    assert captured.value.capacity_bytes == 0


def test_plan_reservation_can_be_smaller_than_runtime_spill_pool() -> None:
    memory = SimpleNamespace(
        execution=SimpleNamespace(physical_capacity=200),
        execution_budget=180,
        spill_budget=100,
    )
    installed = SimpleNamespace(
        admission=SimpleNamespace(
            baseline_bytes=10,
            provider_headroom_bytes=10,
            spill_pool_bytes=112,
        )
    )

    admission = physical_admission(
        memory,  # type: ignore[arg-type]
        installed,  # type: ignore[arg-type]
        workspace_reserve=20,
        predicted_spill_peak_bytes=90,
        predicted_fragmentation_bytes=8,
    )

    assert admission.spill_budget_bytes == 100
    assert admission.spill_reservation_bytes == 90


def test_representatives_and_adapter_path_contract(tmp_path: Path) -> None:
    values = representative_cpu_inputs(
        [TensorSpec((2, 3), torch.float32), torch.empty(4, device="meta"), "x"]
    )
    assert values[0].device.type == "cpu" and values[0].shape == (2, 3)
    assert values[1].device.type == "cpu" and values[1].shape == (4,)
    assert values[2] == "x"

    try:
        configured = _adapter_path(None)
    except RuntimeError:
        configured = None
    missing = tmp_path / "missing.so"
    with pytest.raises(RuntimeError, match="not found"):
        _adapter_path(missing)
    if configured is not None:
        assert _adapter_path(configured) == configured
