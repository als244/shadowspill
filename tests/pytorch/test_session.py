from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from shadowspill.pytorch import PlanningError, TensorSpec
from shadowspill.pytorch.materialization import representative_cpu_inputs
from shadowspill.pytorch.profiling import TaskMeasurement
from shadowspill.pytorch.runtime import _adapter_path
from shadowspill.pytorch.session import (
    _PhaseTimer,
    _simulation_capacity,
    _spill_pool_estimate,
    _validate_forward_request,
    _workspace_reserve,
)


def test_phase_timer_attributes_compilation_and_profiling_without_overlap() -> None:
    timer = _PhaseTimer(verbose=False)
    timer.values = [
        ("capture_lowering", 11),
        ("compiler_manifest", 30),
        ("structural_profiling", 100),
        ("compilation", 20),
        ("program_lowering", 13),
    ]
    profiler = SimpleNamespace(
        compilation_wall_time_ns=40,
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


def test_forward_request_and_admission_helpers_reject_invalid_values() -> None:
    model = nn.Linear(2, 2)
    with pytest.raises(TypeError, match="model"):
        _validate_forward_request(object(), [], 1 << 30, 1 << 30)  # type: ignore[arg-type]
    with pytest.raises(PlanningError, match="example_inputs"):
        _validate_forward_request(model, object(), 1 << 30, 1 << 30)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="execution_budget"):
        _validate_forward_request(model, [], True, 1 << 30)
    with pytest.raises(TypeError, match="spill_budget"):
        _validate_forward_request(model, [], 1 << 30, True)
    with pytest.raises(PlanningError, match="positive"):
        _validate_forward_request(model, [], 0, 1 << 30)
    with pytest.raises(PlanningError, match="spill_budget"):
        _validate_forward_request(model, [], 1 << 30, 0)
    with pytest.raises(PlanningError, match="CPU resident"):
        _validate_forward_request(nn.Linear(2, 2, device="meta"), [], 1 << 30, 1 << 30)
    with pytest.raises(PlanningError, match="spill-pool budget"):
        _spill_pool_estimate(model, [torch.ones(2)], 1)


def test_workspace_and_capacity_helpers_are_explicit() -> None:
    measurement = TaskMeasurement(1, 3, 100, (100,), (1,), "test")
    assert _workspace_reserve((measurement,)) == 512 << 20
    assert _simulation_capacity(1 << 30, 512 << 20, (measurement,)) == (
        (512 << 20) + 100
    )
    assert (
        _simulation_capacity(
            1 << 30,
            512 << 20,
            (measurement,),
            fixed_slab_bytes=32,
        )
        == (512 << 20) + 68
    )
    with pytest.raises(PlanningError, match="smaller"):
        _simulation_capacity(1, 2, ())


def test_representatives_and_adapter_path_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    values = representative_cpu_inputs(
        [TensorSpec((2, 3), torch.float32), torch.empty(4, device="meta"), "x"]
    )
    assert values[0].device.type == "cpu" and values[0].shape == (2, 3)
    assert values[1].device.type == "cpu" and values[1].shape == (4,)
    assert values[2] == "x"

    configured = (
        _adapter_path(None) if "SHADOWSPILL_PYTORCH_LIBRARY" in os.environ else None
    )
    missing = tmp_path / "missing.so"
    monkeypatch.setenv("SHADOWSPILL_PYTORCH_LIBRARY", str(missing))
    with pytest.raises(RuntimeError, match="not found"):
        _adapter_path(None)
    if configured is not None:
        monkeypatch.setenv("SHADOWSPILL_PYTORCH_LIBRARY", str(configured))
