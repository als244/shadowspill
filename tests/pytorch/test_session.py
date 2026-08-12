from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from shadowspill.pytorch import PlanningError, TensorSpec
from shadowspill.pytorch._allocator import installed_allocator
from shadowspill.pytorch.materialization import representative_cpu_inputs
from shadowspill.pytorch.profiling import TaskMeasurement
from shadowspill.pytorch.session import (
    _adapter_path,
    _ensure_allocator,
    _host_arena_estimate,
    _nonnegative_environment_integer,
    _positive_environment_integer,
    _simulation_capacity,
    _validate_forward_request,
    _workspace_reserve,
)


def test_forward_request_and_admission_helpers_reject_invalid_values() -> None:
    model = nn.Linear(2, 2)
    with pytest.raises(TypeError, match="model"):
        _validate_forward_request(object(), [], 1 << 30, 1 << 30)  # type: ignore[arg-type]
    with pytest.raises(PlanningError, match="example_inputs"):
        _validate_forward_request(model, object(), 1 << 30, 1 << 30)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="device_budget"):
        _validate_forward_request(model, [], True, 1 << 30)
    with pytest.raises(TypeError, match="host_budget"):
        _validate_forward_request(model, [], 1 << 30, True)
    with pytest.raises(PlanningError, match="headroom"):
        _validate_forward_request(model, [], 512 << 20, 1 << 30)
    with pytest.raises(PlanningError, match="host_budget"):
        _validate_forward_request(model, [], 1 << 30, 0)
    with pytest.raises(PlanningError, match="CPU resident"):
        _validate_forward_request(nn.Linear(2, 2, device="meta"), [], 1 << 30, 1 << 30)
    with pytest.raises(PlanningError, match="spill-pool budget"):
        _host_arena_estimate(model, [torch.ones(2)], 1)


def test_workspace_and_environment_helpers_are_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    monkeypatch.delenv("SHADOWSPILL_TEST_INTEGER", raising=False)
    assert _nonnegative_environment_integer("SHADOWSPILL_TEST_INTEGER", 7) == 7
    monkeypatch.setenv("SHADOWSPILL_TEST_INTEGER", "11")
    assert _positive_environment_integer("SHADOWSPILL_TEST_INTEGER", 7) == 11
    monkeypatch.setenv("SHADOWSPILL_TEST_INTEGER", "bad")
    with pytest.raises(PlanningError, match="integer"):
        _nonnegative_environment_integer("SHADOWSPILL_TEST_INTEGER", 7)
    monkeypatch.setenv("SHADOWSPILL_TEST_INTEGER", "-1")
    with pytest.raises(PlanningError, match="non-negative"):
        _nonnegative_environment_integer("SHADOWSPILL_TEST_INTEGER", 7)
    monkeypatch.setenv("SHADOWSPILL_TEST_INTEGER", "0")
    with pytest.raises(PlanningError, match="positive"):
        _positive_environment_integer("SHADOWSPILL_TEST_INTEGER", 7)


def test_representatives_and_process_allocator_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    values = representative_cpu_inputs(
        [TensorSpec((2, 3), torch.float32), torch.empty(4, device="meta"), "x"]
    )
    assert values[0].device.type == "cpu" and values[0].shape == (2, 3)
    assert values[1].device.type == "cpu" and values[1].shape == (4,)
    assert values[2] == "x"

    configured = (
        _adapter_path() if "SHADOWSPILL_PYTORCH_LIBRARY" in os.environ else None
    )
    current = installed_allocator()
    if current is not None:
        assert (
            _ensure_allocator(
                device_budget=int(current.admission.device_budget_bytes),
                host_arena=1,
                device_ordinal=int(current.admission.device_ordinal),
            )
            is current
        )
        with pytest.raises(PlanningError, match="incompatible"):
            _ensure_allocator(
                device_budget=int(current.admission.device_budget_bytes) + 1,
                host_arena=1,
                device_ordinal=int(current.admission.device_ordinal),
            )
    missing = tmp_path / "missing.so"
    monkeypatch.setenv("SHADOWSPILL_PYTORCH_LIBRARY", str(missing))
    with pytest.raises(PlanningError, match="not found"):
        _adapter_path()
    if configured is not None:
        monkeypatch.setenv("SHADOWSPILL_PYTORCH_LIBRARY", str(configured))
