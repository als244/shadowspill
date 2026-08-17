from __future__ import annotations

import pytest

from shadowspill.pytorch import AdmissionError
from shadowspill.pytorch.runtime_adapter.runtime import (
    MemoryPool,
    _resolve_dynamic_scratch_reserve,
    _resolve_execution_budget,
)


def test_execution_budget_accepts_runtime_physical_cap_without_double_charge() -> None:
    physical = 16 << 30
    derived = physical - (1280 << 20) - (256 << 20)
    pool = MemoryPool("execution", 0, "device", derived, physical, 0)

    assert _resolve_execution_budget(None, pool) == derived
    assert _resolve_execution_budget(physical, pool) == derived
    assert _resolve_execution_budget(10 << 30, pool) == 10 << 30
    with pytest.raises(AdmissionError, match="falls between"):
        _resolve_execution_budget(derived + 1, pool)
    with pytest.raises(AdmissionError, match="physical capacity"):
        _resolve_execution_budget(physical + 1, pool)


def test_dynamic_scratch_reserve_is_an_optional_bounded_minimum() -> None:
    budget = 16 << 30
    assert _resolve_dynamic_scratch_reserve(None, execution_budget=budget) == 0
    assert _resolve_dynamic_scratch_reserve(1 << 30, execution_budget=budget) == (
        1 << 30
    )
    with pytest.raises(TypeError, match="integer byte count"):
        _resolve_dynamic_scratch_reserve(True, execution_budget=budget)
    with pytest.raises(AdmissionError, match="non-negative"):
        _resolve_dynamic_scratch_reserve(-1, execution_budget=budget)
    with pytest.raises(AdmissionError, match="exceeds"):
        _resolve_dynamic_scratch_reserve(budget + 1, execution_budget=budget)
