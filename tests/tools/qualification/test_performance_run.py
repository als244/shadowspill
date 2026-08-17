from __future__ import annotations

import pytest

from tools.qualification.performance import (
    _manifest_with_overrides,
    _planning_spill_budget,
)


def test_spill_budget_override_preserves_the_canonical_manifest() -> None:
    original = _manifest_with_overrides(
        "llama3", "mlops", spill_budget_gib=None
    )
    overridden = _manifest_with_overrides(
        "llama3", "mlops", spill_budget_gib=72
    )

    assert overridden is not original
    assert overridden.spill_budget_bytes == 72 << 30
    assert original.spill_budget_bytes != overridden.spill_budget_bytes
    assert overridden.identity == original.identity


def test_spill_budget_override_rejects_nonpositive_capacity() -> None:
    with pytest.raises(ValueError, match="spill-budget-gib must be positive"):
        _manifest_with_overrides("llama3", "mlops", spill_budget_gib=0)


def test_planning_spill_budget_may_be_smaller_than_runtime_pool() -> None:
    manifest = _manifest_with_overrides(
        "qwen35", "mlops", spill_budget_gib=112
    )

    assert _planning_spill_budget(
        manifest, planning_spill_budget_gib=100
    ) == 100 << 30


def test_planning_spill_budget_cannot_exceed_runtime_pool() -> None:
    manifest = _manifest_with_overrides(
        "qwen35", "mlops", spill_budget_gib=96
    )

    with pytest.raises(ValueError, match="exceeds the configured runtime spill pool"):
        _planning_spill_budget(manifest, planning_spill_budget_gib=100)
