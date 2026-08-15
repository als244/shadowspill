from __future__ import annotations

import pytest

from qualification.performance.run import _manifest_with_overrides


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
