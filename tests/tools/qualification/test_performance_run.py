from __future__ import annotations

import pytest

from tools.qualification.performance.manifest import (
    _manifest_with_overrides,
    _planning_spill_budget,
)


def test_spill_budget_override_preserves_the_canonical_manifest() -> None:
    original = _manifest_with_overrides("llama3", "mlops", spill_budget_gib=None)
    overridden = _manifest_with_overrides("llama3", "mlops", spill_budget_gib=72)

    assert overridden is not original
    assert overridden.spill_budget_bytes == 72 << 30
    assert original.spill_budget_bytes != overridden.spill_budget_bytes
    assert overridden.identity == original.identity


def test_spill_budget_override_rejects_nonpositive_capacity() -> None:
    with pytest.raises(ValueError, match="spill-budget-gib must be positive"):
        _manifest_with_overrides("llama3", "mlops", spill_budget_gib=0)


def test_planning_spill_budget_may_be_smaller_than_runtime_pool() -> None:
    manifest = _manifest_with_overrides("qwen35", "mlops", spill_budget_gib=112)

    assert _planning_spill_budget(manifest, planning_spill_budget_gib=100) == 100 << 30


def test_planning_spill_budget_cannot_exceed_runtime_pool() -> None:
    manifest = _manifest_with_overrides("qwen35", "mlops", spill_budget_gib=96)

    with pytest.raises(ValueError, match="exceeds the configured runtime spill pool"):
        _planning_spill_budget(manifest, planning_spill_budget_gib=100)


def test_execution_budget_and_model_precision_override_the_recorded_manifest() -> None:
    default = _manifest_with_overrides("llama3", "mlops", spill_budget_gib=None)
    assert default.device_physical_capacity_bytes == 16 << 30
    changed = _manifest_with_overrides(
        "llama3",
        "mlops",
        spill_budget_gib=None,
        execution_budget_gib=8,
        model_dtype="float16",
    )
    assert changed.device_physical_capacity_bytes == 8 << 30
    assert changed.as_dict()["model_dtype"] == "float16"
    assert changed.spill_budget_bytes == default.spill_budget_bytes
    with pytest.raises(ValueError, match="execution-budget-gib must be positive"):
        _manifest_with_overrides(
            "llama3",
            "mlops",
            spill_budget_gib=None,
            execution_budget_gib=0,
        )


def test_performance_defaults_and_all_dtype_overrides_reach_the_case() -> None:
    from pathlib import Path

    from tools.qualification.performance_matrix import _cell_command, _parser

    default = _manifest_with_overrides("llama3", "mlops", spill_budget_gib=None)
    assert default.spill_budget_bytes == 112 << 30
    assert default.dtypes.as_dict() == {
        "model_dtype": "bfloat16",
        "master_dtype": "none",
        "grad_dtype": "bfloat16",
        "opt_state_dtype": "bfloat16",
    }
    changed = _manifest_with_overrides(
        "llama3",
        "mlops",
        spill_budget_gib=None,
        model_dtype="float32",
        master_dtype="bfloat16",
        grad_dtype="float32",
        opt_state_dtype="bfloat16",
    )
    command = _cell_command(
        changed, _parser().parse_args([]), Path("out"), Path("out/cell.json"), {}
    )
    for field, value in changed.dtypes.as_dict().items():
        assert command[command.index("--" + field.replace("_", "-")) + 1] == value
