from __future__ import annotations

import torch

from qualification.numerical.metrics import TensorMetrics, compare_states, state_digest
from qualification.numerical.run import (
    _failure_tensor_values,
    _meets_tensor_tolerance,
    _recomputation_savings_bytes,
    _transfer_pressure_gate_passed,
)
from shadowspill.ir import (
    RecomputationGroup,
    RecomputationOption,
    RecomputationSelection,
)


def test_state_metrics_are_path_specific_and_deterministic() -> None:
    reference = {
        "model": {"weight": torch.tensor([1.0, -2.0, 3.0])},
        "optimizer": {"step": torch.tensor(2)},
    }
    actual = {
        "model": {"weight": torch.tensor([1.0, -2.0, 3.01])},
        "optimizer": {"step": torch.tensor(2)},
    }
    metrics, failures = compare_states(reference, actual)
    assert not failures
    assert set(metrics) == {"state/model/weight"}
    assert metrics["state/model/weight"].cosine > 0.999
    assert state_digest(reference) == state_digest(reference)
    assert state_digest(reference) != state_digest(actual)


def test_state_metrics_reject_nonfloating_and_structural_differences() -> None:
    _metrics, failures = compare_states(
        {"value": torch.tensor([1, 2]), "kind": "x"},
        {"value": torch.tensor([1, 3]), "kind": "y"},
    )
    assert failures == (
        "state/kind: 'x' != 'y'",
        "state/value: integral tensor differs [1, 2] != [1, 3]",
    )


def test_failure_values_resolve_integer_optimizer_state_keys() -> None:
    reference = {"optimizer": {"state": {31: {"exp_avg": torch.tensor([1.0])}}}}
    actual = {"optimizer": {"state": {31: {"exp_avg": torch.tensor([2.0])}}}}

    values = _failure_tensor_values(
        ["state/optimizer/state/31/exp_avg"], reference, actual
    )

    assert values["state/optimizer/state/31/exp_avg"] == {
        "numel": 1,
        "truncated": False,
        "reference": [1.0],
        "actual": [2.0],
    }


def test_numerical_gate_uses_one_global_tensor_policy() -> None:
    assert _meets_tensor_tolerance(TensorMetrics(0.999, 0.025, 0.99, 1.0))
    assert not _meets_tensor_tolerance(TensorMetrics(0.998, 0.0, 1.0, 0.0))
    assert not _meets_tensor_tolerance(TensorMetrics(1.0, 0.026, 1.0, 0.0))
    assert not _meets_tensor_tolerance(TensorMetrics(1.0, 0.0, 0.98, 0.0))


def test_recomputation_diagnostics_count_only_retained_physical_savings() -> None:
    groups = (
        RecomputationGroup(
            "group_0",
            (
                RecomputationOption("save", (), ("a", "b")),
                RecomputationOption("same_size", (), ("a", "c")),
                RecomputationOption("recompute", (), ("a",)),
            ),
        ),
    )
    sizes = {"a": 64, "b": 32, "c": 32}

    assert _recomputation_savings_bytes(
        groups,
        (RecomputationSelection("group_0", "same_size"),),
        sizes,
    ) == (32, 0)
    assert _recomputation_savings_bytes(
        groups,
        (RecomputationSelection("group_0", "recompute"),),
        sizes,
    ) == (32, 32)


def test_recomputation_diagnostics_ignore_equal_footprints() -> None:
    groups = (
        RecomputationGroup(
            "group_0",
            (
                RecomputationOption("save", (), ("a",)),
                RecomputationOption("recompute", (), ("b",)),
            ),
        ),
    )
    assert _recomputation_savings_bytes(
        groups,
        (RecomputationSelection("group_0", "save"),),
        {"a": 64, "b": 64},
    ) == (0, 0)


def test_correctness_pressure_gate_requires_only_real_transfers() -> None:
    assert _transfer_pressure_gate_passed(
        required=True, evicted_bytes=1, fetched_bytes=1
    )
    assert not _transfer_pressure_gate_passed(
        required=True, evicted_bytes=1, fetched_bytes=0
    )
    assert _transfer_pressure_gate_passed(
        required=False, evicted_bytes=0, fetched_bytes=0
    )
