from __future__ import annotations

import torch

from qualification.numerical.metrics import TensorMetrics, compare_states, state_digest
from qualification.numerical.run import _meets_tensor_tolerance


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


def test_numerical_gate_uses_one_global_tensor_policy() -> None:
    assert _meets_tensor_tolerance(TensorMetrics(0.999, 0.025, 0.99, 1.0))
    assert not _meets_tensor_tolerance(TensorMetrics(0.998, 0.0, 1.0, 0.0))
    assert not _meets_tensor_tolerance(TensorMetrics(1.0, 0.026, 1.0, 0.0))
    assert not _meets_tensor_tolerance(TensorMetrics(1.0, 0.0, 0.98, 0.0))
