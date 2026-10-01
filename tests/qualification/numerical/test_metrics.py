from __future__ import annotations

import torch

from qualification.numerical.measures import (
    failure_tensor_values,
    recomputation_savings_bytes,
)
from qualification.numerical.metrics import (
    TensorMetrics,
    compare_states,
    state_digest,
)
from qualification.numerical.tolerances import meets_tensor_tolerance
from qualification.numerical.verdict import transfer_pressure_gate_passed
from shadowspill.ir import (
    TaskAlternativeChoice,
    TaskAlternativeGroup,
    TaskAlternativeOption,
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
    metrics, failures, structure = compare_states(reference, actual)
    assert not failures
    assert not structure
    assert set(metrics) == {"state/model/weight"}
    assert metrics["state/model/weight"].cosine > 0.999
    assert state_digest(reference) == state_digest(reference)
    assert state_digest(reference) != state_digest(actual)


def test_state_metrics_separate_value_differences_from_structural_ones() -> None:
    """A different value and a different shape are different findings.

    Values that disagree are evidence about arithmetic. Shapes that disagree
    mean the two states are not answers to the same question at all, so any
    metric taken across them describes nothing -- which is why they are
    returned apart rather than in one list of failures.
    """

    _metrics, failures, structure = compare_states(
        {"value": torch.tensor([1, 2]), "kind": "x"},
        {"value": torch.tensor([1, 3]), "kind": "y"},
    )
    assert failures == (
        "state/kind: 'x' != 'y'",
        "state/value: integral tensor differs [1, 2] != [1, 3]",
    )
    assert not structure

    _metrics, failures, structure = compare_states(
        {"weight": torch.zeros(2048, 2048), "state": {"a": torch.zeros(4)}},
        {"weight": torch.zeros(2048), "state": {"b": torch.zeros(4)}},
    )
    assert not failures
    assert structure == (
        "state/state: mapping keys differ",
        "state/weight: geometry torch.Size([2048, 2048])/torch.float32 != "
        "torch.Size([2048])/torch.float32",
    )


def test_failure_values_resolve_integer_optimizer_state_keys() -> None:
    reference = {"optimizer": {"state": {31: {"exp_avg": torch.tensor([1.0])}}}}
    actual = {"optimizer": {"state": {31: {"exp_avg": torch.tensor([2.0])}}}}

    values = failure_tensor_values(
        ["state/optimizer/state/31/exp_avg"], reference, actual
    )

    assert values["state/optimizer/state/31/exp_avg"] == {
        "numel": 1,
        "truncated": False,
        "reference": [1.0],
        "actual": [2.0],
    }


def test_numerical_gate_uses_one_global_tensor_policy() -> None:
    assert meets_tensor_tolerance(TensorMetrics(0.999, 0.025, 0.99, 1.0))
    assert not meets_tensor_tolerance(TensorMetrics(0.998, 0.0, 1.0, 0.0))
    # an optimizer moment may drift twice as far as a weight: it is an
    # accumulator whose reduction order follows the plan
    moment = TensorMetrics(0.9995, 0.04, 0.995, 0.0)
    assert meets_tensor_tolerance(moment, key="state/optimizer/state/0/exp_avg_sq")
    assert not meets_tensor_tolerance(moment, key="state/model/layers.0.weight")
    assert not meets_tensor_tolerance(moment)
    assert not meets_tensor_tolerance(TensorMetrics(1.0, 0.026, 1.0, 0.0))
    assert not meets_tensor_tolerance(TensorMetrics(1.0, 0.0, 0.98, 0.0))


def test_a_weight_made_only_of_optimizer_steps_is_judged_absolutely() -> None:
    """A zero-started bias is a few learning rates from zero after a few
    steps, so a relative bound reads one step's sign disagreement as a large
    error; the absolute bound of two learning rates admits exactly that."""

    lr, steps = 3e-4, 5
    bias = TensorMetrics(
        cosine=0.996,
        relative_l2=0.09,
        sign_agreement=1.0,
        maximum_absolute_error=2.9e-4,
        reference_maximum_absolute=1.2e-3,
    )
    key = "state/model/blocks.5.mixer.dt_bias"
    assert meets_tensor_tolerance(bias, key=key, step_size=lr, steps=steps)
    # Without the run's step size and count there is no absolute bound.
    assert not meets_tensor_tolerance(bias, key=key)
    # A weight with a scale of its own keeps the relative bound.
    scaled = TensorMetrics(0.996, 0.09, 1.0, 2.9e-4, reference_maximum_absolute=0.05)
    assert not meets_tensor_tolerance(scaled, key=key, step_size=lr, steps=steps)
    # More than one step's disagreement fails.
    drifted = TensorMetrics(0.996, 0.09, 1.0, 7e-4, reference_maximum_absolute=1.2e-3)
    assert not meets_tensor_tolerance(drifted, key=key, step_size=lr, steps=steps)
    # An optimizer moment is judged by its own relative bound alone.
    moment_key = "state/optimizer/state/0/exp_avg"
    assert not meets_tensor_tolerance(bias, key=moment_key, step_size=lr, steps=steps)


def test_recomputation_diagnostics_count_only_retained_physical_savings() -> None:
    groups = (
        TaskAlternativeGroup(
            "group_0",
            (
                TaskAlternativeOption("save", (), ("a", "b")),
                TaskAlternativeOption("same_size", (), ("a", "c")),
                TaskAlternativeOption("recompute", (), ("a",)),
            ),
        ),
    )
    sizes = {"a": 64, "b": 32, "c": 32}

    assert recomputation_savings_bytes(
        groups,
        (TaskAlternativeChoice("group_0", "same_size"),),
        sizes,
    ) == (32, 0)
    assert recomputation_savings_bytes(
        groups,
        (TaskAlternativeChoice("group_0", "recompute"),),
        sizes,
    ) == (32, 32)


def test_recomputation_diagnostics_ignore_equal_footprints() -> None:
    groups = (
        TaskAlternativeGroup(
            "group_0",
            (
                TaskAlternativeOption("save", (), ("a",)),
                TaskAlternativeOption("recompute", (), ("b",)),
            ),
        ),
    )
    assert recomputation_savings_bytes(
        groups,
        (TaskAlternativeChoice("group_0", "save"),),
        {"a": 64, "b": 64},
    ) == (0, 0)


def test_correctness_pressure_gate_requires_only_real_transfers() -> None:
    assert transfer_pressure_gate_passed(
        required=True, evicted_bytes=1, fetched_bytes=1
    )
    assert not transfer_pressure_gate_passed(
        required=True, evicted_bytes=1, fetched_bytes=0
    )
    assert transfer_pressure_gate_passed(
        required=False, evicted_bytes=0, fetched_bytes=0
    )
