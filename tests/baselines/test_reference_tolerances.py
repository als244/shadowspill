"""The reference check holds the numerical gate's bounds, metrics and
relative rule, and tells a weight that is nothing but its steps by its
zero start."""

import pytest
import torch

from baselines.common import reference
from qualification.numerical import metrics, tolerances

BOUNDS = (
    "LOSS_RELATIVE_TOLERANCE",
    "LOSS_ABSOLUTE_TOLERANCE",
    "MINIMUM_COSINE",
    "MAXIMUM_RELATIVE_L2",
    "MAXIMUM_RELATIVE_L2_OPTIMIZER",
    "MINIMUM_SIGN_AGREEMENT",
    "STEP_QUANTUM_DISAGREEMENTS",
)
FIELDS = (
    "cosine",
    "relative_l2",
    "sign_agreement",
    "maximum_absolute_error",
    "reference_maximum_absolute",
)
RATE = 1e-3


def test_the_bounds_are_the_numerical_gates():
    for name in BOUNDS:
        assert getattr(reference, name) == getattr(tolerances, name), name


def pairs(seed):
    generator = torch.Generator().manual_seed(seed)
    scaled = torch.randn(257, generator=generator)
    noise = torch.randn(257, generator=generator)
    stepped = RATE * torch.sign(torch.randn(257, generator=generator))
    # One element whose sum rounded to exactly zero took no step.
    zeroed = stepped.clone()
    zeroed[3] = 0.0
    far = stepped.clone()
    far[5] += 3 * RATE
    return (
        (scaled, scaled + 1e-3 * noise),
        (scaled, scaled + 0.04 * noise),
        (scaled, -scaled),
        (scaled, scaled.clone()),
        (stepped, zeroed),
        (stepped, far),
    )


@pytest.mark.parametrize("seed", range(3))
def test_the_metrics_and_the_relative_rule_are_the_numerical_gates(seed):
    for expected, actual in pairs(seed):
        mine = reference.tensor_metrics(expected, actual)
        theirs = metrics.tensor_metrics(expected, actual)
        for field in FIELDS:
            assert getattr(mine, field) == getattr(theirs, field), field
        # Without a step-only weight, a weight takes the gate's bounds alone;
        # an accumulator takes them without the sign test, so it passes
        # wherever the gate's passes it.
        assert reference.meets_tolerance(
            mine, weight=True, step_size=RATE
        ) == tolerances.meets_tensor_tolerance(theirs, key="state/model/w")
        accumulator = reference.meets_tolerance(mine, weight=False, step_size=RATE)
        assert accumulator == (
            mine.cosine >= tolerances.MINIMUM_COSINE
            and mine.relative_l2 <= tolerances.MAXIMUM_RELATIVE_L2_OPTIMIZER
        )
        if tolerances.meets_tensor_tolerance(theirs, key="state/optimizer/w"):
            assert accumulator


def test_a_zero_started_weight_takes_the_absolute_bound():
    _, _, _, _, (stepped, zeroed), (_, far) = pairs(0)
    zero = reference.tensor_metrics(stepped, zeroed)
    assert reference.meets_tolerance(
        zero, weight=True, step_size=RATE, zero_started=True
    )
    assert not reference.meets_tolerance(zero, weight=True, step_size=RATE)
    assert not reference.meets_tolerance(
        zero, weight=False, step_size=RATE, zero_started=True
    )
    assert not reference.meets_tolerance(
        reference.tensor_metrics(stepped, far),
        weight=True,
        step_size=RATE,
        zero_started=True,
    )


def test_adamw_steps_a_growing_gradient_past_one_learning_rate():
    # Why a step-only weight is told by its start: two AdamW steps from zero
    # on a growing gradient move it further than two learning rates, past
    # the gate's line for such a weight.
    weight = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([weight], lr=RATE, weight_decay=0.0)
    for gradient in (1.0, 1.1):
        weight.grad = torch.full((1,), gradient)
        optimizer.step()
    assert abs(float(weight.detach())) > 2 * RATE


def test_an_accumulator_is_judged_without_its_signs():
    generator = torch.Generator().manual_seed(0)
    expected = torch.randn(64, generator=generator)
    expected[7] = 1e-6
    actual = expected.clone()
    actual[7] = -1e-6
    metric = reference.tensor_metrics(expected, actual)
    assert metric.sign_agreement < reference.MINIMUM_SIGN_AGREEMENT
    assert reference.meets_tolerance(metric, weight=False, step_size=RATE)
    assert not reference.meets_tolerance(metric, weight=True, step_size=RATE)
