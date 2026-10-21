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


def test_a_zero_started_weight_is_allowed_its_stepped_dtype_rounding():
    # Two opposite full steps, the values held at bf16: they lie a little
    # more than two learning rates apart, by less than one unit in the last
    # place at their magnitude.
    expected = torch.full((8,), -3e-4).to(torch.bfloat16)
    actual = -expected
    distance = float(actual[0].float() - expected[0].float())
    rate = distance / 2 * (1 - 2**-10)
    metric = reference.tensor_metrics(expected, actual)
    assert metric.maximum_absolute_error > 2 * rate
    error = reference.step_error(expected, actual, torch.bfloat16)
    assert error <= 2 * rate
    assert reference.meets_tolerance(
        metric, weight=True, step_size=rate, zero_started=True, step_error=error
    )
    assert not reference.meets_tolerance(
        metric, weight=True, step_size=rate, zero_started=True
    )
    far = actual.clone()
    far[0] = torch.tensor(4 * rate).to(torch.bfloat16)
    assert reference.step_error(expected, far, torch.bfloat16) > 2 * rate
    # At float32 the allowance is negligible.
    assert reference.step_error(
        expected.float(), actual.float(), torch.float32
    ) == pytest.approx(metric.maximum_absolute_error, rel=1e-5)


def test_a_zero_started_weight_is_measured_in_the_references_own_steps():
    # AdamW stepping bf16 weights with bf16 moments: the first step of a
    # zero-started weight is one learning rate only to within the moments'
    # rounding, so two opposite steps can lie more than two learning rates
    # and an ulp apart. The reference's own largest step is the unit.
    rate = 3e-4
    gradient = torch.logspace(-6, 0, 4096).to(torch.bfloat16)

    def first_step(sign):
        weight = torch.zeros_like(gradient, requires_grad=True)
        optimizer = torch.optim.AdamW([weight], lr=rate, foreach=False)
        weight.grad = sign * gradient
        optimizer.step()
        return weight.detach()

    expected, actual = first_step(1.0), first_step(-1.0)
    metric = reference.tensor_metrics(expected, actual)
    error = reference.step_error(expected, actual, torch.bfloat16)
    step = max(rate, float(expected.float().abs().max()))
    assert step > rate
    assert error > 2 * rate
    assert reference.meets_tolerance(
        metric, weight=True, step_size=step, zero_started=True, step_error=error
    )
    assert not reference.meets_tolerance(
        metric, weight=True, step_size=rate, zero_started=True, step_error=error
    )


def test_a_routed_expert_factor_takes_the_wider_accumulator_bound():
    generator = torch.Generator().manual_seed(0)
    expected = torch.randn(4096, generator=generator)
    noise = torch.randn(4096, generator=generator)
    actual = expected + 0.07 * noise * expected.norm() / noise.norm()
    metric = reference.tensor_metrics(expected, actual)
    assert 0.995 < metric.cosine < reference.MINIMUM_COSINE
    assert reference.MAXIMUM_RELATIVE_L2_OPTIMIZER < metric.relative_l2 < 0.10
    assert reference.meets_tolerance(
        metric, weight=False, step_size=RATE, routed_factor=True
    )
    assert not reference.meets_tolerance(metric, weight=False, step_size=RATE)
    # A routed factor's master keeps the weight bounds.
    assert not reference.meets_tolerance(
        metric, weight=True, step_size=RATE, routed_factor=True
    )
