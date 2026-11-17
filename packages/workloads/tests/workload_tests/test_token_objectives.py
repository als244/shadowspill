"""Independent scalar/gradient oracles for the supplied summed token losses."""

import math

import numpy as np
import pytest
import torch
from mlops.objectives import clipped_policy, cross_entropy

from shadowspill.pytorch.capture.objective import capture_training_objective


def oracle(current, valid, weights, old, advantages, clip, reference=None, beta=0.0):
    loss, gradients = 0.0, np.zeros_like(current, dtype=np.float64)
    for i, active in enumerate(valid):
        if not active:
            continue
        ratio = math.exp(float(current[i]) - float(old[i]))
        advantage = float(advantages[i])
        clipped = min(1 + clip, max(1 - clip, ratio))
        term = -min(ratio * advantage, clipped * advantage)
        derivative = (
            -ratio * advantage
            if (advantage > 0 and ratio <= 1 + clip)
            or (advantage < 0 and ratio >= 1 - clip)
            else 0.0
        )
        if reference is not None and beta:
            difference = float(reference[i]) - float(current[i])
            term += beta * (math.expm1(difference) - difference)
            derivative += beta * (1 - math.exp(difference))
        loss += float(weights[i]) * term
        gradients[i] = float(weights[i]) * derivative
    return loss, gradients


def inputs():
    # Both signs of advantage on each side of the clipping interval; zero
    # weights/advantages retain validity. Invalid values must not enter math.
    ratios = [0.5, 0.9, 1.1, 1.6, 0.5, 0.9, 1.1, 1.6, 1.0, 1.3, 2.0, 1.0]
    current = torch.tensor(np.log(ratios) - 2, dtype=torch.float64, requires_grad=True)
    old = torch.full_like(current, -2)
    advantages = torch.tensor([1.2] * 4 + [-0.7] * 4 + [0.0, 0.8, float("nan"), 1.0])
    weights = torch.tensor([1.0, -0.3, 0.4, 0.8, 1.2, 0.7, -0.6, 1.0, 1.1, 0, 1, 0.2])
    valid = torch.tensor([True] * 10 + [False, True])
    reference = current.detach().float() + torch.tensor(
        [0.2, -0.3, 0, 0.01, 0.02, -0.5, 0.1, 0.4, 0, 0.1, float("nan"), 1e-5]
    )
    with torch.no_grad():
        current[10] = old[10] = weights[10] = float("nan")
    return current, valid, weights, old, advantages, reference


@pytest.mark.parametrize("clip", [0.0, 0.2, 0.75])
@pytest.mark.parametrize("beta", [None, 0.0, 0.13])
def test_policy_sum_and_gradients_match_independent_scalar_oracle(clip, beta):
    current, valid, weights, old, advantages, reference = inputs()
    options = (
        {}
        if beta is None
        else {
            "reference_logprobs": reference,
            "reference_beta": torch.tensor(beta),
        }
    )
    loss = clipped_policy(
        current, valid, weights, old, advantages, torch.tensor(clip), **options
    )
    expected, gradient = oracle(
        current.detach().numpy(),
        valid.numpy(),
        weights.numpy(),
        old.numpy(),
        advantages.numpy(),
        clip,
        reference.numpy(),
        beta or 0.0,
    )
    assert loss.dtype == torch.float32
    assert loss.item() == pytest.approx(expected, abs=3e-6, rel=3e-6)
    (actual,) = torch.autograd.grad(loss, current)
    np.testing.assert_allclose(actual.numpy(), gradient, atol=3e-6, rtol=3e-6)
    assert actual[10] == 0


def test_cross_entropy_is_a_signed_weighted_sum_with_masked_nan_neutralization():
    current, valid, weights, *_ = inputs()
    loss = cross_entropy(current, valid, weights)
    active = valid.numpy()
    expected = -(current.detach().numpy()[active] * weights.numpy()[active]).sum()
    assert loss.dtype == torch.float32
    assert loss.item() == pytest.approx(expected, abs=2e-6)
    (gradient,) = torch.autograd.grad(loss, current)
    np.testing.assert_array_equal(
        gradient.numpy(), np.where(active, -weights.numpy(), 0)
    )


def test_empty_and_all_masked_losses_remain_differentiable_zero():
    for size in (0, 4):
        current = torch.full((size,), float("nan"), requires_grad=True)
        invalid = torch.zeros(size, dtype=torch.bool)
        other = torch.full((size,), float("nan"))
        for loss in (
            cross_entropy(current, invalid, other),
            clipped_policy(
                current,
                invalid,
                other,
                other,
                other,
                torch.tensor(0.2),
                reference_logprobs=other,
                reference_beta=torch.tensor(0.1),
            ),
        ):
            assert loss.item() == 0
            (gradient,) = torch.autograd.grad(loss, current)
            assert torch.isfinite(gradient).all() and torch.count_nonzero(gradient) == 0


def test_extreme_ratios_preserve_clipping_and_expose_unbounded_overflow():
    current = torch.tensor([100.0, -100.0], requires_grad=True)
    valid, weights, old = torch.ones(2, dtype=torch.bool), torch.ones(2), torch.zeros(2)
    loss = clipped_policy(
        current, valid, weights, old, torch.tensor([1.0, -1.0]), torch.tensor(0.2)
    )
    assert loss.item() == pytest.approx(-0.4, abs=2e-7)
    (gradient,) = torch.autograd.grad(loss, current)
    assert torch.equal(gradient, torch.zeros(2))
    unbounded = clipped_policy(
        current, valid, weights, old, torch.tensor([-1.0, 1.0]), torch.tensor(0.2)
    )
    assert torch.isinf(unbounded)


def test_zero_reference_beta_avoids_unused_overflow_without_detaching_active_penalty():
    current = torch.tensor([-1.0], requires_grad=True)
    ones, valid = torch.ones(1), torch.ones(1, dtype=torch.bool)
    for beta in (0.0, 0.1):
        loss = clipped_policy(
            current,
            valid,
            ones,
            current.detach(),
            ones,
            torch.tensor(0.2),
            reference_logprobs=torch.tensor([1000.0]),
            reference_beta=torch.tensor(beta),
        )
        if beta == 0:
            assert loss.item() == -1
            assert torch.autograd.grad(loss, current)[0].item() == -1
        else:
            assert torch.isinf(loss)


def test_unequal_split_contributions_match_whole_loss_and_gradient():
    current, valid, weights, old, advantages, reference = inputs()

    def part(selection):
        return clipped_policy(
            current[selection],
            valid[selection],
            weights[selection],
            old[selection],
            advantages[selection],
            torch.tensor(0.2),
            reference_logprobs=reference[selection],
            reference_beta=torch.tensor(0.07),
        )

    whole = part(slice(None))
    split = part(slice(0, 3)) + part(slice(3, 3)) + part(slice(3, None))
    torch.testing.assert_close(whole, split)
    torch.testing.assert_close(
        torch.autograd.grad(whole, current)[0], torch.autograd.grad(split, current)[0]
    )


def test_captured_policy_reads_changed_controls_without_recapture():
    current, valid, weights, old, advantages, reference = inputs()
    model = torch.nn.Linear(1, len(current), bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.nan_to_num(current[:, None]))
    fields = dict(
        valid=valid,
        weights=weights,
        old=old,
        advantages=advantages,
        reference=reference,
    )

    def objective(model, inputs, loss_inputs, controls):
        return clipped_policy(
            model(inputs["x"]),
            loss_inputs["valid"],
            loss_inputs["weights"],
            loss_inputs["old"],
            loss_inputs["advantages"],
            controls["clip"],
            reference_logprobs=loss_inputs["reference"],
            reference_beta=controls["beta"],
        )

    x = {"x": torch.ones(1)}
    capture = capture_training_objective(
        model,
        objective,
        [x, fields, {"clip": torch.tensor(0.2), "beta": torch.tensor(0.1)}],
    )
    graph = capture.exported.exported_program.module()
    # Stay away from nondifferentiable clipping boundaries: rounding the model
    # to FP32 can move a ratio infinitesimally across an exact decimal threshold.
    for clip, beta in ((0.2, 0.0), (0.17, 0.2), (0.63, 0.03)):
        (loss,) = graph(
            x, fields, {"clip": torch.tensor(clip), "beta": torch.tensor(beta)}
        )
        expected, gradient = oracle(
            current.detach().numpy(),
            valid.numpy(),
            weights.numpy(),
            old.numpy(),
            advantages.numpy(),
            clip,
            reference.numpy(),
            beta,
        )
        assert loss.item() == pytest.approx(expected, abs=3e-6)
        (actual,) = torch.autograd.grad(loss, model.weight)
        np.testing.assert_allclose(
            actual.detach().numpy()[:, 0], gradient, atol=3e-6, rtol=3e-6
        )
