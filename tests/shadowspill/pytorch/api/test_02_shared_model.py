"""A training step and a forward pass planned over one imported model.

Evaluating while training is the reason: the forward must see every update
the step makes, in either planning order and either closing order, and the
model must come back to host views when the last plan over it closes.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from shadowspill.pytorch import import_model_state, plan_forward, plan_step

from ..runtime_test_support import public_test_runtime
from .test_01_public_training import _require_adapter


def _network() -> nn.Module:
    torch.manual_seed(7)
    return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))


def _objective(model: nn.Module, value: torch.Tensor, target: torch.Tensor):
    return torch.nn.functional.mse_loss(model(value), target)


def _batch(seed: int) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    value = torch.randn(3, 8, generator=generator)
    return [value, torch.randn(3, 4, generator=generator)]


def _plan_training(model: nn.Module, runtime: object):
    return plan_step(
        model,
        objective=_objective,
        optimizer=torch.optim.AdamW,
        optimizer_state_init=lambda name, tensor, parameter: tensor.zero_(),
        hyperparams=("lr",),
        example_inputs=[_batch(0)],
        runtime=runtime,
        execution="execution",
        spill="spill",
    )


def _plan_forward(model: nn.Module, runtime: object):
    return plan_forward(
        model,
        example_inputs=[_batch(0)[0]],
        runtime=runtime,
        execution="execution",
        spill="spill",
    )


def _forward_matches_the_trained_weights(training, forward) -> None:
    value = _batch(99)[0]
    reference = _network()
    reference.load_state_dict(training.state_dict()["model"])
    torch.testing.assert_close(forward([value]).cpu(), reference(value).detach())


@pytest.mark.cuda
@pytest.mark.fresh_process
def test_a_forward_planned_after_training_sees_every_update() -> None:
    _require_adapter()
    runtime = public_test_runtime()
    model = import_model_state(_network(), runtime=runtime, pool="spill")
    training = _plan_training(model, runtime)
    forward = _plan_forward(model, runtime)
    for step in range(3):
        training([_batch(10 + step)], hyperparams={"lr": 1e-2})
        _forward_matches_the_trained_weights(training, forward)
    forward.close()
    # The step still runs after the forward goes.
    training([_batch(20)], hyperparams={"lr": 1e-2})
    training.close()
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())


@pytest.mark.cuda
@pytest.mark.fresh_process
def test_a_training_step_planned_after_a_forward_shares_its_model() -> None:
    _require_adapter()
    runtime = public_test_runtime()
    model = import_model_state(_network(), runtime=runtime, pool="spill")
    forward = _plan_forward(model, runtime)
    before = forward([_batch(99)[0]]).cpu()
    training = _plan_training(model, runtime)
    training([_batch(10)], hyperparams={"lr": 1e-2})
    _forward_matches_the_trained_weights(training, forward)
    assert not torch.equal(forward([_batch(99)[0]]).cpu(), before)
    training.close()
    # The forward still runs after the training step goes.
    forward([_batch(99)[0]])
    forward.close()
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())


@pytest.mark.cuda
@pytest.mark.fresh_process
def test_state_a_plan_imported_for_itself_is_not_shared() -> None:
    _require_adapter()
    runtime = public_test_runtime()
    model = _network()  # not imported: the training plan imports it and owns it
    training = _plan_training(model, runtime)
    with pytest.raises(RuntimeError, match="import_model_state"):
        _plan_forward(model, runtime)
    training.close()
