"""A supplied model uses the same registry and CPU capture as external models."""

import torch

from shadowspill.pytorch.capture.objective import capture_training_objective
from shadowspill.training.definitions import Registry
from shadowspill_workloads.definitions import register


def test_supplied_regression_definition_uses_existing_model_and_capture():
    registry = Registry()
    register(registry)
    description = registry.resolve_model("shadowspill/regression@1")
    objective = registry.resolve_objective(
        "shadowspill/weighted_squared_error@1", model=description
    )
    model = description.create()
    inputs = {"value": torch.randn(3, 16)}
    loss_inputs = {
        "targets": torch.randn(3, 8),
        "valid": torch.tensor([True, True, False]),
        "weights": torch.tensor([1.0, 0.0, 0.0]),
    }
    captured = capture_training_objective(
        model, objective.computation(), [inputs, loss_inputs, {}]
    )
    actual, loss_sum, count = captured.exported.exported_program.module()(
        inputs, loss_inputs, {}
    )
    expected = (model(**inputs)[0] - loss_inputs["targets"][0]).square().sum()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(loss_sum, actual.detach())
    assert count.item() == 2
    assert description.definition.mutable_buffers == ("updates",)
    assert all(
        torch.isfinite(grad).all()
        for grad in torch.autograd.grad(actual, tuple(model.parameters()))
    )
