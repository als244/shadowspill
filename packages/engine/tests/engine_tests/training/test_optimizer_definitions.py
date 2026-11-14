"""Optimizer definitions declare controls on the existing captured update."""

from dataclasses import dataclass, replace

import pytest
import torch

from engine_tests.fixtures.external_definition import SOURCE
from shadowspill.pytorch.optimizer import capture_optimizer
from shadowspill.training.definitions import OptimizerDefinition, Registry
from shadowspill_client.protocol.controls import ControlSchema, ScalarControl


@dataclass(frozen=True)
class SGDConfig:
    momentum: float = 0.0


def create(model, config, controls):
    return torch.optim.SGD(
        model.parameters(), lr=controls["lr"], momentum=config.momentum, foreach=False
    )


def definition():
    return OptimizerDefinition(
        "external/sgd@1",
        create,
        ControlSchema({"lr": ScalarControl(0.01, minimum=0)}),
        {"lr": ("optimizer_group.0.lr",)},
        config=SGDConfig,
    )


def test_public_optimizer_factory_uses_existing_capture_and_validates_real_controls():
    registry = Registry()
    spec = definition()
    registry.add_optimizer(spec, source=SOURCE)
    resolved = registry.resolve_optimizer(spec.id, defaults={"lr": 0.03})
    assert resolved.config == {"momentum": 0.0}
    assert resolved.defaults["lr"] == 0.03
    model = torch.nn.Linear(2, 1, bias=False)
    model.weight.grad = torch.ones_like(model.weight)
    before = model.weight.detach().clone()
    optimizer = resolved.create(model, {"lr": torch.tensor(resolved.defaults["lr"])})
    captured = capture_optimizer(dict(model.named_parameters()), optimizer)
    spec.validate_capture(captured)
    torch.testing.assert_close(model.weight, before)
    assert captured.update_tasks
    assert any(
        "optimizer_group.0.lr" in task.binding_names for task in captured.update_tasks
    )
    invalid = replace(spec, control_bindings={"lr": ("optimizer_group.0.unknown",)})
    with pytest.raises(ValueError, match="no captured binding"):
        invalid.validate_capture(captured)


def test_static_optimizer_option_cannot_be_advertised_as_runtime_control():
    spec = definition()
    model = torch.nn.Linear(2, 1, bias=False)
    model.weight.grad = torch.ones_like(model.weight)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, foreach=False)
    captured = capture_optimizer(dict(model.named_parameters()), optimizer)
    with pytest.raises(ValueError, match="no captured binding"):
        spec.validate_capture(captured)


def test_unused_tensor_option_cannot_be_advertised_as_runtime_control():
    model = torch.nn.Linear(2, 1, bias=False)
    model.weight.grad = torch.ones_like(model.weight)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, foreach=False)
    optimizer.param_groups[0]["unused"] = torch.tensor(1.0)
    captured = capture_optimizer(dict(model.named_parameters()), optimizer)
    name = "optimizer_group.0.unused"
    assert any(binding.name == name for binding in captured.bindings)
    spec = OptimizerDefinition(
        "external/unused@1",
        create,
        ControlSchema({"unused": ScalarControl(1.0)}),
        {"unused": (name,)},
    )
    with pytest.raises(ValueError, match="no captured binding"):
        spec.validate_capture(captured)


def test_optimizer_default_overrides_are_nonsticky_and_unknowns_fail():
    registry = Registry()
    spec = definition()
    registry.add_optimizer(spec, source=SOURCE)
    assert registry.resolve_optimizer(spec.id, defaults={"lr": 0}).defaults["lr"] == 0
    assert registry.resolve_optimizer(spec.id).defaults["lr"] == 0.01
    with pytest.raises(ValueError, match="unknown controls"):
        registry.resolve_optimizer(spec.id, defaults={"clip_grad_norm": 1})
    with pytest.raises(ValueError, match="exactly"):
        registry.resolve_optimizer(spec.id).create(torch.nn.Linear(2, 1), {})
    with pytest.raises(ValueError, match="every optimizer control"):
        replace(spec, control_bindings={})
