"""Real CPU capture through the same external/supplied component interface."""

import subprocess
import sys
from dataclasses import replace

import pytest
import torch

from engine_tests.fixtures import external_component
from shadowspill.pytorch.capture.objective import capture_training_objective
from shadowspill.training.components import Registry, simple_loss
from shadowspill_client.protocol.components import (
    ComponentSource,
    OutputSet,
    Participation,
)


def external():
    registry = Registry()
    external_component.register(registry)
    model = registry.resolve_model("external/image_classifier@1")
    objective = registry.resolve_objective("external/classification@1", model=model)
    return registry, model, objective


def test_description_is_portable_pinned_and_does_not_construct_models(monkeypatch):
    registry = Registry()
    monkeypatch.setattr(
        external_component,
        "create",
        lambda _: pytest.fail("constructed during inspection"),
    )
    external_component.register(registry)
    description = registry.resolve_model("external/image_classifier@1")
    assert description.config == {"num_classes": 3}
    assert description.inputs.capacities[0].name == "max_examples"
    assert description.source == external_component.SOURCE
    with pytest.raises(TypeError):
        description.config["num_classes"] = 6
    with pytest.raises(ValueError, match="source changed"):
        registry.resolve_model(
            description.component.id,
            source=ComponentSource("external-classifier-fixture", "1", "b" * 64),
        )
    with pytest.raises(ValueError, match="not installed"):
        registry.resolve_model("external/image_classifier@2")
    with pytest.raises(TypeError):
        registry.resolve_model(description.component.id, config={"misspelled": 3})


def test_external_component_captures_weighted_masked_gradients_and_runtime_controls():
    _, description, objective = external()
    torch.manual_seed(82)
    model = description.create()
    inputs = {"pixels": torch.randn(4, 1, 2, 2)}
    loss_inputs = {
        "class_id": torch.tensor([0, 2, 1, -999]),
        "valid": torch.tensor([True, True, True, False]),
        "weights": torch.tensor([1.0, -0.5, 0.0, 0.0]),
    }
    controls = {"temperature": torch.tensor(1.0)}
    capture = capture_training_objective(
        model, objective.computation(), [inputs, loss_inputs, controls]
    )
    graph = capture.exported.exported_program.module()
    for temperature in (1.0, 2.0):
        controls["temperature"] = torch.tensor(temperature)
        actual, valid = graph(inputs, loss_inputs, controls)
        expected = (
            torch.nn.functional.cross_entropy(
                model(**inputs)[:3] / temperature,
                loss_inputs["class_id"][:3],
                reduction="none",
            )
            .mul(loss_inputs["weights"][:3])
            .sum()
        )
        actual_grad = torch.autograd.grad(
            actual, tuple(model.parameters()), retain_graph=True
        )
        expected_grad = torch.autograd.grad(expected, tuple(model.parameters()))
        torch.testing.assert_close(actual, expected)
        for a, b in zip(actual_grad, expected_grad, strict=True):
            torch.testing.assert_close(a, b)
        assert valid.item() == 3  # Zero and negative weights do not alter validity.
    loss_inputs["valid"].zero_()
    actual, valid = graph(inputs, loss_inputs, controls)
    assert valid.item() == 0 and actual.item() == 0
    assert all(
        torch.count_nonzero(g).item() == 0
        for g in torch.autograd.grad(actual, tuple(model.parameters()))
    )


def test_extension_lookup_never_imports_the_catalog():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'shadowspill_workloads', 'mlops', 'transformers'}:
            raise AssertionError('forbidden catalog dependency: '+fullname)
sys.meta_path.insert(0, Block())
from engine_tests.fixtures.external_component import register
from shadowspill.training.components import Registry
registry = Registry()
register(registry)
model = registry.resolve_model('image_classifier')
objective = registry.resolve_objective('classification', model=model)
assert model.component.id == 'external/image_classifier@1'
assert objective.component.id == 'external/classification@1'
assert objective.defaults == {'temperature': 1.0}
assert 'torch' not in sys.modules
assert model.create().head.out_features == 3
""",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_registry_rejects_duplicate_incompatible_and_misdeclared_components():
    registry, model, objective = external()
    with pytest.raises(ValueError, match="already registered"):
        registry.add_model(model.component, source=model.source)
    incompatible = replace(
        objective.component, id="external/requires_other@1", requires=("other",)
    )
    registry.add_objective(incompatible, source=objective.source)
    with pytest.raises(ValueError, match="lacks required"):
        registry.resolve_objective(incompatible.id, model=model)
    bad_validity = replace(
        objective.component,
        id="external/bad_validity@1",
        participation=Participation(("weights",)),
    )
    registry.add_objective(bad_validity, source=objective.source)
    with pytest.raises(ValueError, match="boolean loss field"):
        registry.resolve_objective(bad_validity.id, model=model)
    bad_outputs = replace(
        objective.component,
        id="external/bad_outputs@1",
        outputs={"metrics": OutputSet(model_inputs=(("tokens",),))},
    )
    registry.add_objective(bad_outputs, source=objective.source)
    with pytest.raises(ValueError, match="undeclared input"):
        registry.resolve_objective(bad_outputs.id, model=model)


def test_simple_loss_adapter_uses_the_same_capture_path_without_normalization():
    _, description, _ = external()
    model = description.create()
    inputs = {"pixels": torch.ones(2, 1, 2, 2)}
    targets = {"targets": torch.ones(2, 3)}
    objective = simple_loss(
        lambda prediction, target: (prediction - target).square().sum()
    )
    capture = capture_training_objective(model, objective, [inputs, targets, {}])
    (actual,) = capture.exported.exported_program.module()(inputs, targets, {})
    torch.testing.assert_close(
        actual, (model(**inputs) - targets["targets"]).square().sum()
    )
