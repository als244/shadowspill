"""Generic precision works for ordinary external components and CPU inspection."""

import subprocess
import sys
from dataclasses import replace

import pytest
import torch

from engine_tests.fixtures import external_component
from shadowspill.pytorch.capture.objective import capture_training_objective
from shadowspill.training.components import Registry, simple_loss
from shadowspill_client import Model


def registry_with_defaults():
    registry = Registry()
    external_component.register(registry)
    component = registry.resolve_model("image_classifier").component
    registry.add_model(
        replace(
            component, id="external/precise@1", parameter_dtypes={"head.*": "float64"}
        ),
        source=external_component.SOURCE,
    )
    return registry


def test_custom_component_needs_no_configure_hook_and_keeps_settings_portable():
    registry = registry_with_defaults()
    settings = Model("precise", precision={"parameter_dtypes": {"head.*": "float32"}})
    resolved = registry.resolve_model(settings.name, settings=settings)
    assert Model.from_dict(resolved.settings.to_dict()) == resolved.settings
    assert resolved.settings.precision == settings.precision
    assert resolved.config == {"num_classes": 3}
    assert resolved.component.configure is None
    with torch.device("meta"):
        model = resolved.create()
        default = registry.resolve_model("precise").create()
    assert model.head.weight.dtype == torch.float32
    assert default.head.weight.dtype == torch.float64
    with pytest.raises(TypeError):
        resolved.component.parameter_dtypes["head.weight"] = "float16"


def test_generic_keys_are_removed_before_installed_configuration_hook():
    registry = registry_with_defaults()
    component = registry.resolve_model("precise").component

    def configure(settings):
        assert settings.precision == {"custom": "value"}
        return settings.config

    registry.add_model(
        replace(component, id="external/hook@1", configure=configure),
        source=external_component.SOURCE,
    )
    settings = Model(
        "hook", precision={"custom": "value", "parameter_dtypes": {"head.*": "float32"}}
    )
    resolved = registry.resolve_model("hook", settings=settings)
    assert resolved.create().head.weight.dtype == torch.float32
    assert resolved.settings.precision == settings.precision


def test_override_capture_and_loaded_values_use_selected_dtype():
    registry = registry_with_defaults()
    settings = Model(
        "image_classifier",
        precision={"parameter_dtypes": {"head.*": "float64"}},
        trainable=["head.weight"],
    )
    model = registry.resolve_model("image_classifier", settings=settings).create()
    original = torch.tensor([0.123456789123] * 12, dtype=torch.float64).reshape(3, 4)
    model.load_state_dict(
        {"head.weight": original, "head.bias": torch.zeros(3, dtype=torch.float64)}
    )
    assert model.head.weight.requires_grad and not model.head.bias.requires_grad
    torch.testing.assert_close(model.head.weight, original, rtol=0, atol=0)
    inputs = {"pixels": torch.arange(8, dtype=torch.float64).reshape(2, 1, 2, 2) / 8}
    targets = {"targets": torch.zeros(2, 3, dtype=torch.float64)}
    loss = simple_loss(lambda prediction, target: (prediction - target).square().sum())
    captured = capture_training_objective(model, loss, [inputs, targets, {}])
    (actual,) = captured.exported.exported_program.module()(inputs, targets, {})
    expected = model(**inputs).square().sum()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        torch.autograd.grad(actual, model.head.weight)[0],
        torch.autograd.grad(expected, model.head.weight)[0],
        rtol=0,
        atol=0,
    )


def test_inspection_validates_rules_without_framework_or_model_construction():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        blocked = {'torch', 'triton', 'mlops', 'shadowspill_workloads'}
        if fullname.split('.')[0] in blocked:
            raise AssertionError('forbidden import: ' + fullname)
sys.meta_path.insert(0, Block())
from shadowspill.training.components import Registry
from shadowspill_client import Model
from engine_tests.fixtures.external_component import register
registry = Registry()
register(registry)
settings = Model('image_classifier',
                 precision={'parameter_dtypes': {'head.*': 'float32'}})
resolved = registry.resolve_model('image_classifier', settings=settings)
assert resolved.settings.precision == settings.precision
try:
    invalid = Model('image_classifier',
                    precision={'parameter_dtypes': {'head.*': 'fp32'}})
    registry.resolve_model('image_classifier', settings=invalid)
except ValueError as error:
    assert 'unsupported parameter dtype' in str(error)
else:
    raise AssertionError('invalid dtype accepted')
""",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("rules", [{"head.*": "unknown"}, ["head.*"], None])
def test_setup_rejects_invalid_generic_rules(rules):
    settings = Model("image_classifier", precision={"parameter_dtypes": rules})
    with pytest.raises((ValueError, TypeError), match=r"parameter.dtype"):
        registry_with_defaults().resolve_model("image_classifier", settings=settings)
