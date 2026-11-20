"""Installed destination contracts reach capture without entering client setup."""

from dataclasses import replace

import pytest

from engine_tests.fixtures.component_operations import components
from engine_tests.training.test_capture_precision import prepared
from shadowspill.errors import CaptureError
from shadowspill.pytorch.capture.gradients import GradientDestinationRule
from shadowspill.training.preparation.capture.build import build_template


def rule(version=1):
    return GradientDestinationRule(
        "example.backward.default", "example.accumulate.default", (0,), version
    )


@pytest.mark.parametrize("forward", [False, True])
def test_installed_destination_rules_are_resolved_only_for_training(
    monkeypatch, forward
):
    calls = []
    rules = [rule()]

    def factory(config):
        calls.append(config.num_classes)
        return rules

    registry, _, inventory = prepared("float32", gradient_destination_rules=factory)
    assert not calls
    templates = [
        template
        for template in inventory.templates.values()
        if (template.kind == "forward") == forward
    ]
    assert templates

    class Captured(Exception):
        pass

    def capture(*args, **options):
        if forward:
            assert "gradient_destination_rules" not in options
        else:
            assert options["gradient_destination_rules"] == tuple(rules)
            assert isinstance(options["gradient_destination_rules"], tuple)
        raise Captured

    monkeypatch.setattr("shadowspill.build_step", capture)
    monkeypatch.setattr("shadowspill.build_forward", capture)
    monkeypatch.setattr("shadowspill.build_forward_backward", capture)
    for template in templates:
        with pytest.raises(Captured):
            build_template(registry, inventory, template.id)
    assert calls == ([] if forward else [3] * len(templates))


def test_destination_factory_receives_normalized_build_configuration():
    _, setup, inventory = prepared(
        "float32", gradient_destination_rules=lambda config: [rule(config.num_classes)]
    )
    build = next(iter(inventory.templates.values())).model_build
    changed = replace(build, config={**build.config, "num_classes": 7})
    assert setup.model.config["num_classes"] == 3
    assert setup.model.gradient_destination_rules(changed) == (rule(7),)


@pytest.mark.parametrize(
    "returned,error",
    [(None, TypeError), ([object()], TypeError), ([rule(), rule()], CaptureError)],
)
def test_invalid_installed_destination_contracts_fail_before_model_creation(
    monkeypatch, returned, error
):
    registry, setup, inventory = prepared(
        "float32", gradient_destination_rules=lambda config: returned
    )
    template = next(t for t in inventory.templates.values() if t.kind == "step")
    monkeypatch.setattr(
        type(setup.model),
        "create",
        lambda *args, **kwargs: pytest.fail("construction before rule validation"),
    )
    with pytest.raises(error):
        build_template(registry, inventory, template.id)


def test_destination_hook_must_be_installed_callable():
    with pytest.raises(TypeError, match="installed factory"):
        replace(components()[0].component, gradient_destination_rules=[])
