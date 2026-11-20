"""Geometry/provider descriptions use the same model factory and phase engine."""

import subprocess
import sys
from dataclasses import asdict, dataclass, replace

import pytest

from engine_tests.fixtures.component_operations import components
from engine_tests.training.test_setup_resolution import arguments, installed
from shadowspill.operator_preparation import OperatorPreparation
from shadowspill.parallelism import GroupRef
from shadowspill.resources import Resource
from shadowspill.training.components import ComponentLayout, ModelBuild
from shadowspill.training.preparation.capture.templates import capture_inventory
from shadowspill.training.preparation.configuration.layouts import prepare_layouts
from shadowspill.training.preparation.setup import resolve_training
from shadowspill_client import BuildCandidates, PlanningOptions, Preparation


@dataclass(frozen=True)
class Config:
    num_classes: int = 3
    capacity: int = 0
    rank: int = 0


def describe(config, context):
    capacity = max(layout.buffer.max_datums for layout in context.family)
    return ModelBuild(
        {**asdict(config), "capacity": capacity, "rank": context.layout.logical_rank},
        resources={
            "provider": Resource(
                "not_imported.provider:factory", {"capacity": capacity}
            )
        },
        operator_preparation=[OperatorPreparation("not_imported.binary:factory")],
    )


def prepare(component):
    registry = installed(model=component)
    setup = resolve_training(
        registry,
        **arguments(),
        operations={"step": ["metrics"], "forward": ["predictions"]},
        prepare=Preparation(
            candidates=BuildCandidates(microbatch_sizes=[2, 4]),
            planning=PlanningOptions(microbatch_counts=[1, 2]),
        ),
    )
    return setup, capture_inventory(prepare_layouts(setup))


def test_build_context_covers_the_requested_family_and_never_constructs_a_model():
    calls = []
    model = replace(
        components()[0].component,
        config=Config,
        create=lambda config: calls.append(config) or config,
        build=describe,
    )
    setup, inventory = prepare(model)
    assert not calls
    assert {
        t.binding.inputs.layout.buffer.max_datums for t in inventory.templates.values()
    } == {2, 4}
    constructions = [t.model_build for t in inventory.templates.values()]
    assert all(c.config["capacity"] == 4 for c in constructions)
    assert all(c.resources == constructions[0].resources for c in constructions)
    assert all(
        t.signature()["model_build"] == t.model_build.to_dict()
        for t in inventory.templates.values()
    )
    result = setup.model.create(build=constructions[0])
    assert result == Config(capacity=4) and calls == [result]
    assert setup.model.config["capacity"] == 0
    with pytest.raises(TypeError):
        constructions[0].resources["another"] = constructions[0].resources["provider"]


def test_invalid_provider_groups_and_untyped_description_fail_before_construction():
    model = components()[0].component
    for hook, error, message in (
        (lambda config, context: {}, TypeError, "must return ModelBuild"),
        (
            lambda config, context: ModelBuild(
                asdict(config),
                {
                    "bad": Resource(
                        "absent.provider:factory",
                        groups={"exchange": GroupRef("absent", ((0,),))},
                    )
                },
            ),
            ValueError,
            "unknown logical group",
        ),
        (lambda config, context: ModelBuild({"unknown": 1}), TypeError, "unknown"),
    ):
        with pytest.raises(error, match=message):
            prepare(replace(model, build=hook))


def test_changed_provider_descriptions_change_capture_identity():
    model = replace(components()[0].component, config=Config, build=describe)
    _, first = prepare(model)

    def different(config, context):
        result = describe(config, context)
        return replace(
            result,
            resources={
                "provider": Resource("not_imported.provider:factory", {"capacity": 8})
            },
        )

    _, second = prepare(replace(model, build=different))
    assert first.templates.keys().isdisjoint(second.templates)


def test_component_coordinates_are_source_bound_before_any_model_construction():
    model = replace(components()[0].component, config=Config, build=describe)
    _, first = prepare(model)
    layout = ComponentLayout("weight", ("payload",), (0, 1), "external/encoded@1")

    def different(config, context):
        return replace(describe(config, context), component_layouts=(layout,))

    _, second = prepare(replace(model, build=different))
    assert first.templates.keys().isdisjoint(second.templates)
    for template in second.templates.values():
        assert template.signature()["model_build"]["component_layouts"] == [
            layout.to_dict()
        ]


def test_provider_descriptions_and_inventory_need_no_framework_or_provider_import():
    code = """
import sys
for name in ('torch', 'triton', 'mlops', 'shadowspill_workloads', 'not_imported'):
    sys.modules[name] = None
from dataclasses import replace
from engine_tests.training import test_model_build as fixture
def unexpected(config):
    raise AssertionError('compiler hook during CPU description')
model = replace(fixture.components()[0].component,
                config=fixture.Config, build=fixture.describe,
                gradient_destination_rules=unexpected)
_, inventory = fixture.prepare(model)
assert len(inventory.templates) == 4
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stdout + result.stderr
