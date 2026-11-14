"""Declared timing samples use bounded ordinary physical input materialization."""

from dataclasses import replace

import numpy as np
import pytest

from engine_tests.fixtures.definition_operations import definitions
from engine_tests.fixtures.external_definition import SOURCE
from engine_tests.training.test_input_adapters import sequence_definition
from engine_tests.training.test_preparation_templates import inventory
from engine_tests.training.test_setup_resolution import arguments, installed
from shadowspill.parallelism import InputFieldPlacement, InputPlacement
from shadowspill.training.definitions import (
    InputAdapter,
    InputLayout,
    Registry,
    RepresentativePattern,
    RepresentativeSample,
)
from shadowspill.training.inputs.adapters import InputBinding
from shadowspill.training.inputs.materialize import buffer_size
from shadowspill.training.inputs.shapes import BufferShape
from shadowspill.training.inputs.validation import combined_schema
from shadowspill.training.preparation.layouts import prepare_layouts
from shadowspill.training.preparation.representative import representative_inputs
from shadowspill.training.preparation.setup import resolve_training
from shadowspill.training.preparation.templates import capture_inventory
from shadowspill_client import (
    BatchConfig,
    BuildCandidates,
    PlanningOptions,
    Preparation,
)
from shadowspill_client.protocol.inputs import InputSchema, TensorField


def test_fixed_stacked_samples_are_compact_owned_and_use_schema_defaults():
    prepared = inventory()
    for template in prepared.templates.values():
        binding, pattern = template.binding.inputs, template.representative
        assert len(pattern.samples) == 1
        assert pattern.samples[0].count == binding.layout.buffer.max_datums
        assert template.signature()["scenario"]["pattern"] == pattern.to_dict()
        with representative_inputs(binding, pattern, max_bytes=2 << 20) as inputs:
            assert inputs.fields.keys() == binding.shapes.keys()
            assert not inputs.fields["inputs", "pixels"].flags.writeable
            assert (inputs.fields["inputs", "pixels"] == 1).all()
            if ("loss_inputs", "valid") in inputs.fields:
                assert inputs.fields["loss_inputs", "valid"].all()
                assert (inputs.fields["loss_inputs", "weights"] == 1).all()
            owner = inputs.owner.owner
        assert inputs.closed and owner.closed


def sequence_binding():
    registry = Registry()
    definition = sequence_definition()
    registry.add_model(definition, source=SOURCE)
    resolved = registry.resolve_model(definition.id)
    schema = combined_schema(
        resolved.inputs,
        InputSchema(
            [
                TensorField(
                    ("valid",), "bool", ("positions",), "pack", 0, padding=False
                ),
            ]
        ),
    )
    placement = InputPlacement.independent(
        1, fields=[InputFieldPlacement(tuple(f.path)) for f in schema.fields]
    )
    shape = BufferShape(3, {tuple(f.path): (8,) for f in schema.fields})
    return resolved.bind_inputs(InputLayout(schema, shape, placement, 0))


def test_variable_pattern_preserves_boundaries_groups_values_and_unused_padding():
    binding = sequence_binding()
    pattern = RepresentativePattern(
        "two_short_sequences",
        [
            RepresentativeSample(
                {("inputs", "x"): 2.0, ("loss_inputs", "valid"): True}, {"positions": 2}
            ),
            RepresentativeSample(
                {("inputs", "x"): -1.0, ("loss_inputs", "valid"): True},
                {"positions": 3},
            ),
        ],
        groups=[[0, 1]],
    )
    with representative_inputs(binding, pattern, max_bytes=2 << 20) as inputs:
        assert inputs.fields["inputs", "x"].tolist() == [2, 2, -1, -1, -1, 0, 0, 0]
        assert inputs.fields["inputs", "segments"].tolist() == [
            0,
            0,
            1,
            1,
            1,
            -1,
            -1,
            -1,
        ]
        assert (
            inputs.fields["loss_inputs", "valid"].tolist() == [True] * 5 + [False] * 3
        )
        assert inputs.owner.owner.groups == ((0, 1),)
        regions = inputs.owner.owner.regions
        assert regions[1].fields["inputs", "x"] == (slice(2, 5),)
        assert regions[1].fields["loss_inputs", "valid"] == (slice(2, 5),)
    # Maximum physical shape remains eight; typical use of five positions has
    # not lowered any admission limit or changed the compiled metadata ABI.
    assert binding.layout.buffer.fields["inputs", "x"] == (8,)
    assert binding.shapes["inputs", "segments"] == (8,)


def test_invalid_shapes_dimensions_groups_and_values_fail_without_arrays(monkeypatch):
    binding = sequence_binding()
    valid = RepresentativeSample(
        {("inputs", "x"): 1.0, ("loss_inputs", "valid"): True}, {"positions": 3}
    )

    def forbidden(*args, **kwargs):
        pytest.fail("allocated an array while describing samples")

    monkeypatch.setattr(np, "empty", forbidden)
    monkeypatch.setattr(np, "asarray", forbidden)
    for samples, message in (
        ([replace(valid, count=4)], "records exceed"),
        ([replace(valid, count=3)], "shape exceeds"),
        ([replace(valid, dimensions={})], "symbolic dimensions"),
        ([replace(valid, values={("inputs", "x"): 1})], "required fields"),
        (
            [replace(valid, values={**valid.values, ("loss_inputs", "valid"): 2})],
            "boolean",
        ),
    ):
        with pytest.raises(ValueError, match=message):
            RepresentativePattern("invalid", samples).describe(
                binding.layout.schema, binding.layout.buffer
            )
    with pytest.raises(ValueError, match="exactly once"):
        RepresentativePattern("invalid", [valid], groups=[[0, 0]])
    with pytest.raises(ValueError, match="65536"):
        RepresentativePattern("invalid", [replace(valid, count=65537)])
    with pytest.raises(TypeError):
        valid.dimensions["positions"] = 100


def test_representative_budget_precedes_allocations_and_writers_release_on_failure(
    monkeypatch,
):
    template = next(iter(inventory().templates.values()))
    binding, pattern = template.binding.inputs, template.representative
    size = buffer_size(binding.layout.buffer, binding.layout.schema, pattern.count)
    needed = max(size.staging, size.retained + binding.staging_bytes)
    with monkeypatch.context() as patch:
        patch.setattr(
            np, "empty", lambda *a, **kw: pytest.fail("allocated before admission")
        )
        with (
            pytest.raises(ValueError, match="representative staging needs"),
            representative_inputs(binding, pattern, max_bytes=needed - 1),
        ):
            pass
    with representative_inputs(binding, pattern, max_bytes=needed) as inputs:
        assert len(inputs.owner.owner.regions) == pattern.count
    failure = RuntimeError("metadata failed")
    owners = []

    def fail(context, arrays):
        owners.append(context.routed.owner)
        raise failure

    broken = InputBinding(binding.layout, InputAdapter(fill=fail, padding_safe=True))
    with (
        pytest.raises(RuntimeError) as caught,
        representative_inputs(broken, pattern, max_bytes=2 << 20),
    ):
        pass
    assert caught.value is failure
    assert owners[0].closed


def test_float_conversion_is_validated_including_bfloat16_rounding():
    field = TensorField(("inputs", "x"), "bfloat16", (), minimum=0, maximum=0.999)
    schema = InputSchema([field])
    placement = InputPlacement.independent(
        1, fields=[InputFieldPlacement(tuple(field.path))]
    )
    physical = BufferShape(1, {tuple(field.path): (1,)})
    binding = InputBinding(
        InputLayout(schema, physical, placement, 0), InputAdapter(padding_safe=True)
    )
    good = RepresentativePattern(
        "bf16", [RepresentativeSample({tuple(field.path): 0.5})]
    )
    with representative_inputs(binding, good, max_bytes=1 << 20) as adapted:
        assert adapted.fields["inputs", "x"].dtype == np.uint16
        assert adapted.torch_inputs()[0]["x"].item() == 0.5
    bad = RepresentativePattern(
        "rounded", [RepresentativeSample({tuple(field.path): 0.999})]
    )
    with (
        pytest.raises(ValueError, match=r"converted representative.*maximum"),
        representative_inputs(binding, bad, max_bytes=1 << 20),
    ):
        pass


def test_missing_patterns_and_patterns_exceeding_logical_limits_fail_before_build():
    prepared = inventory().layouts
    no_pattern = replace(
        prepared.setup.preset,
        definition=replace(prepared.setup.preset.definition, representative=None),
    )
    with pytest.raises(ValueError, match="must declare a representative"):
        capture_inventory(
            replace(prepared, setup=replace(prepared.setup, preset=no_pattern))
        )
    model, _, _ = definitions()
    schema = replace(
        model.inputs,
        capacities=[replace(c, scheduled=False) for c in model.inputs.capacities],
    )
    registry = installed(model=replace(model.definition, inputs=schema))
    # Four physical slots are legal under a one-record actual-input cap, but a
    # purported typical sample containing four real records is not.
    setup = resolve_training(
        registry,
        **{
            **arguments(),
            "batch": BatchConfig(limits={"max_examples": 1}),
            "prepare": Preparation(
                candidates=BuildCandidates(microbatch_sizes=[4]),
                planning=PlanningOptions(microbatch_counts=[1]),
            ),
        },
    )
    layouts = prepare_layouts(setup)
    with pytest.raises(ValueError, match="representative pattern exceeds max_examples"):
        capture_inventory(layouts)
