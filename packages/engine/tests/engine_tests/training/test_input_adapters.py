"""Installed input functions preserve boundaries through the capture interface."""

from contextlib import ExitStack, contextmanager
from dataclasses import replace

import numpy as np
import pytest
import torch

from engine_tests.fixtures import external_definition
from shadowspill.parallelism import GroupRef, InputFieldPlacement, InputPlacement
from shadowspill.pytorch.capture.objective import capture_training_objective
from shadowspill.training.definitions import (
    InputAdapter,
    InputLayout,
    InputView,
    ModelDefinition,
    Registry,
)
from shadowspill.training.inputs.adapters import InputBinding
from shadowspill.training.inputs.materialize import materialize_buffer
from shadowspill.training.inputs.packing import pack_batch
from shadowspill.training.inputs.routing import route_buffer
from shadowspill.training.inputs.shapes import BufferShape, PackingSchedule
from shadowspill.training.inputs.validation import combined_schema, validate_batch
from shadowspill_client import BatchConfig, Datum
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.inputs import Capacity, InputSchema, TensorField
from shadowspill_client.protocol.payload import snapshot_batch


@contextmanager
def physical_inputs(schema, limits, shape, datums, *, placement=None):
    placement = placement or InputPlacement.independent(
        1, fields=[InputFieldPlacement(tuple(f.path)) for f in schema.fields]
    )
    budget = BufferBudget(8 << 20, max_requests=8)
    with ExitStack() as stack:
        owned = stack.enter_context(snapshot_batch(datums, budget))
        validated = validate_batch(owned, schema, limits)
        schedule = PackingSchedule("fixture", [[shape]])
        packed = pack_batch(validated, schema, limits, placement, schedule)
        buffer = stack.enter_context(materialize_buffer(packed, schema, 0, budget))
        yield buffer, placement, budget
    assert budget.used == (0, 0)


def test_default_external_classifier_adapter_preserves_masked_padding_and_gradients():
    registry = Registry()
    external_definition.register(registry)
    resolved = registry.resolve_model("external/image_classifier@1")
    objective = registry.resolve_objective("external/classification@1", model=resolved)
    schema = combined_schema(resolved.inputs, objective.inputs)
    limits = BatchConfig(limits={"max_examples": 4})
    shape = BufferShape(4, {tuple(f.path): (4, *f.shape) for f in schema.fields})
    datums = [
        Datum(
            {"pixels": np.arange(4, dtype=np.float32).reshape(1, 2, 2) + i},
            {"class_id": i, "weights": weight},
        )
        for i, weight in enumerate((1.0, -0.5, 0.0))
    ]
    with physical_inputs(schema, limits, shape, datums) as (buffer, placement, budget):
        binding = resolved.bind_inputs(InputLayout(schema, shape, placement, 0))
        assert binding.adapter.padding_safe and binding.metadata_bytes == 0
        with binding.adapt(route_buffer(buffer, placement, 0), budget) as inputs:
            model_inputs, loss_inputs = inputs.torch_inputs()
            assert loss_inputs["valid"].tolist() == [True, True, True, False]
            assert np.shares_memory(
                inputs.fields[("inputs", "pixels")], buffer.fields[("inputs", "pixels")]
            )
            model = resolved.create()
            controls = {"temperature": torch.tensor(1.0)}
            captured = capture_training_objective(
                model, objective.computation(), [model_inputs, loss_inputs, controls]
            )
            loss, count = captured.exported.exported_program.module()(
                model_inputs, loss_inputs, controls
            )
            expected = objective.computation()(
                model,
                {"pixels": model_inputs["pixels"][:3]},
                {k: v[:3] for k, v in loss_inputs.items()},
                controls,
            ).loss
            torch.testing.assert_close(loss, expected)
            assert count.item() == 3
            actual_grad = torch.autograd.grad(
                loss, tuple(model.parameters()), retain_graph=True
            )
            expected_grad = torch.autograd.grad(expected, tuple(model.parameters()))
            for actual, target in zip(actual_grad, expected_grad, strict=True):
                torch.testing.assert_close(actual, target)
        with pytest.raises(RuntimeError, match="closed"):
            inputs.torch_inputs()


def sequence_adapter(config, layout):
    path = ("inputs", "x")
    if layout.local_shape(path) != layout.buffer.fields[path]:
        raise ValueError("this sequence adapter requires complete sequences per worker")
    capacity = layout.local_shape(path)[0]

    def fill(context, arrays):
        segments = arrays[("inputs", "segments")]
        for region in context.routed.owner.regions:
            segments[region.fields[path]] = region.slot

    return InputAdapter(
        metadata=[
            TensorField(
                ("inputs", "segments"),
                "int64",
                (capacity,),
                padding=-1,
                minimum=-1,
                maximum=layout.buffer.max_datums - 1,
            )
        ],
        fill=fill,
        padding_safe=True,
    )


def sequence_definition(*, adapter=sequence_adapter):
    def create(config):
        class Cumulative(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(0.5))

            def forward(self, x, segments):
                positions = torch.arange(x.numel())
                same = (segments[:, None] == segments[None, :]) & (
                    segments[:, None] >= 0
                )
                causal = positions[:, None] >= positions[None, :]
                return (same & causal).to(x.dtype) @ (x * self.weight)

        return Cumulative()

    return ModelDefinition(
        "external/cumulative@1",
        dict,
        create,
        InputSchema(
            [TensorField(("x",), "float32", ("positions",), "pack", 0)],
            [
                Capacity("positions", "positions", "sum", ("x",), 0, scheduled=True),
                Capacity("records", "records", "count"),
            ],
        ),
        input_adapter=adapter,
    )


def test_custom_metadata_changes_boundaries_without_recapture_or_model_name_logic():
    registry = Registry()
    definition = sequence_definition()
    registry.add_model(definition, source=external_definition.SOURCE)
    resolved = registry.resolve_model(definition.id)
    schema = combined_schema(
        resolved.inputs,
        InputSchema(
            [
                TensorField(
                    ("valid",),
                    "bool",
                    ("positions",),
                    "pack",
                    0,
                    required=False,
                    default=True,
                    padding=False,
                )
            ]
        ),
    )
    limits = BatchConfig(limits={"positions": 4, "records": 3})
    shape = BufferShape(3, {tuple(f.path): (4,) for f in schema.fields})
    model, graph, signatures = resolved.create(), None, []

    def objective(model, inputs, loss_inputs, controls):
        prediction = model(**inputs)
        return torch.where(
            loss_inputs["valid"], prediction.square(), 0
        ).sum(), prediction

    for values in (([1.0, 2.0], [5.0]), ([3.0], [4.0, 5.0])):
        datums = [Datum({"x": np.asarray(value, dtype=np.float32)}) for value in values]
        with physical_inputs(schema, limits, shape, datums) as (
            buffer,
            placement,
            budget,
        ):
            binding = resolved.bind_inputs(InputLayout(schema, shape, placement, 0))
            signatures.append(binding.signature())
            with binding.adapt(route_buffer(buffer, placement, 0), budget) as adapted:
                inputs, loss_inputs = adapted.torch_inputs()
                if graph is None:
                    capture = capture_training_objective(
                        model, objective, [inputs, loss_inputs, {}]
                    )
                    graph = capture.exported.exported_program.module()
                loss, prediction = graph(inputs, loss_inputs, {})
                expected = torch.zeros(4)
                cursor = 0
                for region in buffer.regions:
                    index = region.datum
                    sequence = torch.tensor(values[index]).cumsum(0) * model.weight
                    expected[cursor : cursor + len(sequence)] = sequence
                    cursor += len(sequence)
                torch.testing.assert_close(prediction, expected)
                torch.testing.assert_close(loss, expected.square().sum())
                actual_grad = torch.autograd.grad(
                    loss, model.weight, retain_graph=True
                )[0]
                torch.testing.assert_close(
                    actual_grad,
                    torch.autograd.grad(expected.square().sum(), model.weight)[0],
                )
                assert binding.metadata_bytes == 4 * 8
                assert not adapted.fields[("inputs", "segments")].flags.writeable
    assert signatures[0] == signatures[1]


def test_custom_adapter_required_for_packed_inputs_and_resolved_schema_is_pinned():
    registry = Registry()
    definition = sequence_definition(adapter=None)
    registry.add_model(definition, source=external_definition.SOURCE)
    resolved = registry.resolve_model(definition.id)
    schema = combined_schema(resolved.inputs, InputSchema([]))
    shape = BufferShape(2, {("inputs", "x"): (4,)})
    placement = InputPlacement.independent(
        1, fields=[InputFieldPlacement(("inputs", "x"))]
    )
    with pytest.raises(ValueError, match="require a model input adapter"):
        resolved.bind_inputs(InputLayout(schema, shape, placement, 0))
    with pytest.raises(ValueError, match="resolved model input schema"):
        resolved.bind_inputs(InputLayout(InputSchema([]), shape, placement, 0))


def test_generated_metadata_reserves_before_allocation_and_failures_release(
    monkeypatch,
):
    definition = InputSchema(
        [TensorField(("inputs", "x"), "float32", (2,))],
        [Capacity("records", "records", "count")],
    )
    shape = BufferShape(2, {("inputs", "x"): (2, 2)})
    with physical_inputs(
        definition,
        BatchConfig(limits={"records": 2}),
        shape,
        [Datum({"x": [1.0, 2.0]})],
    ) as (buffer, placement, budget):
        layout = InputLayout(definition, shape, placement, 0)
        metadata = TensorField(("inputs", "offsets"), "int64", (3,), minimum=0)
        binding = InputBinding(layout, InputAdapter(metadata=[metadata]))
        routed = route_buffer(buffer, placement, 0)
        original = np.empty
        monkeypatch.setattr(
            np, "empty", lambda *a, **kw: pytest.fail("allocated before reserve")
        )
        with pytest.raises(ValueError, match="staging bytes"):
            binding.adapt(routed, BufferBudget(1))
        monkeypatch.setattr(np, "empty", original)
        before = budget.used

        def bad(context, arrays):
            arrays[("inputs", "offsets")][0] = -1

        binding = InputBinding(layout, InputAdapter(metadata=[metadata], fill=bad))
        with pytest.raises(ValueError, match="below minimum"):
            binding.adapt(routed, budget)
        assert budget.used == before
        with pytest.raises(ValueError, match="output paths overlap"):
            InputBinding(
                layout, InputAdapter(metadata=[replace(metadata, path=("inputs", "x"))])
            )
        with pytest.raises(ValueError, match="preserve every source"):
            InputBinding(
                layout,
                InputAdapter(views=[InputView(("inputs", "x"), ("inputs", "x"), (5,))]),
            )


def test_explicit_reshape_keeps_storage_and_bfloat16_bits():
    definition = InputSchema(
        [TensorField(("inputs", "x"), "bfloat16", (2,))],
        [Capacity("records", "records", "count")],
    )
    shape = BufferShape(1, {("inputs", "x"): (1, 2)})
    with physical_inputs(
        definition,
        BatchConfig(limits={"records": 1}),
        shape,
        [Datum({"x": [1.0, -0.5]})],
    ) as (buffer, placement, budget):
        binding = InputBinding(
            InputLayout(definition, shape, placement, 0),
            InputAdapter(
                views=[InputView(("inputs", "nested", "value"), ("inputs", "x"), (2,))]
            ),
        )
        with binding.adapt(route_buffer(buffer, placement, 0), budget) as adapted:
            inputs, loss_inputs = adapted.torch_inputs()
            assert not loss_inputs and inputs["nested"]["value"].dtype == torch.bfloat16
            torch.testing.assert_close(
                inputs["nested"]["value"],
                torch.tensor([1.0, -0.5], dtype=torch.bfloat16),
            )
            assert np.shares_memory(
                adapted.fields[("inputs", "nested", "value")],
                buffer.fields[("inputs", "x")],
            )


def test_noncontiguous_worker_slice_is_preserved_but_never_copied_to_reshape():
    path = ("inputs", "x")
    definition = InputSchema(
        [TensorField(path, "float32", (2, 4))],
        [Capacity("records", "records", "count")],
    )
    shape = BufferShape(1, {path: (1, 2, 4)})
    placement = InputPlacement(
        GroupRef("together", ((0, 1),)),
        fields=[InputFieldPlacement(path, "slice", 2)],
        loss="partitioned",
    )
    with physical_inputs(
        definition,
        BatchConfig(limits={"records": 1}),
        shape,
        [Datum({"x": np.arange(8, dtype=np.float32).reshape(2, 4)})],
        placement=placement,
    ) as (buffer, placement, budget):
        layout = InputLayout(definition, shape, placement, 1)
        routed = route_buffer(buffer, placement, 1)
        with InputBinding(layout, InputAdapter()).adapt(routed, budget) as adapted:
            inputs, _ = adapted.torch_inputs()
            torch.testing.assert_close(
                inputs["x"], torch.tensor([[[2.0, 3.0], [6.0, 7.0]]])
            )
            assert not inputs["x"].is_contiguous()
        before = budget.used
        binding = InputBinding(
            layout, InputAdapter(views=[InputView(path, path, (4,))])
        )
        with pytest.raises(ValueError, match="contiguous source"):
            binding.adapt(routed, budget)
        assert budget.used == before
