"""External definitions and owned examples shared by CPU/GPU operation checks."""

from contextlib import ExitStack, contextmanager
from dataclasses import replace

from engine_tests.fixtures import external_definition
from shadowspill.parallelism import InputFieldPlacement, InputPlacement
from shadowspill.training.definitions import OptimizerDefinition, Registry
from shadowspill.training.inputs.adapters import InputLayout
from shadowspill.training.inputs.materialize import materialize_buffer
from shadowspill.training.inputs.packing import pack_batch
from shadowspill.training.inputs.routing import route_buffer
from shadowspill.training.inputs.shapes import BufferShape, PackingSchedule
from shadowspill.training.inputs.validation import combined_schema, validate_batch
from shadowspill.training.outputs import OutputLayout
from shadowspill.training.preparation.binding import OperationBinding
from shadowspill.training.preparation.operations import OperationContract
from shadowspill_client import BatchConfig, Datum
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.controls import ControlSchema, ScalarControl
from shadowspill_client.protocol.definitions import OutputSet, TensorOutput
from shadowspill_client.protocol.operations import OperationKey
from shadowspill_client.protocol.payload import snapshot_batch


def predictions(model, inputs, loss_inputs, loss_params):
    assert not loss_inputs and not loss_params
    return {"prediction": model(**inputs)}


def objective_binding(config, key):
    return (
        external_definition.classification_loss
        if key.outputs == "metrics"
        else predictions
    )


def create_optimizer(model, config, controls):
    import torch

    return torch.optim.SGD(model.parameters(), lr=controls["lr"], foreach=False)


def definitions():
    registry = Registry()
    external_definition.register(registry)
    model = registry.resolve_model("external/image_classifier@1")
    objective = registry.resolve_objective("external/classification@1", model=model)
    spec = replace(
        objective.definition,
        id="external/classification_outputs@1",
        outputs={
            **objective.definition.outputs,
            "predictions": OutputSet(
                model_inputs=[("pixels",)],
                tensors={"prediction": TensorOutput("float32", (3,), "datum")},
            ),
        },
        bind=objective_binding,
    )
    registry.add_objective(spec, source=external_definition.SOURCE)
    objective = registry.resolve_objective(spec.id, model=model)
    optimizer = OptimizerDefinition(
        "external/operation_sgd@1",
        create_optimizer,
        ControlSchema({"lr": ScalarControl(0.01, minimum=0)}),
        {"lr": ["optimizer_group.0.lr"]},
    )
    registry.add_optimizer(optimizer, source=external_definition.SOURCE)
    return model, objective, registry.resolve_optimizer(optimizer.id)


def records(*, targets=True, masked=False):
    import numpy as np

    return [
        Datum(
            {"pixels": np.arange(4, dtype=np.float32).reshape(1, 2, 2) / 8 + i / 4},
            {"class_id": i, "weights": weight, "valid": not masked} if targets else {},
        )
        for i, weight in enumerate((1.0, -0.5, 0.0))
    ]


@contextmanager
def example(model, objective, key, *, datums=None, placement=None, rank=0):
    key = OperationKey(key, "metrics") if isinstance(key, str) else key
    contract = OperationContract(key, objective)
    schema = combined_schema(model.inputs, objective.inputs)
    limits = BatchConfig(limits={"max_examples": 4})
    shape = BufferShape(
        4,
        {
            tuple(f.path): (4, *f.shape)
            for f in schema.fields
            if tuple(f.path) in contract.required_paths
        },
    )
    schedule = PackingSchedule("operation", [[shape]])
    placement = placement or InputPlacement.independent(
        1, fields=[InputFieldPlacement(tuple(f.path)) for f in schema.fields]
    )
    budget = BufferBudget(8 << 20, max_requests=8)
    with ExitStack() as stack:
        owner = stack.enter_context(
            snapshot_batch(records() if datums is None else datums, budget)
        )
        validated = validate_batch(
            owner, schema, limits, required_paths=contract.required_paths
        )
        packed = pack_batch(validated, schema, limits, placement, schedule)
        buffer = stack.enter_context(materialize_buffer(packed, schema, 0, budget))
        inputs = model.bind_inputs(InputLayout(schema, shape, placement, rank))
        binding = OperationBinding(
            model,
            contract,
            inputs,
            OutputLayout(schema, contract.outputs, schedule, placement),
            0,
        )
        adapted = stack.enter_context(
            inputs.adapt(route_buffer(buffer, placement, rank), budget)
        )
        yield binding, adapted
    assert budget.used == (0, 0)
