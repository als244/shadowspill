"""The existing regression model through public, framework-free definitions."""

from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

from shadowspill.training.definitions import (
    ModelDefinition,
    ModelInitialization,
    ObjectiveDefinition,
)
from shadowspill_client.protocol.definitions import (
    DefinitionSource,
    OutputSet,
    Participation,
    TensorOutput,
)
from shadowspill_client.protocol.inputs import Capacity, InputSchema, TensorField


@dataclass(frozen=True)
class RegressionConfig:
    pass


def create(config):
    from shadowspill_workloads.models.regression.model import Regression

    return Regression()


def initialization(config, context):
    if context.weights.kind != "random":
        raise ValueError("regression requires explicit random weights or a checkpoint")

    @contextmanager
    def open_loader(runtime):
        from shadowspill_workloads.models.regression.initialization import fill_tensor

        yield partial(fill_tensor, seed=context.weights.seed)

    # The largest logical tensor is 512 FP32 values. Include the destination,
    # full deterministic sampler block and bounded generator/loader state.
    return ModelInitialization(peak_bytes=64 << 10, chunk_bytes=4096, open=open_loader)


def predictions(model, inputs, loss_inputs, loss_params):
    return {"prediction": model(**inputs)}


def bind(config, key):
    return weighted_squared_error if key.outputs == "metrics" else predictions


def weighted_squared_error(model, inputs, loss_inputs, loss_params):
    import torch

    from shadowspill.pytorch.contracts import ObjectiveResult

    prediction = model(**inputs)
    valid = loss_inputs["valid"]
    difference = torch.where(valid[..., None], prediction - loss_inputs["targets"], 0)
    weighted = difference.square().sum(dim=-1) * loss_inputs["weights"]
    return ObjectiveResult(
        loss=weighted.sum(),
        metrics={"loss_sum": weighted.detach().sum(), "valid_examples": valid.sum()},
    )


def register(registry):
    root = Path(__file__).resolve().parents[1]
    digest = sha256()
    # Explicit source closure, including the existing model, pins editable installs.
    for relative in (
        "definitions/regression.py",
        "models/regression/model.py",
        "models/regression/initialization.py",
    ):
        digest.update(relative.encode())
        digest.update((root / relative).read_bytes())
    source = DefinitionSource(
        "shadowspill-workloads", version("shadowspill-workloads"), digest.hexdigest()
    )
    registry.add_model(
        ModelDefinition(
            id="shadowspill/regression@1",
            config=RegressionConfig,
            create=create,
            inputs=InputSchema(
                fields=[TensorField(("value",), "float32", (16,))],
                capacities=[
                    Capacity("max_examples", "examples", "count", scheduled=True)
                ],
            ),
            capabilities=("regression",),
            mutable_buffers=("updates",),
            initialization=initialization,
        ),
        source=source,
    )
    registry.add_objective(
        ObjectiveDefinition(
            id="shadowspill/weighted_squared_error@1",
            compute=weighted_squared_error,
            config=RegressionConfig,
            inputs=InputSchema(
                fields=[
                    TensorField(("targets",), "float32", (8,)),
                    TensorField(
                        ("valid",), "bool", required=False, default=True, padding=False
                    ),
                    TensorField(("weights",), "float32", required=False, default=1.0),
                ]
            ),
            requires=("regression",),
            participation=Participation(("valid",)),
            padding_safe=True,
            bind=bind,
            outputs={
                "metrics": OutputSet(
                    model_inputs=(("value",),),
                    loss_inputs=(("targets",), ("valid",), ("weights",)),
                    metrics={"loss_sum": "sum", "valid_examples": "sum"},
                ),
                "predictions": OutputSet(
                    model_inputs=(("value",),),
                    tensors={"prediction": TensorOutput("float32", (8,), "datum")},
                ),
            },
        ),
        source=source,
    )
