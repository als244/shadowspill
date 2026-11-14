"""The existing regression model through public, framework-free definitions."""

from dataclasses import dataclass
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

from shadowspill.training.definitions import ModelDefinition, ObjectiveDefinition
from shadowspill_client.protocol.definitions import (
    DefinitionSource,
    OutputSet,
    Participation,
)
from shadowspill_client.protocol.inputs import Capacity, InputSchema, TensorField


@dataclass(frozen=True)
class RegressionConfig:
    pass


def create(config):
    from shadowspill_workloads.recipes.regression.model import Regression

    return Regression()


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
    for relative in ("definitions/regression.py", "recipes/regression/model.py"):
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
            outputs={
                "metrics": OutputSet(
                    model_inputs=(("value",),),
                    loss_inputs=(("targets",), ("valid",), ("weights",)),
                    metrics={"loss_sum": "sum", "valid_examples": "sum"},
                )
            },
        ),
        source=source,
    )
