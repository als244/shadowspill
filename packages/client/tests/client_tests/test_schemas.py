"""Schema descriptions travel without callables or inferred field semantics."""

import json

import pytest

from shadowspill_client.protocol.components import (
    ComponentSource,
    OutputSet,
    Participation,
    TensorOutput,
)
from shadowspill_client.protocol.controls import ControlSchema, ScalarControl
from shadowspill_client.protocol.inputs import Capacity, InputSchema, TensorField


@pytest.mark.parametrize(
    "record",
    [
        InputSchema(
            [
                TensorField(("tokens",), "int64", ("positions",), "pack", 0),
                TensorField(("nested", "pixels"), "float32", (3, 224, 224)),
            ],
            [
                Capacity(
                    "token_capacity", "positions", "sum", ("tokens",), 0, scheduled=True
                ),
                Capacity("max_sequence_length", "positions", "max", ("tokens",), 0),
                Capacity("max_examples", "examples", "count"),
            ],
        ),
        ControlSchema(
            {"temperature": ScalarControl(1.0, minimum=0, minimum_inclusive=False)}
        ),
        OutputSet(
            model_inputs=(("tokens",),),
            tensors={
                "logprobs": TensorOutput(
                    "float32", ("positions",), "aligned", "positions"
                )
            },
            metrics={"loss_sum": "sum"},
        ),
        Participation(("valid",)),
        TensorOutput("float32", (3,), "global", reduction="sum"),
        TensorOutput("float32", ("n",), "aligned", "n", source=("inputs", "tokens")),
        ComponentSource("external-models", "1.0", "0" * 64),
    ],
)
def test_schema_json_roundtrip(record):
    assert type(record).from_dict(json.loads(json.dumps(record.to_dict()))) == record


@pytest.mark.parametrize(
    "factory",
    [
        lambda: TensorField((), "float32"),
        lambda: TensorField(("x",), "object"),
        lambda: TensorField(("x",), "int64", (True,)),
        lambda: TensorField(("x",), "int64", (2,), "pack", 2),
        lambda: TensorField(("x",), "int64", (2,), pack_axis=0),
        lambda: TensorField(("x",), "int64", default=1),
        lambda: TensorField(("x",), "int64", required=False),
        lambda: TensorField(("x",), "int64", padding=float("nan")),
        lambda: InputSchema(
            [TensorField(("x",), "int64"), TensorField(("x", "nested"), "float32")]
        ),
        lambda: InputSchema(
            [TensorField(("x",), "int64")],
            [Capacity("limit", "positions", "sum", ("x",), 0)],
        ),
        lambda: Capacity("limit", "examples", "count", ("x",)),
        lambda: TensorOutput("float32", (3,), "aligned", "positions"),
        lambda: TensorOutput("float32", (), "global"),
        lambda: TensorOutput("float32", ("n",), "global", reduction="sum"),
        lambda: TensorOutput("float32", (), "global", reduction="mean"),
        lambda: TensorOutput("float32", (), "datum", reduction="sum"),
        lambda: TensorOutput("float32", ("n",), "datum", source="inputs.x"),
        lambda: TensorOutput("float32", ("n",), "datum", source=("x",)),
        lambda: TensorOutput("float32", ("n",), "datum", source=("wrong", "x")),
        lambda: TensorOutput("float32", (3,), "datum", source=("inputs", "x")),
        lambda: OutputSet(
            tensors={"x": TensorOutput("float32", (), "global", reduction="sum")},
            metrics={"x": "sum"},
        ),
        lambda: ControlSchema(
            {"x": ScalarControl(0.0, minimum=0.0, minimum_inclusive=False)}
        ),
        lambda: ScalarControl(True, "float"),
        lambda: ScalarControl(1.2, "int"),
        lambda: ScalarControl(True, "bool", minimum=0),
        lambda: ComponentSource("external", "1", "short"),
    ],
)
def test_invalid_schema_is_rejected_before_input_processing(factory):
    with pytest.raises((TypeError, ValueError)):
        factory()


def test_controls_are_nonsticky_validate_defaults_and_keep_zero_distinct():
    schema = ControlSchema(
        {
            "lr": ScalarControl(0.01, minimum=0),
            "beta": ScalarControl(0.9, minimum=0, maximum=1, maximum_inclusive=False),
            "signed_coefficient": ScalarControl(1.0),
        }
    )
    override = {"lr": 0.1, "signed_coefficient": -1}
    first = schema.resolve(defaults={"lr": 0.03}, overrides=override)
    override["lr"] = 0.9
    assert first["lr"] == 0.1
    assert schema.resolve(defaults={"lr": 0.03})["lr"] == 0.03
    assert schema.resolve(overrides={"lr": 0})["lr"] == 0
    for overrides in (
        {"lr": float("nan")},
        {"lr": True},
        {"beta": 1},
        {"clip_grad_norm": 1},
    ):
        with pytest.raises(ValueError):
            schema.resolve(overrides=overrides)
    with pytest.raises(ValueError):
        schema.resolve(defaults={"beta": 1}, overrides={"beta": 0.8})
    with pytest.raises(TypeError):
        first["lr"] = 3
