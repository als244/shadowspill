"""Every field and global capacity is checked before any physical packing."""

import subprocess
import sys
from contextlib import contextmanager

import numpy as np
import pytest

from shadowspill_client import BatchConfig, Datum
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.inputs.validation import combined_schema, validate_batch
from shadowspill_client.protocol.components import Participation
from shadowspill_client.protocol.inputs import Capacity, InputSchema, TensorField
from shadowspill_client.protocol.payload import snapshot_batch


def sequence_schema():
    return combined_schema(
        InputSchema(
            fields=[
                TensorField(
                    ("tokens",),
                    "int64",
                    ("position",),
                    "pack",
                    0,
                    minimum=0,
                    maximum=100,
                )
            ],
            capacities=[
                Capacity("positions", "tokens", "sum", ("tokens",), 0, scheduled=True),
                Capacity("longest", "tokens", "max", ("tokens",), 0),
                Capacity("examples", "records", "count"),
            ],
        ),
        InputSchema(
            fields=[
                TensorField(("targets",), "int64", ("position",), "pack", 0),
                TensorField(
                    ("valid",),
                    "bool",
                    ("position",),
                    "pack",
                    0,
                    required=False,
                    default=True,
                    padding=False,
                ),
                TensorField(
                    ("weights",),
                    "float32",
                    ("position",),
                    "pack",
                    0,
                    required=False,
                    default=1.0,
                ),
            ]
        ),
    )


LIMITS = BatchConfig(limits={"positions": 10, "longest": 6, "examples": 4})


@contextmanager
def validated(datums, *, schema=None, limits=LIMITS, **options):
    with snapshot_batch(datums, BufferBudget(16 << 20)) as snapshot:
        yield validate_batch(snapshot, schema or sequence_schema(), limits, **options)


def test_complete_batch_preserves_alignment_defaults_and_zero_weight_participation():
    datums = [
        Datum(
            {"tokens": [1, 2, 3]}, {"targets": [2, 3, 4], "weights": [0.0, 0.0, 0.0]}
        ),
        Datum({"tokens": [4, 5]}, {"targets": [5, 6], "valid": [1, 0]}),
    ]
    with validated(datums) as batch:
        assert batch.capacities == {"positions": 5, "longest": 3, "examples": 2}
        assert batch.participating_positions(Participation(("valid",))) == 4
        assert batch.participating_positions(Participation()) == 2
        with pytest.raises(TypeError, match="boolean field"):
            batch.participating_positions(Participation(("weights",)))
        assert batch.dimensions == ({"position": 3}, {"position": 2})
        mask = batch.fields[0][("loss_inputs", "valid")]
        assert mask.source is None  # Broadcast defaults do not allocate a batch copy.
        output = np.zeros(6, dtype=np.bool_)[::2]
        mask.copy_into(output)
        np.testing.assert_array_equal(output, [True, True, True])
        weight = batch.fields[0][("loss_inputs", "weights")]
        assert weight.source.dtype == "float64" and weight.dtype == "float32"
        result = np.ones(6, dtype=np.float32)[::2]
        weight.copy_into(result)
        np.testing.assert_array_equal(result, [0, 0, 0])


@pytest.mark.parametrize(
    "bad, match",
    [
        (Datum({"tokens": [1, 2]}, {"targets": [1]}), "aligned dimension"),
        (Datum({"tokens": [1]}, {}), "missing required"),
        (Datum({"tokens": [1]}, {"targets": [1], "typo": {}}), "unknown input"),
        (Datum({"tokens": [-1]}, {"targets": [1]}), "below minimum"),
        (Datum({"tokens": [101]}, {"targets": [1]}), "above maximum"),
        (Datum({"tokens": [1]}, {"targets": [1], "valid": [2]}), "boolean values"),
        (Datum({"tokens": [1]}, {"targets": [1], "valid": {}}), "expected a tensor"),
        (Datum({"tokens": [1]}, {"targets": [1.5]}), "represented as int64"),
        (Datum({"tokens": [1]}, {"targets": [1], "weights": [float("nan")]}), "finite"),
    ],
)
def test_bad_later_datum_rejects_the_entire_call(bad, match):
    good = Datum({"tokens": [1]}, {"targets": [2]})
    with pytest.raises(ValueError, match=match), validated([good, bad]):
        pytest.fail("no partial batch may reach packing or execution")


@pytest.mark.parametrize(
    "lengths, match", [([6, 5], "positions"), ([7], "longest"), ([1] * 5, "examples")]
)
def test_global_limits_are_hard_not_typical_length_hints(lengths, match):
    datums = [Datum({"tokens": [1] * n}, {"targets": [2] * n}) for n in lengths]
    with pytest.raises(ValueError, match=match), validated(datums):
        pass


def test_output_closure_can_omit_unused_loss_fields_but_validates_supplied_ones():
    with validated(
        [Datum({"tokens": [1, 2]})], required_paths=[("inputs", "tokens")]
    ) as batch:
        assert set(batch.fields[0]) == {("inputs", "tokens")}
        with pytest.raises(ValueError, match="missing participation"):
            batch.participating_positions(Participation(("valid",)))
    with (
        pytest.raises(ValueError, match="aligned"),
        validated(
            [Datum({"tokens": [1, 2]}, {"targets": [1]})],
            required_paths=[("inputs", "tokens")],
        ),
    ):
        pass


def test_empty_batch_and_closed_owner_have_distinct_meanings():
    with validated([]) as batch:
        assert batch.capacities == {"positions": 0, "longest": 0, "examples": 0}
        assert batch.participating_positions(Participation(("valid",))) == 0
        snapshot = batch.owner
    with pytest.raises(RuntimeError, match="closed"):
        validate_batch(snapshot, sequence_schema(), LIMITS)


def test_nontext_geometry_and_nested_field_schema_use_the_same_path():
    schema = combined_schema(
        InputSchema(
            [
                TensorField(("camera", "pixels"), "float32", (3, "height", "width")),
                TensorField(("depth",), "float32", ("height", "width")),
            ],
            [Capacity("images", "records", "count")],
        ),
        InputSchema([TensorField(("label",), "int32")]),
    )
    datum = Datum(
        {"camera": {"pixels": np.zeros((3, 4, 5))}, "depth": np.ones((4, 5))},
        {"label": 3},
    )
    with validated(
        [datum], schema=schema, limits=BatchConfig(limits={"images": 2})
    ) as batch:
        assert batch.dimensions == ({"height": 4, "width": 5},)
        assert batch.fields[0][("loss_inputs", "label")].shape == ()


@pytest.mark.parametrize(
    "dtype,value",
    [
        ("int64", [float(1 << 63)]),
        ("int8", [128]),
        ("uint8", [-1]),
        ("float16", [-(1 << 63)]),
        ("float16", [65505.0]),
        ("float32", [1e300]),
        ("bfloat16", [3.4e38]),
    ],
)
def test_dtype_overflow_is_rejected_before_packed_conversion(dtype, value):
    schema = InputSchema(
        [TensorField(("inputs", "x"), dtype, (1,))], [Capacity("n", "records", "count")]
    )
    with (
        pytest.raises(ValueError, match=r"represented|overflow"),
        validated(
            [Datum({"x": value})], schema=schema, limits=BatchConfig(limits={"n": 1})
        ),
    ):
        pass


def test_bfloat16_rounding_matches_torch_and_preserves_existing_bfloat16_bits():
    torch = pytest.importorskip("torch")

    values = np.array(
        [0.0, -0.0, 1.0, 1.00390625, 1.01171875, -2.5, 1e-38], dtype=np.float32
    )
    schema = InputSchema(
        [TensorField(("inputs", "x"), "bfloat16", (len(values),))],
        [Capacity("n", "records", "count")],
    )
    expected = torch.from_numpy(values).bfloat16().view(torch.uint16).numpy()
    for value in (values, torch.from_numpy(values).bfloat16()):
        with validated(
            [Datum({"x": value})], schema=schema, limits=BatchConfig(limits={"n": 1})
        ) as batch:
            output = np.empty_like(expected)
            batch.fields[0][("inputs", "x")].copy_into(output)
            np.testing.assert_array_equal(output, expected)


def test_input_validation_imports_without_torch_catalog_or_operator_library():
    code = """
import sys
for name in ('torch', 'shadowspill_workloads', 'mlops', 'triton'):
    sys.modules[name] = None
from shadowspill_client.inputs.validation import validate_batch
from shadowspill_client import BatchConfig, Datum
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.inputs import Capacity, InputSchema, TensorField
from shadowspill_client.protocol.payload import snapshot_batch
s = InputSchema([TensorField(('inputs','x'), 'float32', (2,))],
                [Capacity('n','records','count')])
with snapshot_batch([Datum({'x':[1,2]})],BufferBudget(1048576)) as p:
    assert validate_batch(p,s,BatchConfig(limits={'n':1})).capacities['n']==1
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stdout + result.stderr
