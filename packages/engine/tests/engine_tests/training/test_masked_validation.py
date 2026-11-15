"""Masked sentinels are neutralized before conversion; active errors reject calls."""

from dataclasses import replace

import numpy as np
import pytest

from engine_tests.training.test_input_validation import validated
from shadowspill.training.inputs.validation import combined_schema
from shadowspill.training.inputs.values import CHUNK_ELEMENTS
from shadowspill_client import BatchConfig, Datum
from shadowspill_client.protocol.codec import decode, encode
from shadowspill_client.protocol.inputs import Capacity, InputSchema, TensorField


def schema(dtype="int8", **options):
    return combined_schema(
        InputSchema(
            [TensorField(("tokens",), "int64", ("position",), "pack", 0)],
            [Capacity("tokens", "tokens", "sum", ("tokens",), 0, scheduled=True)],
        ),
        InputSchema(
            [
                # Deliberately declare the consumer first to prove validation order.
                TensorField(
                    ("value",),
                    dtype,
                    ("position",),
                    "pack",
                    0,
                    validity=("valid",),
                    **options,
                ),
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
            ]
        ),
    )


def test_masked_casts_cross_chunk_boundaries_and_fill_noncontiguous_storage():
    length = 2 * CHUNK_ELEMENTS + 3
    values = np.resize(np.array([1.0, np.nan, np.inf, -1e300]), length)
    mask = np.arange(length) % 4 == 0
    inputs = Datum(
        {"tokens": np.zeros(length, dtype=np.int64)}, {"value": values, "valid": mask}
    )
    declared = schema(minimum=0, maximum=5)
    assert decode(encode(declared)) == declared
    with validated(
        [inputs], schema=declared, limits=BatchConfig(limits={"tokens": length})
    ) as batch:
        field = batch.fields[0][("loss_inputs", "value")]
        destination = np.full(length * 2, -1, dtype=np.int8)
        with np.errstate(all="raise"):
            field.copy_into(destination[::2])
            blocks = tuple(field.encoded_blocks())
        np.testing.assert_array_equal(destination[::2], mask.astype(np.int8))
        np.testing.assert_array_equal(destination[1::2], -1)
        np.testing.assert_array_equal(np.concatenate(blocks), mask.astype(np.int8))
        assert all(len(block) <= CHUNK_ELEMENTS for block in blocks)
        # The owned caller snapshot remains unchanged.
        assert np.isnan(field.source.array()[1])


@pytest.mark.parametrize(
    "value,match",
    [
        (np.nan, "finite"),
        (np.inf, "finite"),
        (9.0, "above maximum"),
        (-1.0, "below minimum"),
        (1.5, "represented as int8"),
    ],
)
def test_one_bad_active_position_in_later_datum_rejects_everything(value, match):
    good = Datum({"tokens": [0, 0]}, {"value": [1, np.nan], "valid": [True, False]})
    bad = Datum({"tokens": [0]}, {"value": [value], "valid": [True]})
    with (
        pytest.raises(ValueError, match=match),
        validated(
            [good, bad],
            schema=schema(minimum=0, maximum=5),
            limits=BatchConfig(limits={"tokens": 3}),
        ),
    ):
        pytest.fail("an invalid active value reached packing")


@pytest.mark.parametrize("dtype", ["float16", "float32", "bfloat16"])
def test_masked_nonfinite_and_overflowing_auxiliaries_are_neutral_before_cast(dtype):
    data = Datum(
        {"tokens": [0] * 4},
        {"value": [-0.5, np.nan, 1e300, -np.inf], "valid": [True, False, False, False]},
    )
    with validated(
        [data], schema=schema(dtype), limits=BatchConfig(limits={"tokens": 4})
    ) as batch:
        field = batch.fields[0][("loss_inputs", "value")]
        result = np.empty(4, dtype=np.uint16 if dtype == "bfloat16" else dtype)
        with np.errstate(all="raise"):
            field.copy_into(result)
        if dtype == "bfloat16":
            result = (result.astype(np.uint32) << 16).view(np.float32)
        np.testing.assert_array_equal(result, [-0.5, 0, 0, 0])


def test_default_mask_is_active_and_default_values_are_neutralized_when_masked():
    declared = schema("float32", required=False, default=0.5)
    data = Datum({"tokens": [0, 0, 0]}, {"valid": [True, False, True]})
    with validated(
        [data], schema=declared, limits=BatchConfig(limits={"tokens": 3})
    ) as batch:
        field = batch.fields[0][("loss_inputs", "value")]
        assert field.source is None
        result = np.empty(3, dtype=np.float32)
        field.copy_into(result)
        np.testing.assert_array_equal(result, [0.5, 0, 0.5])
    with (
        pytest.raises(ValueError, match="finite"),
        validated(
            [Datum({"tokens": [0]}, {"value": [np.nan]})],
            schema=declared,
            limits=BatchConfig(limits={"tokens": 1}),
        ),
    ):
        pytest.fail("an omitted optional mask made positions inactive")


def test_unused_training_mask_cannot_hide_bad_forward_targets_or_change_good_ones():
    selected = (("inputs", "tokens"), ("loss_inputs", "value"))
    options = dict(
        schema=schema(minimum=0),
        limits=BatchConfig(limits={"tokens": 1}),
        required_paths=selected,
    )
    with (
        pytest.raises(ValueError, match="below minimum"),
        validated(
            [Datum({"tokens": [0]}, {"value": [-1], "valid": [False]})], **options
        ),
    ):
        pytest.fail("an unused mask bypassed target validation")
    with validated(
        [Datum({"tokens": [0]}, {"value": [3], "valid": [False]})], **options
    ) as batch:
        field = batch.fields[0][("loss_inputs", "value")]
        assert field.validity is None
        result = np.empty(1, dtype=np.int8)
        field.copy_into(result)
        assert result[0] == 3


def test_masks_are_checked_before_consumers_and_schema_rejects_ambiguous_alignment():
    with (
        pytest.raises(ValueError, match="boolean values"),
        validated(
            [Datum({"tokens": [0]}, {"value": [np.nan], "valid": [2]})],
            schema=schema(),
            limits=BatchConfig(limits={"tokens": 1}),
        ),
    ):
        pytest.fail("an invalid mask reached its consumer")
    consumer = TensorField(("x",), "float32", ("p",), validity=("mask",))
    for mask in (
        TensorField(("mask",), "float32", ("p",)),
        TensorField(("mask",), "bool", ("q",)),
        TensorField(("mask",), "bool", ("p",), "pack", 0),
        TensorField(("mask",), "bool", ("p",), validity=("mask",)),
    ):
        with pytest.raises(ValueError, match="identical alignment"):
            InputSchema([consumer, mask])
    with pytest.raises(ValueError, match="padding must satisfy"):
        replace(consumer, minimum=1)
