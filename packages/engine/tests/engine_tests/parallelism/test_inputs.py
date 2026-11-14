"""Input ownership is explicit and does not follow optimizer/expert storage."""

import json
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest

from shadowspill.parallelism import (
    GroupRef,
    InputFieldPlacement,
    InputPlacement,
    OutputPlacement,
    Parallelism,
)
from shadowspill.parallelism.resolve import ParameterGeometry, resolve_parameters


def pairs(*, loss="leader", sliced=False):
    return InputPlacement(
        GroupRef("input_pairs", ((2, 0), (3, 1))),
        fields=(
            InputFieldPlacement(
                ("inputs", "x"),
                "slice" if sliced else "replicate",
                0 if sliced else None,
            ),
            InputFieldPlacement(
                ("loss_inputs", "mask"),
                "slice" if sliced else "replicate",
                0 if sliced else None,
            ),
            InputFieldPlacement(("inputs", "metadata"), "shared"),
        ),
        loss=loss,
        outputs={
            "prediction": OutputPlacement("concat", 0) if sliced else OutputPlacement()
        },
    )


def test_input_partitions_do_not_change_parameter_ownership_or_legacy_serialization():
    base = Parallelism.dp(ranks_per_replica=4)
    declarations = (ParameterGeometry(("weight",), (8, 8), True),)
    independent = replace(base, inputs=InputPlacement.independent(4))
    cooperating = replace(base, inputs=pairs())
    assert (
        resolve_parameters(base, declarations)
        == resolve_parameters(independent, declarations)
        == resolve_parameters(cooperating, declarations)
    )
    assert "inputs" not in base.to_dict()
    assert independent.inputs.partition_count == 4
    assert cooperating.inputs.partition_count == 2
    for value in (base, independent, cooperating):
        assert Parallelism.from_dict(json.loads(json.dumps(value.to_dict()))) == value
    assert cooperating.inputs.partition_index(0) == 0
    assert cooperating.inputs.partition_index(1) == 1
    assert [rank for rank in range(4) if cooperating.inputs.contributes_loss(rank)] == [
        2,
        3,
    ]
    assert [
        rank for rank in range(4) if cooperating.inputs.owns_output("prediction", rank)
    ] == [2, 3]


@pytest.mark.parametrize("sliced", [False, True])
@pytest.mark.parametrize("length", [0, 1, 7])
def test_aligned_delivery_counts_each_loss_and_output_once(sliced, length):
    placement = pairs(loss="partitioned" if sliced else "leader", sliced=sliced)
    records = [np.arange(length * 3).reshape(length, 3) + offset for offset in (0, 100)]
    masks = [np.arange(length) % 2 == 0 for _ in records]
    totals, outputs = [0, 0], [[], []]
    for group in placement.partitions.groups:
        for rank in group:
            partition = placement.partition_index(rank)
            original, mask = records[partition], masks[partition]
            values = original[placement.region(("inputs", "x"), original.shape, rank)]
            valid = mask[placement.region(("loss_inputs", "mask"), mask.shape, rank)]
            assert len(values) == len(valid)
            if placement.contributes_loss(rank):
                totals[partition] += values[valid].sum()
            if placement.owns_output("prediction", rank):
                outputs[partition].append(values)
    for index, original in enumerate(records):
        assert totals[index] == original[masks[index]].sum()
        np.testing.assert_array_equal(np.concatenate(outputs[index]), original)


def test_scheduled_capacity_counts_padding_and_unique_input_partitions():
    distinct, cooperating = InputPlacement.independent(4), pairs()
    assert distinct.scheduled_capacity([[64 << 10]] * 4) == 256 << 10
    assert cooperating.scheduled_capacity([[64 << 10]] * 2) == 128 << 10
    one = InputPlacement(GroupRef("all", ((0, 1, 2, 3),)))
    assert one.scheduled_capacity([[64 << 10] * 3]) > 128 << 10
    assert one.scheduled_capacity([[128 << 10]]) == 128 << 10
    # Masked model buffers still use their actual shape. Asymmetry is accounted
    # for here; collective-safe schedule support must be checked by preparation.
    assert (
        cooperating.scheduled_capacity([[64 << 10], [32 << 10, 32 << 10]]) == 128 << 10
    )
    with pytest.raises(ValueError, match="every input partition"):
        cooperating.scheduled_capacity([[64 << 10]])
    with pytest.raises(ValueError, match="nonnegative integers"):
        one.scheduled_capacity([[True]])


def test_invalid_missing_or_ambiguous_declarations_fail_before_routing():
    with pytest.raises(ValueError, match="every logical rank"):
        Parallelism.dp(ranks_per_replica=4, inputs=InputPlacement.independent(2))
    with pytest.raises(ValueError, match="nonempty names"):
        InputFieldPlacement("tokens")
    with pytest.raises(ValueError, match="unique"):
        replace(pairs(), fields=[pairs().fields[0]] * 2)
    for mode, axis in (
        ("slice", None),
        ("slice", True),
        ("shared", 0),
        ("unknown", None),
    ):
        with pytest.raises(ValueError):
            InputFieldPlacement(("x",), mode, axis)
    with pytest.raises(ValueError, match="slice axis"):
        InputFieldPlacement(("x",), "slice", 2).region((2,), index=0, workers=2)
    with pytest.raises(ValueError, match="no delivery declaration"):
        pairs().region(("missing",), (2,), 0)
    with pytest.raises(ValueError, match="outside"):
        pairs().contributes_loss(10)
    with pytest.raises(TypeError):
        pairs().outputs["prediction"] = OutputPlacement("sum")
    with pytest.raises(ValueError, match="version"):
        InputPlacement.from_dict({**pairs().to_dict(), "version": True})


def test_input_placement_is_available_in_framework_free_planning_process():
    code = """
import json, sys
for name in ('torch', 'numpy', 'shadowspill_client', 'shadowspill_workloads', 'mlops'):
    sys.modules[name] = None
from shadowspill.parallelism import InputPlacement, Parallelism
p = Parallelism.dp(ranks_per_replica=4, inputs=InputPlacement.independent(4))
assert Parallelism.from_dict(json.loads(json.dumps(p.to_dict()))) == p
assert p.inputs.scheduled_capacity([[32] for _ in range(4)]) == 128
"""
    run = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=30
    )
    assert run.returncode == 0, run.stdout + run.stderr
