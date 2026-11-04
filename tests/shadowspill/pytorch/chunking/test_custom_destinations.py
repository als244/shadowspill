"""Opaque VJPs reuse canonical gradients through regions and nested chunks."""

from collections import Counter
from functools import reduce
from operator import add

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from shadowspill.pytorch.capture.gradients import accumulate_gradient_outputs
from shadowspill.pytorch.chunking import (
    AxisChunks,
    ChunkAxis,
    ChunkRegions,
    OperatorRule,
)
from shadowspill.pytorch.chunking.regions import region_artifact
from shadowspill.pytorch.chunking.transform import chunk_artifact
from shadowspill.pytorch.partition import IsolateOperations
from tests.shadowspill.pytorch.capture.test_gradient_destinations import (
    CALLS,
    RULE,
    artifact,
    function,
    reference,
    values,
)
from tests.shadowspill.pytorch.chunking.test_composition import executable

AXIS_RULES = tuple(
    OperatorRule(target, "1", (0, None, None), reductions=(1, 2))
    for target in (RULE.target, RULE.replacement)
)


def shared(x, a, b, tangent):
    contributions = [
        function(x + index * 0.25, a, b, tangent - index * 0.125) for index in range(3)
    ]
    return tuple(reduce(add, parts) for parts in zip(*contributions, strict=True))


def independent(inputs):
    x, a, b, tangent = inputs
    contributions = [
        reference((x + index * 0.25, a, b, tangent - index * 0.125))
        for index in range(3)
    ]
    return tuple(sum(parts) for parts in zip(*contributions, strict=True))


class Inner:
    def configure_stage(self, stage):
        axes = {
            i: domains["positions"]
            for i, domains in enumerate(stage.input_domains)
            if "positions" in domains
        }
        return (
            AxisChunks(4, axes, domain="positions", rules=AXIS_RULES) if axes else None
        )


def composed(base, rows, nested):
    policy = ChunkRegions(
        IsolateOperations((RULE.target, RULE.replacement), base="whole"), Inner()
    )
    if not nested:
        return region_artifact(
            base,
            policy,
            stage_id="shared",
            variant="recompute",
            gradient_destination_rules=(RULE,),
            gradient_leaves=(1, 2),
        )
    config = AxisChunks(
        9,
        {0: 0, 3: 0},
        domain="positions",
        rules=AXIS_RULES,
        regions=policy,
    )
    return chunk_artifact(
        base,
        (
            ChunkAxis(0),
            None,
            None,
            ChunkAxis(0),
            *(None for _ in base.example_arguments[4:]),
        ),
        rows,
        config,
        gradient_destination_rules=(RULE,),
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("nested", [False, True])
def test_shared_custom_vjp_one_buffer_across_regions_chunks_microbatches(dtype, nested):
    x, a, b, tangent = values(rows=23, dtype=dtype)
    with FakeTensorMode() as mode:
        base = artifact(tuple(mode.from_tensor(v) for v in (x, a, b, tangent)), shared)
    first = composed(base, len(x), nested)
    later = composed(
        accumulate_gradient_outputs(base, (1, 2), gradient_destination_rules=(RULE,)),
        len(x),
        nested,
    )
    CALLS.clear()
    priors = ()
    pointers = None
    expected = [torch.zeros_like(a), torch.zeros_like(b)]
    for micro, transformed in enumerate((first, later)):
        inputs = (x + micro * 0.5, a, b, tangent - micro * 0.75)
        wanted = independent(inputs)
        expected[0].add_(wanted[1])
        expected[1].add_(wanted[2])
        actual = executable(transformed)(*inputs, *priors)
        tolerance = 3e-5 if dtype == torch.float32 else 1e-11
        torch.testing.assert_close(
            actual, (wanted[0], *expected), rtol=tolerance, atol=tolerance
        )
        priors = actual[1:]
        current = tuple(v.data_ptr() for v in priors)
        if pointers is not None:
            assert current == pointers
        pointers = current
    # Three uses, two microbatches, nested tile sizes 9/9/5 and 4/4/1,4/4/1,4/1.
    calls = 3 * 2 * (8 if nested else 1)
    assert Counter(calls=calls, fresh_parameter_buffers=2) == CALLS


def test_observed_partial_sum_is_preserved_after_later_shared_use():
    def observed(x, a, b, tangent):
        first = function(x, a, b, tangent)
        second = function(x + 0.25, a, b, tangent - 0.125)
        third = function(x + 0.5, a, b, tangent - 0.25)
        partial = tuple(left + right for left, right in zip(first, second, strict=True))
        result = tuple(left + right for left, right in zip(partial, third, strict=True))
        return (*result, *partial)

    inputs = values(dtype=torch.float64)
    with FakeTensorMode() as mode:
        base = artifact(tuple(mode.from_tensor(v) for v in inputs), observed)
    transformed = composed(base, len(inputs[0]), nested=False)
    actual = executable(transformed)(*inputs)
    wanted = observed(*inputs)
    torch.testing.assert_close(actual, wanted, rtol=1e-11, atol=1e-11)
    assert actual[1].data_ptr() != actual[4].data_ptr()
    assert actual[2].data_ptr() != actual[5].data_ptr()
