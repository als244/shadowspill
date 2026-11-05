"""Inactive optional leaves do not prevent compiler-compatible destinations."""

import operator
from dataclasses import replace

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from shadowspill.errors import CaptureError
from shadowspill.pytorch.capture.artifacts import GraphArtifact
from shadowspill.pytorch.capture.gradients import (
    GradientDestinationMapping,
    GradientDestinationRule,
    accumulate_gradient_outputs,
)
from shadowspill.pytorch.graph_pairs.serialization import CachedGraphArtifact
from tests.shadowspill.pytorch.capture._optional_gradient_destination_ops import fresh

RULE = GradientDestinationRule(
    "gradient_destination_optional_test.fresh.default",
    "gradient_destination_optional_test.accumulating.default",
    (2,),
    1,
)


def values(dtype):
    generator = torch.Generator().manual_seed(57281)
    return tuple(
        torch.randn(shape, generator=generator, dtype=dtype)
        for shape in ((7, 3), (3, 5), (7, 5))
    )


def artifact(inputs, *, bias=False, observe_none=False):
    def function(x, weight, tangent):
        dx, _bias, dw = fresh(x, weight, tangent, bias)
        return dx, dw

    graph = make_fx(function, tracing_mode="fake")(*inputs)
    if observe_none:
        producer = next(
            n
            for n in graph.graph.nodes
            if n.target == torch.ops.gradient_destination_optional_test.fresh.default
        )
        output = next(n for n in graph.graph.nodes if n.op == "output")
        with graph.graph.inserting_before(output):
            leaf = graph.graph.call_function(operator.getitem, (producer, 1))
            leaf.meta["val"] = None
        output.args = ((*output.args[0], leaf),)
        graph.recompile()
    return GraphArtifact.capture(
        kind="backward", graph_module=graph, example_inputs=inputs
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("bias", [False, True])
def test_optional_bias_vjp_reuses_weight_destination_and_restores(dtype, bias):
    inputs = values(dtype)
    original = artifact(inputs, bias=bias)
    transformed = accumulate_gradient_outputs(
        original, (1,), gradient_destination_rules=(RULE,)
    )
    assert transformed.operator_targets.count(RULE.replacement) == 1
    assert "aten.add_.Tensor" not in transformed.operator_targets
    assert {m.input_position for m in transformed.storage_contract.mutations} == {3}
    restored = CachedGraphArtifact.capture(transformed).restore()
    assert restored.storage_contract == transformed.storage_contract
    assert restored.compatibility_digest == transformed.compatibility_digest
    running = torch.ones_like(inputs[1])
    pointer = running.data_ptr()
    for multiplier in (1.0, -0.25):
        x, weight, tangent = inputs
        x, weight = x.clone().requires_grad_(), weight.clone().requires_grad_()
        dx, dw = torch.autograd.grad(x @ weight, (x, weight), tangent * multiplier)
        expected = running + dw
        actual = restored.graph_module(
            x.detach(), weight.detach(), tangent * multiplier, running
        )
        torch.testing.assert_close(actual[0], dx)
        torch.testing.assert_close(actual[1], expected)
        assert actual[1].data_ptr() == pointer


def test_observed_none_output_cannot_be_replaced_by_a_tensor():
    with pytest.raises(CaptureError, match="changed an unconsumed output"):
        accumulate_gradient_outputs(
            artifact(values(torch.float32), observe_none=True),
            (1,),
            gradient_destination_rules=(RULE,),
        )


def test_unobserved_tensor_output_still_preserves_its_contract():
    def discard_aux(_context):
        return GradientDestinationMapping((2,), ((),), {"drop_aux": True})

    with pytest.raises(CaptureError, match="changed an unconsumed output"):
        accumulate_gradient_outputs(
            artifact(values(torch.float32), bias=True),
            (1,),
            gradient_destination_rules=(replace(RULE, mapper=discard_aux),),
        )
