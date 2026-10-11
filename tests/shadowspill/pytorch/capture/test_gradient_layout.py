"""Activation cotangents must match the layout expected by the preceding task."""

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from shadowspill.pytorch.capture.aot import capture_graph_pair
from shadowspill.pytorch.capture.artifacts import TaskInputProvenance
from shadowspill.pytorch.graph_pairs.artifacts import GraphPairVariant
from shadowspill.task.inputs import TaskInputRole


@pytest.mark.parametrize("role", [TaskInputRole.ACTIVATION, TaskInputRole.PARAMETER])
def test_transposed_gradient_uses_boundary_layout_for_activations_only(role):
    x = torch.randn(3, 4, 4, requires_grad=True)

    def function(value):
        return (value * 2).transpose(-1, -2).contiguous()

    pair = capture_graph_pair(
        make_fx(function)(x),
        (x,),
        original_output=function(x),
        input_provenance=(TaskInputProvenance(role),),
    )
    result = GraphPairVariant("save", 1.0, pair).with_gradient_dtype(None).pair
    forward = result.forward.graph_module(x)
    dy = torch.randn_like(forward[0])
    got = result.backward.graph_module(*forward[1:], dy)[0]
    expected = torch.autograd.grad(function(x), x, dy)[0]
    torch.testing.assert_close(got, expected)
    if role is TaskInputRole.ACTIVATION:
        assert got.is_contiguous(), got.stride()
    else:
        # A dense transposed parameter gradient is valid and needs no copy.
        assert got.stride() == expected.stride()


def test_channels_last_activation_keeps_its_canonical_memory_format():
    x = torch.randn(2, 3, 4, 5).to(memory_format=torch.channels_last).requires_grad_()

    def function(value):
        return (value * 3).permute(0, 2, 3, 1).contiguous()

    pair = capture_graph_pair(
        make_fx(function)(x),
        (x,),
        original_output=function(x),
        input_provenance=(TaskInputProvenance(TaskInputRole.ACTIVATION),),
    )
    result = GraphPairVariant("save", 1.0, pair).with_gradient_dtype(None).pair
    forward = result.forward.graph_module(x)
    dy = torch.randn_like(forward[0])
    got = result.backward.graph_module(*forward[1:], dy)[0]
    expected = torch.autograd.grad(function(x), x, dy)[0]
    torch.testing.assert_close(got, expected)
    assert got.is_contiguous(memory_format=torch.channels_last)
