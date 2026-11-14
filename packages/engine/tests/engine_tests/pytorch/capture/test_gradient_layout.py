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


@pytest.mark.parametrize("transpose", [False, True])
def test_shard_padding_is_in_the_backward_contract_and_preserved_by_accumulation(
    transpose,
):
    x = torch.randn(3, 5)
    if transpose:
        x = x.T
    x.requires_grad_()

    def function(value):
        return value.square().sum()

    pair = capture_graph_pair(
        make_fx(function)(x),
        (x,),
        original_output=function(x),
        input_provenance=(TaskInputProvenance(TaskInputRole.PARAMETER),),
    )
    variant = GraphPairVariant("save", 1.0, pair).with_gradient_dtype(
        torch.float32,
        gradient_padding={0: 4},
    )
    forward = variant.pair.forward.graph_module(x)
    arguments = (*forward[1:], torch.ones_like(forward[0]))
    got = variant.pair.backward.graph_module(*arguments)[0]
    torch.testing.assert_close(got, torch.autograd.grad(function(x), x)[0])
    assert got.shape == x.shape
    assert got.untyped_storage().nbytes() == 16 * 4
    from shadowspill.build.compilation import compile_task
    from shadowspill.pytorch.executable import restore_executable

    compiled = compile_task(variant.pair.backward)
    (compiled_output,) = restore_executable(compiled.payload)(*arguments)
    torch.testing.assert_close(compiled_output, got)
    view = compiled.manifest.storage_contract.output_views[0]
    allocation = next(
        item
        for item in compiled.manifest.root_allocations
        if item.root_id == view.root_id
    )
    assert allocation.requested_bytes == 16 * 4

    accumulating = variant.accumulating().pair.backward
    prior = torch.ones(16).as_strided(got.shape, got.stride())
    output = accumulating.graph_module(*arguments, prior)[0]
    torch.testing.assert_close(output, got + 1)
    assert output.data_ptr() == prior.data_ptr()
    assert accumulating.example_arguments[-1].untyped_storage().nbytes() == 16 * 4
