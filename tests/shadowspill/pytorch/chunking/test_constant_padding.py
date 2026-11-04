"""Padding independent dimensions preserves chunk values and additive VJPs."""

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from shadowspill.errors import CaptureError
from shadowspill.pytorch.capture.aot import capture_graph_pair
from shadowspill.pytorch.capture.artifacts import GraphArtifact
from shadowspill.pytorch.chunking import AxisChunks, ChunkAxis
from shadowspill.pytorch.chunking.transform import chunk_artifact
from tests.shadowspill.pytorch.chunking.test_axis_vjp import executable


@pytest.mark.parametrize("budget", [0.0, 1.0])
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("shared", [False, True])
def test_padding_independent_activation_or_weight_preserves_vjp(budget, batch, shared):
    torch.manual_seed(731)
    inputs = (
        torch.randn(batch, 17, 5, dtype=torch.float64, requires_grad=True),
        torch.randn(2 if shared else 8, 7, dtype=torch.float64, requires_grad=True),
    )

    def region(x, weight):
        if shared:
            weight = torch.nn.functional.pad(weight, (0, 0, 1, 2))
        else:
            x = torch.nn.functional.pad(x, (2, 1), value=1.25)
        return torch.sin(x @ weight)

    output = region(*inputs)
    tangent = torch.randn_like(output)
    expected = torch.autograd.grad(output, inputs, tangent)
    pair = capture_graph_pair(
        make_fx(region)(*inputs), inputs, original_output=output, memory_budget=budget
    )
    config = AxisChunks(7, {0: 1})
    forward = chunk_artifact(pair.forward, (ChunkAxis(1), None), 17, config)
    saved = forward.chunking.outputs[-pair.saved_value_count :]
    backward = chunk_artifact(pair.backward, (*saved, ChunkAxis(1)), 17, config)
    with torch.no_grad():
        actual = executable(forward)(*inputs)
        gradients = executable(backward)(*actual[-pair.saved_value_count :], tangent)
    torch.testing.assert_close(actual[0], output, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(gradients, expected, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("padding", [(0, 0, 1, 0), (0, 0, -1, 0), (0, 0, 1, -1)])
def test_padding_or_cropping_the_chunk_axis_is_refused(padding):
    x = torch.randn(17, 5)
    graph = make_fx(lambda x: torch.nn.functional.pad(x, padding))(x)
    artifact = GraphArtifact.capture(
        kind="forward", graph_module=graph, example_inputs=(x,)
    )
    with pytest.raises(CaptureError, match="changes the logical chunk axis"):
        chunk_artifact(artifact, (ChunkAxis(0),), 17, AxisChunks(7, {0: 0}))


@pytest.mark.parametrize("padding,value", [((2, 1), 0), ((-1, 0), 2), ((0, 0), 2)])
def test_reduction_padding_only_adds_zero_or_crops(padding, value):
    x = torch.randn(17, 5, dtype=torch.float64)

    def region(x):
        return (torch.nn.functional.pad(x.sum(0), padding, value=value),)

    artifact = GraphArtifact.capture(
        kind="forward", graph_module=make_fx(region)(x), example_inputs=(x,)
    )
    chunked = chunk_artifact(artifact, (ChunkAxis(0),), 17, AxisChunks(7, {0: 0}))
    torch.testing.assert_close(
        executable(chunked)(x), region(x), rtol=1e-12, atol=1e-12
    )


def test_nonzero_padding_of_reduction_is_refused():
    x = torch.randn(17, 5)
    graph = make_fx(lambda x: torch.nn.functional.pad(x.sum(0), (2, 1), value=3))(x)
    artifact = GraphArtifact.capture(
        kind="forward", graph_module=graph, example_inputs=(x,)
    )
    with pytest.raises(CaptureError, match="adds a shared value to a row reduction"):
        chunk_artifact(artifact, (ChunkAxis(0),), 17, AxisChunks(7, {0: 0}))
