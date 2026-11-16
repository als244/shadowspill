"""Padding independent dimensions preserves chunk values and additive VJPs."""

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from engine_tests.pytorch.chunking.test_axis_vjp import executable
from shadowspill.errors import CaptureError
from shadowspill.pytorch.capture.aot import capture_graph_pair
from shadowspill.pytorch.capture.artifacts import GraphArtifact
from shadowspill.pytorch.capture.export import capture_forward, inference_artifact
from shadowspill.pytorch.chunking import AxisChunks, ChunkAxis
from shadowspill.pytorch.chunking.transform import chunk_artifact


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


def exported_padding(function, inputs):
    class Model(torch.nn.Module):
        def forward(self, *values):
            return function(*values)

    artifact = inference_artifact(capture_forward(Model(), inputs))
    assert "aten.pad.default" in artifact.operator_targets
    return artifact


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("fill", [None, 1.25])
def test_forward_export_padding_alias_preserves_independent_and_shared_values(
    shared, fill
):
    inputs = (torch.randn(17, 5), torch.randn(2 if shared else 8, 7))

    def region(x, weight):
        if shared:
            weight = torch.nn.functional.pad(weight, (0, 0, 1, 2), value=fill)
        else:
            x = torch.nn.functional.pad(x, (2, 1), value=fill)
        return (x @ weight,)

    artifact = exported_padding(region, inputs)
    chunked = chunk_artifact(artifact, (ChunkAxis(0), None), 17, AxisChunks(7, {0: 0}))
    torch.testing.assert_close(executable(chunked)(*inputs), region(*inputs))


@pytest.mark.parametrize("padding", [(0, 0, 1, 0), (0, 0, -1, 0)])
def test_forward_padding_alias_cannot_change_token_coordinates(padding):
    inputs = (torch.randn(17, 5),)
    artifact = exported_padding(
        lambda x: (torch.nn.functional.pad(x, padding),), inputs
    )
    with pytest.raises(CaptureError, match="changes the logical chunk axis"):
        chunk_artifact(artifact, (ChunkAxis(0),), 17, AxisChunks(7, {0: 0}))


@pytest.mark.parametrize("fill", [None, 2.0])
def test_forward_padding_alias_preserves_reduction_safety(fill):
    inputs = (torch.randn(17, 5),)

    def region(x):
        return (torch.nn.functional.pad(x.sum(0), (2, 1), value=fill),)

    artifact = exported_padding(region, inputs)
    if fill is not None:
        with pytest.raises(CaptureError, match="adds a shared value"):
            chunk_artifact(artifact, (ChunkAxis(0),), 17, AxisChunks(7, {0: 0}))
    else:
        chunked = chunk_artifact(artifact, (ChunkAxis(0),), 17, AxisChunks(7, {0: 0}))
        torch.testing.assert_close(executable(chunked)(*inputs), region(*inputs))


@pytest.mark.parametrize("mode", ["reflect", "replicate", "circular"])
def test_forward_padding_alias_requires_constant_mode(mode):
    inputs = (torch.randn(17, 5),)
    artifact = exported_padding(
        lambda x: (torch.nn.functional.pad(x, (1, 1), mode=mode),), inputs
    )
    with pytest.raises(CaptureError, match="constant padding"):
        chunk_artifact(artifact, (ChunkAxis(0),), 17, AxisChunks(7, {0: 0}))
