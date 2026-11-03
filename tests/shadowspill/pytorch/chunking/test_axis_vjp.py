"""Axis chunking agrees with full explicit VJPs, including batch/tail layouts."""

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from shadowspill.errors import CaptureError
from shadowspill.pytorch.capture.aot import capture_graph_pair
from shadowspill.pytorch.chunking import AxisChunks
from shadowspill.pytorch.chunking.executable import ChunkedExecutable
from shadowspill.pytorch.chunking.geometry import ChunkAxis
from shadowspill.pytorch.chunking.transform import chunk_artifact


def executable(artifact):
    task = artifact.chunking
    return ChunkedExecutable(
        {
            **task.identity,
            "storage": task.storage_contract.to_dict(),
            "output_count": artifact.output_count,
            "input_offsets": [x.storage_offset for x in artifact.tensor_inputs],
        },
        {size: child.graph_module for size, child in task.children},
        {size: child.graph_module for size, child in task.accumulating_children},
    )


@pytest.mark.parametrize(
    "batch,rows,size",
    [(1, 1, 7), (2, 1, 7), (1, 17, 7), (2, 17, 7), (3, 14, 7), (2, 7, 20), (2, 3, 1)],
)
@pytest.mark.parametrize("budget", [0.0, 1.0])
@pytest.mark.parametrize("lora", [False, True])
def test_chunk_forward_and_arbitrary_vjp_match_full(batch, rows, size, budget, lora):
    torch.manual_seed(731)
    dtype = torch.float64
    inputs = (
        torch.randn(batch, rows, 5, dtype=dtype, requires_grad=True),
        torch.randn(5, 9, dtype=dtype, requires_grad=not lora),
        torch.randn(5, 3, dtype=dtype, requires_grad=True),
        torch.randn(3, 9, dtype=dtype, requires_grad=True),
        torch.randn(9, dtype=dtype, requires_grad=True),
    )

    def region(x, weight, a, b, bias):
        value = x @ weight + (x @ a) @ b + bias
        return torch.nn.functional.silu(value)

    graph = make_fx(region)(*inputs)
    pair = capture_graph_pair(
        graph, inputs, original_output=region(*inputs), memory_budget=budget
    )
    config = AxisChunks(size, {0: 1})
    forward = chunk_artifact(
        pair.forward, (ChunkAxis(1), None, None, None, None), rows, config
    )
    with torch.no_grad():
        expected = pair.forward.graph_module(*inputs)
        actual = executable(forward)(*inputs)
        torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)
        saved = forward.chunking.outputs[-pair.saved_value_count :]
        backward = chunk_artifact(pair.backward, (*saved, ChunkAxis(1)), rows, config)
        tangent = torch.randn_like(expected[0])
        full_gradients = pair.backward.graph_module(
            *expected[-pair.saved_value_count :], tangent
        )
        gradients = executable(backward)(*actual[-pair.saved_value_count :], tangent)
        torch.testing.assert_close(gradients, full_gradients, rtol=1e-10, atol=1e-10)


def test_sequence_softmax_is_refused():
    x = torch.randn(17, 5, requires_grad=True)
    graph = make_fx(lambda x: x.softmax(dim=0))(x)
    pair = capture_graph_pair(graph, (x,), original_output=x.softmax(0))
    with pytest.raises(CaptureError, match="mixes rows"):
        chunk_artifact(pair.forward, (ChunkAxis(0),), 17, AxisChunks(7, {0: 0}))


@pytest.mark.parametrize("budget", [0.0, 1.0])
@pytest.mark.parametrize("batch", [1, 2])
def test_layer_norm_affine_gradients_match_full_arbitrary_vjp(batch, budget):
    torch.manual_seed(731)
    rows = 17
    inputs = (
        torch.randn(batch, rows, 5, dtype=torch.float64, requires_grad=True),
        torch.randn(5, dtype=torch.float64, requires_grad=True),
        torch.randn(5, dtype=torch.float64, requires_grad=True),
    )

    def region(x, weight, bias):
        return torch.nn.functional.layer_norm(x, (5,), weight, bias)

    pair = capture_graph_pair(
        make_fx(region)(*inputs),
        inputs,
        original_output=region(*inputs),
        memory_budget=budget,
    )
    config = AxisChunks(7, {0: 1})
    forward = chunk_artifact(pair.forward, (ChunkAxis(1), None, None), rows, config)
    with torch.no_grad():
        expected = pair.forward.graph_module(*inputs)
        actual = executable(forward)(*inputs)
        torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)
        saved = forward.chunking.outputs[-pair.saved_value_count :]
        backward = chunk_artifact(pair.backward, (*saved, ChunkAxis(1)), rows, config)
        tangent = torch.randn_like(expected[0])
        gradients = executable(backward)(*actual[-pair.saved_value_count :], tangent)
        full = pair.backward.graph_module(*expected[-pair.saved_value_count :], tangent)
        torch.testing.assert_close(gradients, full, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("budget", [0.0, 1.0])
def test_shared_statistic_cotangent_broadcasts_into_chunk_rows(batch, budget):
    rows = 17
    x = torch.randn(batch, rows, 5, dtype=torch.float64, requires_grad=True)

    def region(x):
        probabilities = x.reshape(-1, 5).softmax(-1)
        return probabilities.sin(), probabilities.sum(0)

    pair = capture_graph_pair(
        make_fx(region)(x), (x,), original_output=region(x), memory_budget=budget
    )
    config = AxisChunks(7, {0: 1})
    forward = chunk_artifact(pair.forward, (ChunkAxis(1),), rows, config)
    with torch.no_grad():
        expected = pair.forward.graph_module(x)
        actual = executable(forward)(x)
        torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)
        saved = forward.chunking.outputs[-pair.saved_value_count :]
        backward = chunk_artifact(
            pair.backward, (*saved, ChunkAxis(0, batch), None), rows, config
        )
        tangents = tuple(torch.randn_like(y) for y in expected[:2])
        gradients = executable(backward)(*actual[-pair.saved_value_count :], *tangents)
        full = pair.backward.graph_module(
            *expected[-pair.saved_value_count :], *tangents
        )
        torch.testing.assert_close(gradients, full, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("budget", [0.0, 1.0])
def test_shared_output_cotangent_contributes_once_to_a_summed_weight_gradient(budget):
    torch.manual_seed(731)
    rows = 17
    inputs = (
        torch.randn(2, rows, 5, dtype=torch.float64, requires_grad=True),
        torch.randn(5, 9, dtype=torch.float64, requires_grad=True),
    )

    def region(x, weight):
        return x @ weight, weight.sin()

    pair = capture_graph_pair(
        make_fx(region)(*inputs),
        inputs,
        original_output=region(*inputs),
        memory_budget=budget,
    )
    config = AxisChunks(7, {0: 1})
    forward = chunk_artifact(pair.forward, (ChunkAxis(1), None), rows, config)
    with torch.no_grad():
        expected = pair.forward.graph_module(*inputs)
        actual = executable(forward)(*inputs)
        torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)
        saved = forward.chunking.outputs[-pair.saved_value_count :]
        backward = chunk_artifact(
            pair.backward, (*saved, ChunkAxis(1), "sum"), rows, config
        )
        tangents = tuple(torch.randn_like(value) for value in expected[:2])
        gradients = executable(backward)(*actual[-pair.saved_value_count :], *tangents)
        full = pair.backward.graph_module(
            *expected[-pair.saved_value_count :], *tangents
        )
        torch.testing.assert_close(gradients, full, rtol=1e-10, atol=1e-10)
