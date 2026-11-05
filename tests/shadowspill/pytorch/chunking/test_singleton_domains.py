"""Singleton tails keep their logical position through packed-head views."""

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from shadowspill.pytorch.capture.aot import capture_graph_pair
from shadowspill.pytorch.chunking import ChunkAxis
from tests.shadowspill.pytorch.chunking.test_composition import compose, executable


@pytest.mark.parametrize("batch,extent", [(1, 1), (1, 257), (2, 257)])
@pytest.mark.parametrize("save", [False, True])
def test_packed_projection_singleton_tail_vjp(batch, extent, save):
    torch.manual_seed(731)
    inputs = (
        torch.randn(batch, extent, 16, 32, dtype=torch.float64, requires_grad=True),
        torch.randn(512, 7, dtype=torch.float64, requires_grad=True),
    )

    def projection(x, weight):
        # An ordinary head-major representation, followed by a token projection.
        # The extent-one tail still has a position between batch and features.
        packed = x.permute(2, 0, 1, 3).reshape(16, -1, 32)
        tokens = packed.permute(1, 0, 2).reshape(batch, -1, 512)
        return tokens @ weight

    expected = projection(*inputs)
    tangent = torch.randn_like(expected)
    expected_gradients = torch.autograd.grad(expected, inputs, tangent)
    pair = capture_graph_pair(
        make_fx(projection)(*inputs),
        inputs,
        original_output=expected,
        memory_budget=float(save),
    )
    forward = compose(pair.forward, (ChunkAxis(1), None), extent, size=128)
    with torch.no_grad():
        actual = executable(forward)(*inputs)
        torch.testing.assert_close(actual[0], expected, rtol=1e-11, atol=1e-11)
        saved = forward.chunking.outputs[-pair.saved_value_count :]
        backward = compose(pair.backward, (*saved, ChunkAxis(1)), extent, size=128)
        gradients = executable(backward)(*actual[-pair.saved_value_count :], tangent)
    torch.testing.assert_close(gradients, expected_gradients, rtol=1e-10, atol=1e-10)
