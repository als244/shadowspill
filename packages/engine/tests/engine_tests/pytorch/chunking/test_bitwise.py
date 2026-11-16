"""Bitwise masks keep row coordinates, including tails and shared broadcasts."""

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from engine_tests.pytorch.chunking.test_axis_vjp import executable
from engine_tests.pytorch.chunking.test_contracts import artifact
from shadowspill.errors import CaptureError
from shadowspill.pytorch.capture.aot import capture_graph_pair
from shadowspill.pytorch.capture.export import capture_forward, inference_artifact
from shadowspill.pytorch.chunking import AxisChunks, ChunkAxis
from shadowspill.pytorch.chunking.transform import chunk_artifact


@pytest.mark.parametrize("name", ["__or__", "__and__", "__xor__"])
@pytest.mark.parametrize("dtype", [torch.bool, torch.int64])
def test_forward_export_bitwise_aliases_match_whole(name, dtype):
    class Masks(torch.nn.Module):
        def forward(self, left, right):
            operation = getattr(torch.ops.aten, name)
            return operation.Tensor(left, right), operation.Scalar(left, 1)

    torch.manual_seed(731)
    values = tuple(
        torch.randint(0, 2 if dtype == torch.bool else 8, shape).to(dtype)
        for shape in ((2, 17, 5), (2, 17, 1))
    )
    model = Masks()
    full = inference_artifact(capture_forward(model, values))
    assert f"aten.{name}.Tensor" in full.operator_targets
    chunked = chunk_artifact(
        full, (ChunkAxis(1), ChunkAxis(1)), 17, AxisChunks(7, {0: 1, 1: 1})
    )
    torch.testing.assert_close(
        executable(chunked)(*values), model(*values), rtol=0, atol=0
    )


@pytest.mark.parametrize(
    "operation", [torch.bitwise_or, torch.bitwise_and, torch.bitwise_xor]
)
@pytest.mark.parametrize("dtype", [torch.bool, torch.int64])
def test_bitwise_rows_and_shared_broadcasts_match_whole(operation, dtype):
    generator = torch.Generator().manual_seed(731)
    values = tuple(
        torch.randint(
            0, 2 if dtype == torch.bool else 8, shape, generator=generator
        ).to(dtype)
        for shape in ((2, 17, 5), (2, 17, 1), (5,))
    )

    def region(left, right, shared):
        return operation(left, right), operation(left, shared), operation(left, 1)

    full = artifact(region, values)
    chunked = chunk_artifact(
        full, (ChunkAxis(1), ChunkAxis(1), None), 17, AxisChunks(7, {0: 1, 1: 1})
    )
    torch.testing.assert_close(
        executable(chunked)(*values), region(*values), rtol=0, atol=0
    )


@pytest.mark.parametrize(
    "operation", [torch.bitwise_or, torch.bitwise_and, torch.bitwise_xor]
)
def test_bitwise_misaligned_domains_are_refused(operation):
    values = (
        torch.ones(34, 5, dtype=torch.int64),
        torch.zeros(34, 5, dtype=torch.int64),
    )
    full = artifact(lambda a, b: (operation(a, b),), values)
    with pytest.raises(CaptureError, match="differently aligned chunk domains"):
        chunk_artifact(
            full,
            (ChunkAxis(0, before=2), ChunkAxis(0, after=2)),
            17,
            AxisChunks(7, {0: ChunkAxis(0, before=2), 1: ChunkAxis(0, after=2)}),
        )


@pytest.mark.parametrize(
    "operation", [torch.bitwise_or, torch.bitwise_and, torch.bitwise_xor]
)
def test_bitwise_of_partial_reductions_is_refused(operation):
    x = torch.ones(17, 5, dtype=torch.int64)
    full = artifact(lambda x: (operation(x.sum(0), 1),), (x,))
    with pytest.raises(CaptureError, match="nonlinearly transforms a row reduction"):
        chunk_artifact(full, (ChunkAxis(0),), 17, AxisChunks(7, {0: 0}))


@pytest.mark.parametrize("budget", [0.0, 1.0])
def test_combined_padding_and_prediction_masks_preserve_vjp(budget):
    torch.manual_seed(731)
    inputs = (
        torch.randn(2, 17, 5, dtype=torch.float64, requires_grad=True),
        torch.randn(5, 9, dtype=torch.float64, requires_grad=True),
        torch.rand(2, 17, 1) > 0.3,
        torch.rand(2, 17, 1) > 0.5,
    )

    def region(x, weight, padding, prediction):
        mask = (torch.zeros_like(padding) | padding) & prediction
        return torch.where(mask, (x @ weight).sin(), 0.0)

    output = region(*inputs)
    tangent = torch.randn_like(output)
    expected = torch.autograd.grad(output, inputs[:2], tangent)
    pair = capture_graph_pair(
        make_fx(region)(*inputs), inputs, original_output=output, memory_budget=budget
    )
    config = AxisChunks(7, {0: 1, 2: 1, 3: 1})
    forward = chunk_artifact(
        pair.forward, (ChunkAxis(1), None, ChunkAxis(1), ChunkAxis(1)), 17, config
    )
    saved = forward.chunking.outputs[-pair.saved_value_count :]
    backward = chunk_artifact(pair.backward, (*saved, ChunkAxis(1)), 17, config)
    with torch.no_grad():
        actual = executable(forward)(*inputs)
        gradients = executable(backward)(*actual[-pair.saved_value_count :], tangent)
    torch.testing.assert_close(actual[0], output, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(
        gradients, (*expected, None, None), rtol=1e-11, atol=1e-11
    )
