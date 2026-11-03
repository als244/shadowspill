"""Odd distributed gradient storage survives chunk and microbatch accumulation."""

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx

from shadowspill.errors import CaptureError
from shadowspill.pytorch.capture.artifacts import GraphArtifact
from shadowspill.pytorch.capture.gradients import accumulate_gradient_outputs
from shadowspill.pytorch.capture.padding import pad_output_storage
from shadowspill.pytorch.chunking import AxisChunks, ChunkAxis
from shadowspill.pytorch.chunking.transform import chunk_artifact
from shadowspill.pytorch.operations import ACCUMULATE_MATMUL
from shadowspill.pytorch.operators.materialize import PAD_STORAGE
from tests.shadowspill.pytorch.chunking.test_axis_vjp import executable


@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
)
def test_padded_gradient_destination_is_reused(transpose, existing, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires an accelerator")
    torch.manual_seed(731)
    dtype = torch.float64 if device == "cpu" else torch.float32
    tolerance = 1e-12 if device == "cpu" else 3e-5
    x = torch.randn(17, 5, dtype=dtype, device=device)
    dy = torch.randn(17, 9, dtype=dtype, device=device)
    padded_bytes = 46 * x.element_size()

    def backward(x, dy):
        result = x.T @ dy
        return (result.T if transpose else result,)

    with FakeTensorMode() as mode:
        args = tuple(mode.from_tensor(value) for value in (x, dy))
        base = GraphArtifact.capture(
            kind="backward", graph_module=make_fx(backward)(*args), example_inputs=args
        )
    base = pad_output_storage(base, {0: 2})
    assert base.storage_contract.roots[0].minimum_span_bytes == padded_bytes
    artifact = accumulate_gradient_outputs(base, (0,)) if existing else base
    prior = PAD_STORAGE(torch.randn_like(backward(x, dy)[0]), 46)
    expected = backward(x, dy)[0] + (prior if existing else 0)
    chunked = chunk_artifact(
        artifact,
        (ChunkAxis(0), ChunkAxis(0), *((None,) if existing else ())),
        17,
        AxisChunks(7, {0: 0, 1: 0}),
    )
    runner = executable(chunked)
    children = runner.children if existing else runner.accumulating_children
    calls = []
    for size, child in children.copy().items():
        targets = {node.target for node in child.graph.nodes}
        if device == "cuda":
            assert ACCUMULATE_MATMUL in targets
            assert PAD_STORAGE not in targets

        def observe(*args, child=child):
            result = child(*args)
            calls.append(result[0].data_ptr())
            return result

        children[size] = observe
    actual = runner(x, dy, *((prior,) if existing else ()))[0]
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    assert actual.untyped_storage().nbytes() == padded_bytes
    assert set(calls) == {actual.data_ptr()}
    if existing:
        assert actual.data_ptr() == prior.data_ptr()

    following = chunk_artifact(
        accumulate_gradient_outputs(base, (0,)),
        (ChunkAxis(0), ChunkAxis(0), None),
        17,
        AxisChunks(7, {0: 0, 1: 0}),
    )
    expected = expected + backward(x, dy)[0]
    result = executable(following)(x, dy, actual)[0]
    torch.testing.assert_close(result, expected, rtol=tolerance, atol=tolerance)
    assert result.data_ptr() == actual.data_ptr()
    assert result.untyped_storage().nbytes() == padded_bytes


def test_fixed_padding_refuses_a_sliced_activation():
    x = torch.randn(17, 5)

    def forward(x):
        return (PAD_STORAGE(x, 86),)

    artifact = GraphArtifact.capture(
        kind="inference", graph_module=make_fx(forward)(x), example_inputs=(x,)
    )
    with pytest.raises(CaptureError, match="fixed storage padding"):
        chunk_artifact(artifact, (ChunkAxis(0),), 17, AxisChunks(7, {0: 0}))
