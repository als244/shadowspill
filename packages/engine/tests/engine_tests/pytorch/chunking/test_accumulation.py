"""Gradient buffers span chunks and microbatches, including mixed creation."""

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx

from engine_tests.pytorch.chunking.test_axis_vjp import executable
from shadowspill.pytorch.capture.artifacts import GraphArtifact
from shadowspill.pytorch.capture.gradients import accumulate_gradient_outputs
from shadowspill.pytorch.chunking import AxisChunks, ChunkAxis
from shadowspill.pytorch.chunking.transform import chunk_artifact
from shadowspill.pytorch.operations import ACCUMULATE_MATMUL


def test_nested_accumulation_names_do_not_collide_with_retained_prior_arguments():
    def backward(x, prior):
        return prior + x.sum(0), x.T @ x

    x, prior = torch.randn(17, 5), torch.randn(5)
    graph = make_fx(backward)(x, prior)
    arguments = [node for node in graph.graph.nodes if node.op == "placeholder"]
    arguments[1].target = "shadowspill_prior_grad_1"
    graph.recompile()
    artifact = GraphArtifact.capture(
        kind="backward", graph_module=graph, example_inputs=(x, prior)
    )
    accumulating = accumulate_gradient_outputs(artifact, (1,))
    running = torch.randn(5, 5)
    expected = (prior + x.sum(0), running + x.T @ x)
    actual = accumulating.graph_module(x, prior, running)
    torch.testing.assert_close(actual, expected)
    assert actual[1].data_ptr() == running.data_ptr()
    names = [
        node.target
        for node in accumulating.graph_module.graph.nodes
        if node.op == "placeholder"
    ]
    assert len(names) == len(set(names))


@pytest.mark.parametrize(
    "rows,size", [(3, 7), (7, 7), (10, 7), (14, 7), (17, 7), (3, 1)]
)
@pytest.mark.parametrize("existing", [(), (1,), (1, 2)])
def test_first_contribution_owns_gradients_and_all_later_chunks_reuse_them(
    rows, size, existing
):
    torch.manual_seed(731)
    x, dy, weight = (
        torch.randn(*shape, dtype=torch.float64)
        for shape in ((rows, 5), (rows, 9), (5, 9))
    )

    def backward(x, dy, weight):
        return dy @ weight.T, x.T @ dy, dy.sum(0)

    with FakeTensorMode() as mode:
        examples = tuple(mode.from_tensor(value) for value in (x, dy, weight))
        base = GraphArtifact.capture(
            kind="backward",
            graph_module=make_fx(backward)(*examples),
            example_inputs=examples,
        )
    artifact = accumulate_gradient_outputs(base, existing)
    prior = [torch.randn_like(backward(x, dy, weight)[leaf]) for leaf in existing]
    expected = list(backward(x, dy, weight))
    for leaf, value in zip(existing, prior, strict=True):
        expected[leaf] += value
    composed = chunk_artifact(
        artifact,
        (ChunkAxis(0), ChunkAxis(0), None, *(None for _ in prior)),
        rows,
        AxisChunks(size, {0: 0, 1: 0}),
    )
    function = executable(composed)
    pointers = {
        leaf: [value.data_ptr()] for leaf, value in zip(existing, prior, strict=True)
    }
    calls = []

    def observe(child, mode, count):
        def run(*args):
            result = child(*args)
            calls.append((mode, count))
            for leaf in (1, 2):
                pointers.setdefault(leaf, []).append(result[leaf].data_ptr())
            return result

        return run

    for mode, children in (
        ("first", function.children),
        ("accumulate", function.accumulating_children),
    ):
        for count, child in children.copy().items():
            children[count] = observe(child, mode, count)
    result = function(x, dy, weight, *prior)
    torch.testing.assert_close(result, tuple(expected), rtol=1e-12, atol=1e-12)
    for leaf in (1, 2):
        assert set(pointers[leaf]) == {result[leaf].data_ptr()}
    if function.accumulation_leaves:
        assert calls[0] == ("first", size)
        assert all(mode == "accumulate" for mode, _ in calls[1:])
        assert set(count for mode, count in calls if mode == "accumulate") == set(
            function.accumulating_children
        )

    # A following microbatch arrives with both buffers, so even its first
    # chunk must mutate those inputs instead of creating fresh gradients.
    accumulating = accumulate_gradient_outputs(base, (1, 2))
    following = chunk_artifact(
        accumulating,
        (ChunkAxis(0), ChunkAxis(0), None, None, None),
        rows,
        AxisChunks(size, {0: 0, 1: 0}),
    )
    assert not following.chunking.accumulating_children
    for leaf in (1, 2):
        expected[leaf] += backward(x, dy, weight)[leaf]
    final = executable(following)(x, dy, weight, result[1], result[2])
    torch.testing.assert_close(final, tuple(expected), rtol=1e-12, atol=1e-12)
    assert all(final[leaf].data_ptr() == result[leaf].data_ptr() for leaf in (1, 2))
    assert {
        m.replacement_output_leaf
        for m in following.storage_contract.mutations
        if m.replacement_output_leaf is not None
    } == {1, 2}


@pytest.mark.cuda
@pytest.mark.parametrize("fuse", [False, True])
def test_later_chunk_gemms_honor_narrow_accumulation_fusion(fuse):
    x = torch.randn(17, 5, device="cuda", dtype=torch.bfloat16)
    dy = torch.randn(17, 9, device="cuda", dtype=torch.bfloat16)

    def backward(x, dy):
        return (x.T @ dy,)

    with FakeTensorMode() as mode:
        examples = tuple(mode.from_tensor(value) for value in (x, dy))
        artifact = GraphArtifact.capture(
            kind="backward",
            graph_module=make_fx(backward)(*examples),
            example_inputs=examples,
        )
    chunked = chunk_artifact(
        artifact,
        (ChunkAxis(0), ChunkAxis(0)),
        17,
        AxisChunks(7, {0: 0, 1: 0}),
        fuse_narrow_gradient_accumulation=fuse,
    )
    assert set(dict(chunked.chunking.children)) == {7}
    assert set(dict(chunked.chunking.accumulating_children)) == {3, 7}
    for _, child in chunked.chunking.accumulating_children:
        targets = {node.target for node in child.graph_module.graph.nodes}
        assert (ACCUMULATE_MATMUL in targets) is fuse
    (actual,) = executable(chunked)(x, dy)
    expected = sum(
        x[start : start + 7].T @ dy[start : start + 7] for start in range(0, 17, 7)
    )
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.125)
