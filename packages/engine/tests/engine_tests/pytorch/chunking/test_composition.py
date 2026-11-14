"""Bounded tiles compose independently sized execution regions without new saves."""

import weakref

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx

from shadowspill.pytorch.capture.aot import capture_graph_pair
from shadowspill.pytorch.capture.artifacts import GraphArtifact
from shadowspill.pytorch.capture.gradients import accumulate_gradient_outputs
from shadowspill.pytorch.chunking import AxisChunks, ChunkAxis, ChunkRegions
from shadowspill.pytorch.chunking.executable import ChunkedExecutable
from shadowspill.pytorch.chunking.region_executable import RegionExecutable
from shadowspill.pytorch.chunking.transform import chunk_artifact
from shadowspill.pytorch.partition import IsolateOperations


class Inner:
    def configure_stage(self, stage):
        axes = {
            i: domains["positions"]
            for i, domains in enumerate(stage.input_domains)
            if "positions" in domains
        }
        size = 5 if any("silu" in op for op in stage.operators) else 3
        return AxisChunks(size, axes, domain="positions") if axes else None


def compose(artifact, inputs, extent, size=11):
    config = AxisChunks(
        size,
        {0: 0},
        domain="positions",
        regions=ChunkRegions(
            IsolateOperations(
                ("aten.silu.default", "aten.silu_backward.default"), base="whole"
            ),
            Inner(),
        ),
    )
    return chunk_artifact(artifact, inputs, extent, config)


def executable(artifact):
    if artifact.regions:
        return RegionExecutable(
            artifact.regions.identity,
            {i: executable(child) for i, child in enumerate(artifact.regions.children)},
        )
    if artifact.chunking:
        task = artifact.chunking
        return ChunkedExecutable(
            {
                **task.identity,
                "storage": task.storage_contract.to_dict(),
                "output_count": artifact.output_count,
                "input_offsets": [x.storage_offset for x in artifact.tensor_inputs],
            },
            {size: executable(child) for size, child in task.children},
            {size: executable(child) for size, child in task.accumulating_children},
        )
    return artifact.graph_module


@pytest.mark.parametrize("batch,extent,axis", [(1, 2, 1), (2, 29, 1), (3, 22, 0)])
@pytest.mark.parametrize("save", [False, True])
@pytest.mark.parametrize("lora", [False, True])
def test_independent_vjp(batch, extent, axis, save, lora):
    torch.manual_seed(731)
    shape = (batch, extent, 5) if axis == 1 else (extent, batch, 5)
    inputs = (
        torch.randn(shape, dtype=torch.float64, requires_grad=True),
        torch.randn(5, 9, dtype=torch.float64, requires_grad=not lora),
        torch.randn(5, 3, dtype=torch.float64, requires_grad=True),
        torch.randn(3, 9, dtype=torch.float64, requires_grad=True),
        torch.randn(9, 5, dtype=torch.float64, requires_grad=True),
    )

    def layer(x, w, a, b, out):
        projected = x @ w + (x @ a) @ b
        return torch.nn.functional.silu(projected) @ out + x

    reference = layer(*inputs)
    tangent = torch.randn_like(reference)
    expected_gradients = torch.autograd.grad(
        reference, [x for x in inputs if x.requires_grad], tangent
    )
    pair = capture_graph_pair(
        make_fx(layer)(*inputs),
        inputs,
        original_output=layer(*inputs),
        memory_budget=float(save),
    )
    forward = compose(pair.forward, (ChunkAxis(axis), None, None, None, None), extent)
    with torch.no_grad():
        actual = executable(forward)(*inputs)
        torch.testing.assert_close(actual[0], reference, rtol=1e-11, atol=1e-11)
        saved = forward.chunking.outputs[-pair.saved_value_count :]
        backward = compose(pair.backward, (*saved, ChunkAxis(axis)), extent)
        gradients = executable(backward)(*actual[-pair.saved_value_count :], tangent)
    torch.testing.assert_close(
        tuple(x for x in gradients if x is not None),
        expected_gradients,
        rtol=1e-10,
        atol=1e-10,
    )


@pytest.mark.parametrize("prior_leaves", [(), (1,), (1, 2)])
def test_nested_chunks_reuse_gradient_destinations_across_microbatches(prior_leaves):
    torch.manual_seed(731)
    extent = 29
    x, dy, weight = (
        torch.randn(shape, dtype=torch.float64)
        for shape in ((extent, 5), (extent, 9), (5, 9))
    )

    def backward(x, dy, weight):
        activated = torch.nn.functional.silu(x)
        return dy @ weight.T, activated.T @ dy, dy.sum(0)

    with FakeTensorMode() as mode:
        examples = tuple(mode.from_tensor(value) for value in (x, dy, weight))
        base = GraphArtifact.capture(
            kind="backward",
            graph_module=make_fx(backward)(*examples),
            example_inputs=examples,
        )
    artifact = accumulate_gradient_outputs(base, prior_leaves)
    expected = list(backward(x, dy, weight))
    priors = [torch.randn_like(expected[leaf]) for leaf in prior_leaves]
    for leaf, prior in zip(prior_leaves, priors, strict=True):
        expected[leaf] += prior
    layouts = (ChunkAxis(0), ChunkAxis(0), None, *(None for _ in priors))
    nested = compose(artifact, layouts, extent)
    runner = executable(nested)
    pointers = {leaf: [] for leaf in (1, 2)}

    def observe(child):
        def run(*args):
            values = child(*args)
            for leaf in pointers:
                pointers[leaf].append(values[leaf].data_ptr())
            return values

        return run

    for children in (runner.children, runner.accumulating_children):
        for size, child in children.copy().items():
            children[size] = observe(child)
    actual = runner(x, dy, weight, *priors)
    torch.testing.assert_close(actual, tuple(expected), rtol=1e-12, atol=1e-12)
    for leaf in (1, 2):
        assert set(pointers[leaf]) == {actual[leaf].data_ptr()}
    for leaf, prior in zip(prior_leaves, priors, strict=True):
        assert actual[leaf].data_ptr() == prior.data_ptr()

    next_artifact = accumulate_gradient_outputs(base, (1, 2))
    following = compose(
        next_artifact, (ChunkAxis(0), ChunkAxis(0), None, None, None), extent
    )
    contributions = backward(x, dy, weight)
    for leaf in (1, 2):
        expected[leaf] += contributions[leaf]
    final = executable(following)(x, dy, weight, actual[1], actual[2])
    torch.testing.assert_close(final, tuple(expected), rtol=1e-12, atol=1e-12)
    assert all(final[i].data_ptr() == actual[i].data_ptr() for i in (1, 2))


@pytest.mark.parametrize("prior", [False, True])
def test_owned_activation_gradient_crosses_an_inner_cut_with_returned_view(prior):
    from shadowspill.pytorch.chunking.regions import region_artifact

    torch.manual_seed(129)
    inputs = tuple(torch.randn(29, 5, dtype=torch.float64) for _ in range(2))

    def backward(x, dy):
        first = dy * x
        activated = torch.nn.functional.silu(dy)
        return (first + activated * x).view(1, 29, 5), x.T @ activated

    with FakeTensorMode() as mode:
        examples = tuple(mode.from_tensor(value) for value in inputs)
        base = GraphArtifact.capture(
            kind="backward",
            graph_module=make_fx(backward)(*examples),
            example_inputs=examples,
        )
    artifact = accumulate_gradient_outputs(base, (1,)) if prior else base

    class Outer:
        def configure_stage(self, stage):
            return AxisChunks(
                11,
                {0: 0, 1: 0},
                domain="positions",
                regions=ChunkRegions(
                    IsolateOperations(("aten.silu.default",), base="whole"), Inner()
                ),
            )

    nested = region_artifact(
        artifact,
        ChunkRegions(IsolateOperations((), base="whole"), Outer()),
        stage_id="layer",
        variant="recompute",
    )
    initial = torch.randn(5, 5, dtype=torch.float64)
    expected = list(backward(*inputs))
    if prior:
        expected[1] += initial
    actual = executable(nested)(*inputs, *((initial,) if prior else ()))
    torch.testing.assert_close(actual, tuple(expected), rtol=1e-12, atol=1e-12)
    if prior:
        assert actual[1].data_ptr() == initial.data_ptr()


def test_shared_cotangent_broadcast_keeps_its_axis_across_inner_cuts():
    torch.manual_seed(817)
    inputs = tuple(
        torch.randn(shape, dtype=torch.float64) for shape in ((29, 5), (29, 5), (5,))
    )

    def backward(x, dy, statistics_tangent):
        broadcast = statistics_tangent.unsqueeze(0).expand(29, 5)
        activated = torch.nn.functional.silu(x)
        return activated * broadcast + dy, x.T @ dy

    with FakeTensorMode() as mode:
        examples = tuple(mode.from_tensor(value) for value in inputs)
        base = GraphArtifact.capture(
            kind="backward",
            graph_module=make_fx(backward)(*examples),
            example_inputs=examples,
        )
    nested = compose(base, (ChunkAxis(0), ChunkAxis(0), None), 29)
    torch.testing.assert_close(
        executable(nested)(*inputs), backward(*inputs), rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize("save", [False, True])
def test_nested_shared_output_cotangent_contributes_once(save):
    x = torch.randn(2, 29, 5, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(5, 9, dtype=torch.float64, requires_grad=True)

    def layer(x, weight):
        return torch.nn.functional.silu(x @ weight), weight.sin()

    expected = layer(x, weight)
    tangents = tuple(torch.randn_like(value) for value in expected)
    reference = torch.autograd.grad(expected, (x, weight), tangents)
    pair = capture_graph_pair(
        make_fx(layer)(x, weight),
        (x, weight),
        original_output=layer(x, weight),
        memory_budget=float(save),
    )
    forward = compose(pair.forward, (ChunkAxis(1), None), 29)
    with torch.no_grad():
        values = executable(forward)(x, weight)
        saved = forward.chunking.outputs[-pair.saved_value_count :]
        backward = compose(pair.backward, (*saved, ChunkAxis(1), "sum"), 29)
        actual = executable(backward)(*values[-pair.saved_value_count :], *tangents)
    torch.testing.assert_close(actual, reference, rtol=1e-10, atol=1e-10)


def test_tile_intermediates_die_before_the_next_tile():
    x = torch.randn(2, 29, 5)

    def layer(x):
        return torch.nn.functional.silu(x.sin()).cos()

    base = GraphArtifact.capture(
        kind="inference", graph_module=make_fx(layer)(x), example_inputs=(x,)
    )
    nested = compose(base, (ChunkAxis(1),), 29)
    runner = executable(nested)
    observed = []
    calls = []

    def observe(child, extent):
        def run(*args):
            assert all(reference() is None for reference in observed)
            calls.append(extent)
            result = child(*args)
            observed.extend(weakref.ref(value) for value in result)
            return result

        return run

    for size, child in runner.children.copy().items():
        runner.children[size] = observe(child, size)
    actual = runner(x)
    torch.testing.assert_close(actual[0], layer(x))
    assert calls == [11, 11, 7]
    assert all(reference() is None for reference in observed)
    for extent, tile in nested.chunking.children:
        assert extent <= 11
        for region in tile.regions.children:
            if region.chunking:
                assert region.chunking.rows == extent
                assert region.chunking.size in {3, 5}


def test_reduced_boundary_survives_an_unchunked_linear_region():
    x = torch.randn(29, 5, dtype=torch.float64)
    dy = torch.randn(29, 9, dtype=torch.float64)

    def backward(x, dy):
        subtotal = -(x.T @ dy)
        return subtotal, subtotal + torch.nn.functional.silu(x).T @ dy

    with FakeTensorMode() as mode:
        examples = tuple(mode.from_tensor(value) for value in (x, dy))
        base = GraphArtifact.capture(
            kind="backward",
            graph_module=make_fx(backward)(*examples),
            example_inputs=examples,
        )
    config = AxisChunks(
        11,
        {0: 0, 1: 0},
        domain="positions",
        regions=ChunkRegions(
            IsolateOperations(("aten.neg.default", "aten.silu.default"), base="whole"),
            Inner(),
        ),
    )
    nested = chunk_artifact(base, (ChunkAxis(0), ChunkAxis(0)), 29, config)
    torch.testing.assert_close(
        executable(nested)(x, dy), backward(x, dy), rtol=1e-11, atol=1e-11
    )


@pytest.mark.parametrize("prior", [False, True])
@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
)
def test_shared_projection_gradients_reuse_destinations_through_both_chunk_levels(
    prior, device
):
    from shadowspill.pytorch.operations import ACCUMULATE_MATMUL

    dtype = torch.float64 if device == "cpu" else torch.float32
    inputs = tuple(
        torch.randn(shape, device=device, dtype=dtype)
        for shape in ((29, 5), (29, 9), (29, 5), (29, 9))
    )

    def backward(x, dy, z, dz):
        first = x.T @ dy
        activated = torch.nn.functional.silu(dz)
        return (first + z.T @ activated,)

    with FakeTensorMode() as mode:
        examples = tuple(mode.from_tensor(value) for value in inputs)
        base = GraphArtifact.capture(
            kind="backward",
            graph_module=make_fx(backward)(*examples),
            example_inputs=examples,
        )
    artifact = accumulate_gradient_outputs(base, (0,)) if prior else base
    initial = torch.randn(5, 9, device=device, dtype=dtype)
    expected = backward(*inputs)[0] + (initial if prior else 0)
    nested = compose(artifact, (ChunkAxis(0),) * 4 + ((None,) if prior else ()), 29)
    if device == "cuda":
        from shadowspill.build.common import compile_artifacts
        from shadowspill.pytorch.executable import restore_executable

        compiled = compile_artifacts((nested,), torch.device("cuda"))[
            nested.compatibility_digest
        ]
        runner = restore_executable(
            compiled.payload, dependencies=compiled.dependencies
        )
        for group, products in (
            (nested.chunking.children, 2 if prior else 1),
            (nested.chunking.accumulating_children, 2),
        ):
            for _, tile in group:
                targets = [
                    node.target
                    for region in tile.regions.children
                    for node in region.graph_module.graph.nodes
                ]
                assert targets.count(ACCUMULATE_MATMUL) == products
                assert torch.ops.aten.add_.Tensor not in targets
    else:
        runner = executable(nested)
    pointers = []

    def observe(child):
        def run(*args):
            values = child(*args)
            pointers.append(values[0].data_ptr())
            return values

        return run

    for children in (runner.children, runner.accumulating_children):
        for size, child in children.copy().items():
            children[size] = observe(child)
    (actual,) = runner(*inputs, *((initial,) if prior else ()))
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    assert set(pointers) == {actual.data_ptr()}
    if prior:
        assert actual.data_ptr() == initial.data_ptr()
