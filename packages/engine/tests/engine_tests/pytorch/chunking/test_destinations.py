"""Contributions separated by execution cuts share one gradient destination."""

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx

from engine_tests.pytorch.chunking.test_regions import (
    RegionChunks,
    region_executable,
)
from shadowspill.pytorch.capture.artifacts import GraphArtifact, TaskInputProvenance
from shadowspill.pytorch.capture.gradients import accumulate_gradient_outputs
from shadowspill.pytorch.chunking import ChunkRegions
from shadowspill.pytorch.chunking.regions import region_artifact
from shadowspill.pytorch.operations import ACCUMULATE_MATMUL
from shadowspill.pytorch.partition import IsolateOperations
from shadowspill.task.inputs import TaskInputRole


@pytest.mark.parametrize("prior", [False, True])
@pytest.mark.parametrize(
    "device,compiled",
    [
        ("cpu", False),
        pytest.param("cuda", False, marks=pytest.mark.cuda),
        pytest.param("cuda", True, marks=pytest.mark.cuda),
    ],
)
def test_separate_projection_contributions_reuse_one_buffer(prior, device, compiled):
    torch.manual_seed(731)
    dtype = torch.float64 if device == "cpu" else torch.float32
    inputs = tuple(
        torch.randn(*shape, device=device, dtype=dtype)
        for shape in ((17, 5), (17, 9), (17, 5), (17, 9))
    )

    def backward(x, dy, z, dz):
        first = x.T @ dy
        through_boundary = dz.sin()
        return (first + z.T @ through_boundary,)

    with FakeTensorMode() as mode:
        examples = tuple(mode.from_tensor(value) for value in inputs)
        base = GraphArtifact.capture(
            kind="backward",
            graph_module=make_fx(backward)(*examples),
            example_inputs=examples,
            input_provenance=tuple(
                TaskInputProvenance(role)
                for role in (TaskInputRole.ACTIVATION, TaskInputRole.TANGENT) * 2
            ),
        )
    artifact = accumulate_gradient_outputs(base, (0,)) if prior else base
    initial = torch.randn(5, 9, dtype=dtype, device=device)
    args = (*inputs, initial) if prior else inputs
    expected = backward(*inputs)[0] + (initial if prior else 0)
    composed = region_artifact(
        artifact,
        ChunkRegions(
            IsolateOperations(("aten.sin.default",), base="whole"), RegionChunks(17)
        ),
        stage_id="shared_projection",
        variant="recompute",
    )
    if compiled:
        from shadowspill.build.common import compile_artifacts
        from shadowspill.pytorch.executable import restore_executable

        built = compile_artifacts([composed], torch.device("cuda:0"))[
            composed.compatibility_digest
        ]
        function = restore_executable(built.payload, dependencies=built.dependencies)
    else:
        function = region_executable(composed)
    owners = []
    for index, child in function.children.copy().items():

        def observed(*args, child=child):
            values = child(*args)
            owners.extend(
                value.data_ptr()
                for value in values
                if isinstance(value, torch.Tensor) and value.shape == (5, 9)
            )
            return values

        function.children[index] = observed
    (actual,) = function(*args)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    if prior:
        assert actual.data_ptr() == initial.data_ptr()
    else:
        assert owners and all(pointer == actual.data_ptr() for pointer in owners)
    if device == "cuda":
        # The second projection writes into the destination established by
        # the first; an existing prior makes both projections accumulate.
        targets = [
            node.target
            for child in composed.regions.children
            for node in child.graph_module.graph.nodes
        ]
        assert targets.count(ACCUMULATE_MATMUL) == (2 if prior else 1)
        assert torch.ops.aten.add_.Tensor not in targets
        assert torch.ops.aten.add.Tensor not in targets


@pytest.mark.cuda
def test_region_compilation_keeps_retained_host_input_off_device():
    from shadowspill.build.common import compile_artifacts
    from shadowspill.pytorch.executable import restore_executable

    x = torch.randn(17, 5, device="cuda")
    control = torch.tensor([0, 8, 17], dtype=torch.int64, device="cpu")

    def forward(x, control):
        return x.sin(), control

    with FakeTensorMode() as mode:
        examples = tuple(mode.from_tensor(value) for value in (x, control))
        artifact = GraphArtifact.capture(
            kind="forward",
            graph_module=make_fx(forward)(*examples),
            example_inputs=examples,
        )
    composed = region_artifact(
        artifact,
        ChunkRegions("whole", RegionChunks(17)),
        stage_id="host_retention",
        variant="recompute",
    )
    compiled = compile_artifacts([composed], torch.device("cuda:0"))[
        composed.compatibility_digest
    ]
    assert compiled.memory["off_device_output_leaves"] == (1,)
    result = restore_executable(compiled.payload, dependencies=compiled.dependencies)(
        x, control
    )
    torch.testing.assert_close(result[0], x.sin())
    assert result[1].device.type == "cpu" and result[1].data_ptr() == control.data_ptr()


def test_a_gradient_contribution_returned_separately_is_not_donated():
    inputs = tuple(
        torch.randn(*shape, dtype=torch.float64)
        for shape in ((17, 5), (17, 9), (17, 9))
    )

    def backward(x, dy, dz):
        shared = x.T @ dy
        return shared, shared + x.T @ dz

    with FakeTensorMode() as mode:
        examples = tuple(mode.from_tensor(value) for value in inputs)
        artifact = GraphArtifact.capture(
            kind="backward",
            graph_module=make_fx(backward)(*examples),
            example_inputs=examples,
        )
    composed = region_artifact(
        artifact,
        ChunkRegions("whole", RegionChunks(17)),
        stage_id="observed_contribution",
        variant="recompute",
    )
    actual = region_executable(composed)(*inputs)
    torch.testing.assert_close(actual, backward(*inputs), rtol=1e-12, atol=1e-12)
    assert actual[0].data_ptr() != actual[1].data_ptr()
    assert torch.ops.aten.add.Tensor in {
        n.target for n in composed.regions.children[0].graph_module.graph.nodes
    }


def test_parameter_gradient_routing_stops_at_activation_adjoint_producers():
    inputs = tuple(torch.randn(17, 5, dtype=torch.float64) for _ in range(3))

    def backward(x, dy, dz):
        activation_gradient = dy + dz
        return activation_gradient, x.T @ activation_gradient

    with FakeTensorMode() as mode:
        examples = tuple(mode.from_tensor(value) for value in inputs)
        artifact = GraphArtifact.capture(
            kind="backward",
            graph_module=make_fx(backward)(*examples),
            example_inputs=examples,
        )
    composed = region_artifact(
        artifact,
        ChunkRegions("whole", RegionChunks(17)),
        stage_id="activation_adjoint",
        variant="recompute",
        gradient_leaves=(1,),
    )
    torch.testing.assert_close(
        region_executable(composed)(*inputs), backward(*inputs), rtol=1e-12, atol=1e-12
    )
    targets = {n.target for n in composed.regions.children[0].graph_module.graph.nodes}
    assert torch.ops.aten.add.Tensor in targets
    assert torch.ops.aten.add_.Tensor not in targets
