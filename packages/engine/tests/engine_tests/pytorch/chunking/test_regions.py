"""Inner execution cuts preserve the outer graph pair and arbitrary VJPs."""

import weakref

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from engine_tests.pytorch.chunking.test_axis_vjp import executable
from shadowspill.errors import CaptureError
from shadowspill.pytorch.capture.aot import capture_graph_pair
from shadowspill.pytorch.chunking import AxisChunks, ChunkAxis, ChunkRegions
from shadowspill.pytorch.chunking.region_executable import RegionExecutable
from shadowspill.pytorch.chunking.regions import region_artifact
from shadowspill.pytorch.partition import IsolateOperations


class RegionChunks:
    def __init__(self, rows, batch=1):
        self.rows = rows
        self.batch = batch
        self.seen = []

    def configure_stage(self, stage):
        self.seen.append(stage)
        if any("softmax" in target for target in stage.operators):
            return None
        axes = {}
        for i, value in enumerate(stage.inputs):
            if value is None:
                continue
            if stage.input_domains and "tokens" in stage.input_domains[i]:
                axes[i] = stage.input_domains[i]["tokens"]
                continue
            for dim, extent in enumerate(value.shape):
                if extent == self.rows:
                    axes[i] = ChunkAxis(dim)
                    break
                if extent == self.batch * self.rows:
                    axes[i] = ChunkAxis(dim, before=self.batch)
                    break
        size = 7 if any("silu" in op for op in stage.operators) else 3
        return AxisChunks(size, axes, domain="tokens") if axes else None


def region_executable(artifact):
    task = artifact.regions
    return RegionExecutable(
        task.identity,
        {
            index: executable(child) if child.chunking else child.graph_module
            for index, child in enumerate(task.children)
        },
    )


@pytest.mark.parametrize("batch,rows", [(1, 2), (2, 17), (3, 14)])
@pytest.mark.parametrize("budget", [0.0, 1.0])
@pytest.mark.parametrize("lora", [False, True])
def test_regions_preserve_checkpoint_saves_and_arbitrary_gradients(
    batch, rows, budget, lora
):
    torch.manual_seed(731)
    inputs = (
        torch.randn(batch, rows, 5, dtype=torch.float64, requires_grad=True),
        torch.randn(5, 9, dtype=torch.float64, requires_grad=not lora),
        torch.randn(5, 4, dtype=torch.float64, requires_grad=True),
        torch.randn(4, 9, dtype=torch.float64, requires_grad=True),
        torch.randn(9, 5, dtype=torch.float64, requires_grad=True),
    )

    def layer(x, weight, a, b, out):
        q = (x @ weight + (x @ a) @ b).sin()
        return torch.nn.functional.silu(q.softmax(1) @ out) + x

    pair = capture_graph_pair(
        make_fx(layer)(*inputs),
        inputs,
        original_output=layer(*inputs),
        memory_budget=budget,
    )
    axis_policy = RegionChunks(rows, batch)
    policy = ChunkRegions(
        IsolateOperations(
            ("aten._softmax.default", "aten._softmax_backward_data.default"),
            base="whole",
        ),
        axis_policy,
    )
    forward, backward = (
        region_artifact(
            artifact,
            policy,
            stage_id="block",
            variant="save" if budget else "recompute",
        )
        for artifact in (pair.forward, pair.backward)
    )
    assert forward.storage_contract == pair.forward.storage_contract
    assert backward.storage_contract == pair.backward.storage_contract
    assert len(forward.regions.children) == 3
    assert {
        child.chunking.size for child in forward.regions.children if child.chunking
    } == {3, 7}
    assert all(
        info.structural_contract and info.checkpoint_stage_id == "block"
        for info in axis_policy.seen
    )
    with torch.no_grad():
        expected = pair.forward.graph_module(*inputs)
        actual = region_executable(forward)(*inputs)
        torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)
        tangent = torch.randn_like(expected[0])
        gradients = region_executable(backward)(
            *actual[-pair.saved_value_count :], tangent
        )
        reference = pair.backward.graph_module(
            *expected[-pair.saved_value_count :], tangent
        )
        torch.testing.assert_close(gradients, reference, rtol=1e-10, atol=1e-10)


def test_execution_releases_internal_values_before_the_next_region():
    observed = []

    class Child:
        def __init__(self, index):
            self.index = index

        def __call__(self, value):
            if self.index == 2:
                assert observed[0]() is None
            result = value.sin()
            observed.append(weakref.ref(result))
            return (result,)

    # Use actual region metadata, so this checks the generated last-use program.
    x = torch.randn(17, 5)
    from shadowspill.pytorch.capture.artifacts import GraphArtifact

    class NoChunks:
        def configure_stage(self, stage):
            return None

    def layer(x):
        return x.sin().cos().exp()

    artifact = GraphArtifact.capture(
        kind="inference", graph_module=make_fx(layer)(x), example_inputs=(x,)
    )
    composed = region_artifact(
        artifact,
        ChunkRegions(
            IsolateOperations(("aten.cos.default",), base="whole"), NoChunks()
        ),
        stage_id="block",
        variant="save",
    )
    runner = RegionExecutable(
        composed.regions.identity, {i: Child(i) for i in range(3)}
    )
    runner(x)


def test_derived_parameter_extent_does_not_become_an_execution_row():
    from shadowspill.pytorch.capture.artifacts import GraphArtifact, TaskInputProvenance
    from shadowspill.task.inputs import TaskInputRole

    class Rows:
        def configure_stage(self, stage):
            axes = {
                i: 0
                for i, role in enumerate(stage.input_roles)
                if role not in {TaskInputRole.PARAMETER, TaskInputRole.BUFFER}
            }
            return AxisChunks(3, axes) if axes else None

    def layer(x, weight):
        shared = weight.float().t()
        return x.sin() @ shared

    x, weight = torch.randn(17, 17), torch.randn(5, 17, dtype=torch.float64)
    artifact = GraphArtifact.capture(
        kind="inference",
        graph_module=make_fx(layer)(x, weight),
        example_inputs=(x, weight),
        input_provenance=(
            TaskInputProvenance(TaskInputRole.ACTIVATION),
            TaskInputProvenance(TaskInputRole.PARAMETER),
        ),
    )
    composed = region_artifact(
        artifact,
        ChunkRegions(IsolateOperations(("aten.sin.default",), base="whole"), Rows()),
        stage_id="block",
        variant=None,
    )
    child = composed.regions.children[-1]
    assert any(
        p.role is TaskInputRole.PARAMETER and geometry.shape == (17, 5)
        for p, geometry in zip(child.input_provenance, child.tensor_inputs, strict=True)
    )
    torch.testing.assert_close(
        region_executable(composed)(x, weight)[0], layer(x, weight)
    )


def test_compiler_common_storage_cannot_merge_distinct_boundary_roots():
    from shadowspill.pytorch.capture.artifacts import GraphArtifact
    from shadowspill.pytorch.chunking.boundary import boundary_values
    from shadowspill.pytorch.chunking.contract import boundary_metadata

    x = torch.randn(17, 5)

    def layer(x):
        return x.sin(), x.sin()

    artifact = GraphArtifact.capture(
        kind="inference", graph_module=make_fx(layer)(x), example_inputs=(x,)
    )
    metadata = boundary_metadata(artifact, list(layer(x)))
    assert len(metadata["storage"]["roots"]) == 2
    shared = x.sin()
    first, second = boundary_values(metadata, (x,), [shared, shared])
    assert first.untyped_storage()._cdata != second.untyped_storage()._cdata
    torch.testing.assert_close(first, second)


def test_one_structural_region_can_override_its_chunk_size_independently():
    from dataclasses import replace

    from shadowspill.pytorch.capture.artifacts import GraphArtifact

    def layer(x):
        return torch.nn.functional.silu(x.sin().softmax(1))

    x = torch.randn(2, 17, 5)
    artifact = GraphArtifact.capture(
        kind="inference", graph_module=make_fx(layer)(x), example_inputs=(x,)
    )
    original_policy = RegionChunks(17, 2)
    partition = IsolateOperations(("aten._softmax.default",), base="whole")
    original = region_artifact(
        artifact,
        ChunkRegions(partition, original_policy),
        stage_id="block",
        variant=None,
    )
    selected = original_policy.seen[0].structural_contract

    class Override(RegionChunks):
        def configure_stage(self, stage):
            config = super().configure_stage(stage)
            return (
                replace(config, chunk_size=11)
                if stage.structural_contract == selected
                else config
            )

    changed = region_artifact(
        artifact,
        ChunkRegions(partition, Override(17, 2)),
        stage_id="block",
        variant=None,
    )
    assert changed.compatibility_digest != original.compatibility_digest
    assert changed.storage_contract == original.storage_contract
    assert changed.regions.children[0].chunking.size == 11
    assert changed.regions.children[2].chunking.size == 7
    assert [c.compatibility_digest for c in changed.regions.children[1:]] == [
        c.compatibility_digest for c in original.regions.children[1:]
    ]
    torch.testing.assert_close(region_executable(changed)(x)[0], layer(x))


@pytest.mark.parametrize("budget", [0.0, 1.0])
def test_flattened_statistics_keep_chunk_axes_across_execution_cuts(budget):
    torch.manual_seed(731)
    x = torch.randn(2, 17, 3, 5, dtype=torch.float64, requires_grad=True)

    def layer(x):
        normalized = torch.nn.functional.layer_norm(x.reshape(-1, 5), (5,))
        return normalized.reshape(2, 17, 3, 5).softmax(1).sin()

    pair = capture_graph_pair(
        make_fx(layer)(x), (x,), original_output=layer(x), memory_budget=budget
    )
    policy = ChunkRegions(
        IsolateOperations(
            ("aten._softmax.default", "aten._softmax_backward_data.default"),
            base="whole",
        ),
        RegionChunks(17, 2),
    )
    forward = region_artifact(
        pair.forward,
        policy,
        stage_id="block",
        variant="save" if budget else "recompute",
    )
    count = pair.saved_value_count
    inherited = (
        *forward.regions.output_domains[-count:],
        forward.regions.output_domains[0],
    )
    backward = region_artifact(
        pair.backward,
        policy,
        stage_id="block",
        variant="save" if budget else "recompute",
        input_domains=inherited,
    )
    with torch.no_grad():
        expected = pair.forward.graph_module(x)
        actual = region_executable(forward)(x)
        tangent = torch.randn_like(actual[0])
        gradients = region_executable(backward)(*actual[-count:], tangent)
        reference = pair.backward.graph_module(*expected[-count:], tangent)
        torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-11)
        torch.testing.assert_close(gradients, reference, atol=1e-11, rtol=1e-11)
    assert any(
        any(
            isinstance(axis, ChunkAxis) and axis.after == 3
            for axis in child.chunking.inputs
        )
        for child in backward.regions.children
        if child.chunking
    )


@pytest.mark.parametrize("budget", [0.0, 1.0])
def test_regions_choose_different_logical_axes_and_chunk_sizes(budget):
    x = torch.randn(7, 11, dtype=torch.float64, requires_grad=True)

    def layer(x):
        return x.sin().cos()

    class DifferentAxes:
        def configure_stage(self, stage):
            dim = 0 if "aten.sin.default" in stage.operators else 1
            return AxisChunks(
                3 if dim == 0 else 4,
                {i: dim - 2 for i in range(len(stage.inputs))},
                domain="items" if dim == 0 else "features",
            )

    pair = capture_graph_pair(
        make_fx(layer)(x), (x,), original_output=layer(x), memory_budget=budget
    )
    policy = ChunkRegions(
        IsolateOperations(("aten.sin.default",), base="whole"), DifferentAxes()
    )
    forward, backward = (
        region_artifact(artifact, policy, stage_id="block", variant=None)
        for artifact in (pair.forward, pair.backward)
    )
    assert {
        (c.chunking.inputs[0].dim, c.chunking.size) for c in forward.regions.children
    } == {(0, 3), (1, 4)}
    with torch.no_grad():
        expected = pair.forward.graph_module(x)
        actual = region_executable(forward)(x)
        tangent = torch.randn_like(actual[0])
        count = pair.saved_value_count
        gradients = region_executable(backward)(*actual[-count:], tangent)
        reference = pair.backward.graph_module(*expected[-count:], tangent)
        torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-11)
        torch.testing.assert_close(gradients, reference, atol=1e-11, rtol=1e-11)


def test_same_named_domain_cannot_change_coordinates_at_an_execution_cut():
    from shadowspill.pytorch.capture.artifacts import GraphArtifact

    x = torch.randn(7, 7)
    artifact = GraphArtifact.capture(
        kind="inference",
        graph_module=make_fx(lambda x: x.sin().cos())(x),
        example_inputs=(x,),
    )

    class ConflictingAxes:
        def configure_stage(self, stage):
            dim = 0 if "aten.sin.default" in stage.operators else 1
            return AxisChunks(3, {0: dim}, domain="items")

    with pytest.raises(CaptureError, match="disagrees with chunk domain"):
        region_artifact(
            artifact,
            ChunkRegions(
                IsolateOperations(("aten.sin.default",), base="whole"),
                ConflictingAxes(),
            ),
            stage_id="block",
            variant=None,
        )
