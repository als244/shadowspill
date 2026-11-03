"""Source selections survive differentiation, structural reuse and persistence."""

import json

import pytest
import torch
from torch.fx import Graph, GraphModule

from shadowspill.errors import CaptureError
from shadowspill.pytorch.capture.objective import capture_training
from shadowspill.pytorch.capture.source import (
    SourceOperation,
    bind_source_metadata,
    set_source_operation,
    source_operation,
    traceable_source_graph,
)
from shadowspill.pytorch.graph_pairs import GraphPairStore, partition_training_capture
from shadowspill.pytorch.partition.metadata import module_paths
from tests.shadowspill.pytorch.graph_pairs.test_graph_pairs import _RepeatedNetwork


@pytest.mark.parametrize("disk_restore", [False, True])
def test_source_modules_belong_to_each_reused_stage(tmp_path, disk_restore):
    model = _RepeatedNetwork()
    inputs = (torch.randn(7, 8), torch.randn(7, 8))

    def objective(current, value, target):
        return (current(value) - target).square().mean()

    captured = capture_training(model, objective, inputs)
    store = GraphPairStore(tmp_path)
    first = partition_training_capture(
        captured, graph_pair_store=store, accumulating=True
    )
    assert store.misses == 3 and store.hits == 1
    if disk_restore:
        store = GraphPairStore(tmp_path)
        selected = partition_training_capture(
            captured, graph_pair_store=store, accumulating=True
        )
        assert store.misses == 0 and store.hits == 4
    else:
        selected = first
    for index, stage in enumerate(selected.stages):
        expected = set(module_paths(stage.example.stage.graph_module))
        for variant in stage.graph_pairs.variants:
            for artifact in (variant.pair.forward, variant.pair.backward):
                actual = set(module_paths(artifact.graph_module))
                assert f"model.blocks.{index}" in actual
                assert actual <= expected
                products = [
                    node
                    for node in artifact.graph_module.graph.nodes
                    if str(node.target) in {"aten.addmm.default", "aten.mm.default"}
                ]
                assert products and all(source_operation(node) for node in products)
    for left, right in zip(
        selected.stages[1].graph_pairs.variants,
        selected.stages[2].graph_pairs.variants,
        strict=True,
    ):
        assert (
            left.pair.forward.compatibility_digest
            == right.pair.forward.compatibility_digest
        )
        assert (
            left.pair.backward.compatibility_digest
            == right.pair.backward.compatibility_digest
        )
        assert left.pair.forward.graph_module is not right.pair.forward.graph_module
    for manifest in tmp_path.rglob("manifest.json"):
        assert (
            json.loads(manifest.read_text())["metadata_policy"] == "source_precision"
        )


@pytest.mark.parametrize("variant", ["save", "recompute"])
def test_execution_policy_observes_current_source_module_after_reuse(variant):
    from shadowspill.pytorch.chunking import AxisChunks, ChunkRegions
    from shadowspill.pytorch.chunking.regions import region_artifact
    from tests.shadowspill.pytorch.chunking.test_regions import region_executable

    torch.manual_seed(71)
    model = _RepeatedNetwork().double()
    inputs = (torch.randn(7, 8).double(), torch.randn(7, 8).double())
    captured = capture_training(
        model, lambda current, x, y: (current(x) - y).square().mean(), inputs
    )
    stages = partition_training_capture(captured).stages

    class Chunks:
        def configure_stage(self, stage):
            sizes = {
                2 if path.startswith("model.blocks.1") else 3
                for path in stage.module_paths
                if path.startswith("model.blocks.")
            }
            assert len(sizes) == 1
            axes = {
                i: 0
                for i, geometry in enumerate(stage.inputs)
                if geometry is not None and geometry.shape == (7, 8)
            }
            return AxisChunks(sizes.pop(), axes) if axes else None

    resolved = []
    for stage in stages[1:3]:
        pair = stage.graph_pairs.variant(variant).pair
        forward, backward = (
            region_artifact(
                artifact,
                ChunkRegions("whole", Chunks()),
                stage_id=stage.example.stage.stage_id,
                variant=variant,
            )
            for artifact in (pair.forward, pair.backward)
        )
        resolved.append(forward.regions.children[0].chunking.size)
        assert backward.regions.children[0].chunking.size == resolved[-1]
        with torch.no_grad():
            arguments = tuple(
                torch.empty_strided(
                    geometry.shape, geometry.stride, dtype=geometry.dtype
                ).normal_()
                for geometry in pair.forward.tensor_inputs
            )
            expected = pair.forward.graph_module(*arguments)
            actual = region_executable(forward)(*arguments)
            torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
            tangent = torch.randn_like(expected[0])
            reference = pair.backward.graph_module(
                *expected[-pair.saved_value_count :], tangent
            )
            gradients = region_executable(backward)(
                *actual[-pair.saved_value_count :], tangent
            )
            torch.testing.assert_close(gradients, reference, rtol=1e-11, atol=1e-11)
    assert resolved == [2, 3]


def _source_graph(path, *, unused_static=False):
    graph = Graph()
    x = graph.placeholder("x")
    if unused_static:
        graph.placeholder("unused")
    y = graph.call_function(torch.ops.aten.sin.default, (x,))
    y.meta["nn_module_stack"] = {"source": (path, "Block")}
    graph.output(y)
    return GraphModule({}, graph)


def test_source_identity_ignores_static_placeholders_and_rebinds_without_mutation():
    first = traceable_source_graph(_source_graph("blocks.1"))
    second = traceable_source_graph(_source_graph("blocks.2", unused_static=True))
    left = next(node for node in first.graph.nodes if node.op == "call_function")
    right = next(node for node in second.graph.nodes if node.op == "call_function")
    assert source_operation(left) == source_operation(right)
    rebound = bind_source_metadata(first, second)
    assert module_paths(first) == ("blocks.1",)
    assert module_paths(rebound) == ("blocks.2",)


@pytest.mark.parametrize(
    "source,message",
    [
        (SourceOperation(5, "call_function", "aten.sin.default"), "outside"),
        (SourceOperation(0, "call_function", "aten.cos.default"), "contract differs"),
    ],
)
def test_source_binding_refuses_a_mismatched_contract(source, message):
    graph = _source_graph("blocks.1")
    node = next(node for node in graph.graph.nodes if node.op == "call_function")
    set_source_operation(node, source)
    with pytest.raises(CaptureError, match=message):
        bind_source_metadata(graph, graph)
