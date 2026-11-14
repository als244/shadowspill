"""Resolved leaf transforms reuse work without caching occurrence policies."""

from dataclasses import replace

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from engine_tests.pytorch.chunking.test_composition import executable
from shadowspill.pytorch.capture.artifacts import GraphArtifact, TaskInputProvenance
from shadowspill.pytorch.chunking import AxisChunks, ChunkAxis, ChunkRegions
from shadowspill.pytorch.chunking.transform import chunk_artifact
from shadowspill.pytorch.chunking.transform_cache import (
    current_transform_cache,
    execution_analysis_scope,
)
from shadowspill.pytorch.partition import IsolateOperations
from shadowspill.task.inputs import TaskInputRole


def artifact(x):
    return GraphArtifact.capture(
        kind="forward",
        graph_module=make_fx(lambda x: (x.sin(), x.cos()))(x),
        example_inputs=(x,),
        input_provenance=(TaskInputProvenance(TaskInputRole.ACTIVATION, "first"),),
    )


def test_hit_rebinds_parent_values_and_avoids_rebuilding_leaf_graphs(monkeypatch):
    from shadowspill.pytorch.chunking import transform

    x, y = torch.randn(11, 5), torch.randn(11, 5)
    first = artifact(x)
    second = first.rebind_examples(
        (y,), input_provenance=(TaskInputProvenance(TaskInputRole.ACTIVATION, "next"),)
    )
    config = AxisChunks(4, {0: 0}, domain="tokens")
    with execution_analysis_scope() as cache:
        before = chunk_artifact(first, (ChunkAxis(0),), 11, config)

        def unexpected(*args, **kwargs):
            raise AssertionError("a resolved leaf was built twice")

        monkeypatch.setattr(transform, "_chunk_artifact", unexpected)
        after = chunk_artifact(second, (ChunkAxis(0),), 11, config)
        assert cache.hits == cache.misses == 1
        assert after.example_arguments[0] is y
        assert after.input_provenance == second.input_provenance
        assert after.graph_module is second.graph_module
        assert after.compatibility_digest == before.compatibility_digest
        torch.testing.assert_close(executable(after)(y), second.graph_module(y))
    assert not cache._entries
    assert current_transform_cache.get() is None


def test_nested_callbacks_run_for_every_occurrence_and_changed_sizes_are_honored():
    seen = []

    class Inner:
        def configure_stage(self, info):
            seen.append(info)
            axes = {
                i: domains["tokens"]
                for i, domains in enumerate(info.input_domains)
                if "tokens" in domains
            }
            size = 2 if info.checkpoint_stage_id == "small" else 3
            return AxisChunks(size, axes, domain="tokens") if axes else None

    x = torch.randn(11, 5)
    graph = artifact(x)
    config = AxisChunks(
        7,
        {0: 0},
        domain="tokens",
        regions=ChunkRegions(
            IsolateOperations(("aten.cos.default",), base="whole"), Inner()
        ),
    )
    with execution_analysis_scope() as cache:
        built = []
        for name in ("small", "large", "small"):
            start = len(seen)
            result = chunk_artifact(
                graph,
                (ChunkAxis(0),),
                11,
                config,
                stage_id=name,
                checkpoint_stage_id=name,
            )
            assert len(seen) > start
            assert all(info.checkpoint_stage_id == name for info in seen[start:])
            torch.testing.assert_close(executable(result)(x), graph.graph_module(x))
            sizes = {
                child.chunking.size
                for _, outer in result.chunking.children
                for child in outer.regions.children
                if child.chunking
            }
            assert sizes == ({2} if name == "small" else {3})
            built.append(result.compatibility_digest)
        assert built[0] == built[2] and built[0] != built[1]
        assert cache.hits > 0 and cache.bypasses == 3


def test_changed_axis_contract_roles_and_backing_extent_do_not_hit():
    x = torch.randn(11, 5)
    original = artifact(x)
    different_backing = artifact(torch.randn(12, 5)[:11])
    different_role = replace(
        original, input_provenance=(TaskInputProvenance(TaskInputRole.CONTROL),)
    )
    with execution_analysis_scope() as cache:
        for graph, config in (
            (original, AxisChunks(4, {0: 0}, domain="tokens")),
            (original, AxisChunks(3, {0: 0}, domain="tokens")),
            (original, AxisChunks(4, {0: 0}, domain="other")),
            (different_backing, AxisChunks(4, {0: 0}, domain="tokens")),
            (different_role, AxisChunks(4, {0: 0}, domain="tokens")),
        ):
            chunk_artifact(graph, (ChunkAxis(0),), 11, config)
        assert cache.misses == 5 and cache.hits == 0


def test_cache_is_bounded_and_released_after_failure():
    x = torch.randn(11, 5)
    graph = artifact(x)
    with (
        pytest.raises(RuntimeError, match="end build"),
        execution_analysis_scope(max_entries=1) as cache,
    ):
        for size in (4, 3, 4):
            chunk_artifact(graph, (ChunkAxis(0),), 11, AxisChunks(size, {0: 0}))
        assert cache.misses == 3 and cache.hits == 0
        with execution_analysis_scope() as nested:
            assert nested is cache
        raise RuntimeError("end build")
    assert not cache._entries and current_transform_cache.get() is None


def test_nested_slices_reuse_transforms_but_resolve_each_inner_policy(monkeypatch):
    from torch._subclasses.fake_tensor import FakeTensorMode

    from shadowspill.pytorch.chunking import transform

    seen = []

    class Inner:
        def configure_stage(self, info):
            seen.append(info)
            axes = {
                i: d["tokens"]
                for i, d in enumerate(info.input_domains)
                if "tokens" in d
            }
            size = 2 if info.checkpoint_stage_id == "small" else 3
            return AxisChunks(size, axes, domain="tokens") if axes else None

    config = AxisChunks(
        7, {0: 0}, domain="tokens", regions=ChunkRegions("whole", Inner())
    )
    original = transform._chunk_artifact
    with FakeTensorMode():
        first = artifact(torch.empty(11, 5))
        second = first.rebind_examples((torch.empty(11, 5),))
        with execution_analysis_scope() as cache:
            a = chunk_artifact(
                first, (ChunkAxis(0),), 11, config, checkpoint_stage_id="small"
            )

            def only_inner(*args, **kwargs):
                assert args[3].regions is None, "outer slices were transformed twice"
                return original(*args, **kwargs)

            monkeypatch.setattr(transform, "_chunk_artifact", only_inner)
            start = len(seen)
            b = chunk_artifact(
                second, (ChunkAxis(0),), 11, config, checkpoint_stage_id="large"
            )
            assert len(seen) > start
            assert all(info.checkpoint_stage_id == "large" for info in seen[start:])
            assert a.compatibility_digest != b.compatibility_digest
            assert b.example_arguments[0] is second.example_arguments[0]
            assert cache.hits > 0 and cache.bypasses == 0
    x = torch.randn(11, 5)
    torch.testing.assert_close(executable(a)(x), (x.sin(), x.cos()))
    torch.testing.assert_close(executable(b)(x), (x.sin(), x.cos()))


def test_nested_metadata_and_representative_values_are_not_substituted():
    from torch._subclasses.fake_tensor import FakeTensorMode

    from shadowspill.pytorch.capture.torch_deprecations import copy_graph_module

    seen = []
    sources = []

    class Partition:
        def assign_stages(self, graph, module):
            sources.append(
                (dict(graph.meta), tuple(dict(n.meta) for n in graph.graph.nodes))
            )
            return {n.name: 0 for n in graph.graph.nodes if n.op == "call_function"}

    class Inner:
        def configure_stage(self, info):
            seen.append(info)
            return None

    config = AxisChunks(
        7, {0: 0}, domain="tokens", regions=ChunkRegions(Partition(), Inner())
    )
    old_value, new_value = torch.ones(11, 5), torch.zeros(11, 5)
    opaque = object()
    with FakeTensorMode():
        first = artifact(torch.empty(11, 5))
        first = replace(
            first,
            input_provenance=(
                TaskInputProvenance(
                    TaskInputRole.ACTIVATION, "old", representative_value=old_value
                ),
            ),
        )
        changed = replace(
            first,
            graph_module=copy_graph_module(first.graph_module),
            input_provenance=(
                TaskInputProvenance(
                    TaskInputRole.ACTIVATION, "new", representative_value=new_value
                ),
            ),
        )
        changed.graph_module.meta["selector"] = "second"
        node = next(
            n for n in changed.graph_module.graph.nodes if n.op == "call_function"
        )
        node.meta["nn_module_stack"] = {"source": ("other.block", torch.nn.Linear)}
        node.meta["custom_selector_state"] = opaque
        with execution_analysis_scope() as cache:
            chunk_artifact(first, (ChunkAxis(0),), 11, config)
            start, source_start = len(seen), len(sources)
            result = chunk_artifact(changed, (ChunkAxis(0),), 11, config)
            assert cache.hits > 0
            assert len(seen) > start
            assert all("other.block" in info.module_paths for info in seen[start:])
            assert all(
                meta["selector"] == "second" for meta, _ in sources[source_start:]
            )
            assert all(
                any(item.get("custom_selector_state") is opaque for item in nodes)
                for _, nodes in sources[source_start:]
            )
            assert all(
                child.input_provenance[0].representative_value is new_value
                for _, child in result.chunking.children
            )
            assert all(
                child.input_provenance[0].source == "new"
                for _, child in result.chunking.children
            )
