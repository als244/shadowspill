"""Analysis reuse preserves storage, effects and occurrence policy decisions."""

import gc
import weakref

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from shadowspill.errors import CaptureError
from shadowspill.pytorch.capture.aot import capture_graph_pair
from shadowspill.pytorch.capture.torch_deprecations import copy_graph_module
from shadowspill.pytorch.chunking import (
    AxisChunks,
    ChunkRegions,
    ExecutionLifetimes,
    LifetimeRule,
    LifetimeSelection,
    ReadonlyOperator,
)
from shadowspill.pytorch.chunking.lifetime_analysis import analyze_lifetimes
from shadowspill.pytorch.chunking.lifetime_cache import (
    current_analysis_cache,
    lifetime_analysis_scope,
)
from shadowspill.pytorch.chunking.regions import region_artifact
from shadowspill.pytorch.partition import IsolateOperations
from tests.shadowspill.pytorch.chunking.test_lifetime_analysis import opaque_readonly
from tests.shadowspill.pytorch.chunking.test_regions import (
    RegionChunks,
    region_executable,
)


def test_hit_rebinds_nodes_and_renamed_storage_and_mutation_producers(monkeypatch):
    import shadowspill.pytorch.chunking.lifetime_analysis as implementation

    x, destination = torch.randn(7, 12), torch.zeros(7)

    def function(x, destination):
        parts = x.sin().split(6, -1)
        destination.add_(parts[0].sum(-1))
        return parts[1].cos(), destination

    graph = make_fx(function)(x, destination.clone())
    renamed = copy_graph_module(graph)
    for index, node in enumerate(renamed.graph.nodes):
        node.name = f"occurrence_{index}"
    renamed.recompile()
    expected = analyze_lifetimes(renamed, (x, destination))
    with lifetime_analysis_scope() as cache:
        analyze_lifetimes(graph, (x, destination))

        def unexpected(*args, **kwargs):
            raise AssertionError("cache hit repeated fake storage execution")

        monkeypatch.setattr(implementation, "capture_task_storage_contract", unexpected)
        actual = analyze_lifetimes(renamed, (x, destination))
        assert actual == expected
        assert cache.hits == cache.misses == 1
        assert not set(actual.order).intersection(graph.graph.nodes)
        assert actual.order == tuple(renamed.graph.nodes)


def test_input_alias_groups_are_part_of_analysis_identity():
    x = torch.randn(7, 12)
    inputs = (x, x.detach())
    separate = (x, x.clone())
    graph = make_fx(lambda x, y: (x.sin(), y.add_(1)))(x.clone(), x.clone())
    expected = analyze_lifetimes(graph, separate)
    with lifetime_analysis_scope() as cache:
        shared = analyze_lifetimes(graph, inputs)
        distinct = analyze_lifetimes(graph, separate)
        assert distinct == expected
        assert shared.storage != distinct.storage
        assert cache.misses == 2 and cache.hits == 0
        assert analyze_lifetimes(graph, (x.clone(), x.clone())) == distinct
        assert cache.hits == 1


@pytest.mark.parametrize("change", ["shape", "stride", "offset", "dtype", "backing"])
def test_input_geometry_and_backing_extent_invalidate_reuse(change):
    x = torch.randn(7, 12)
    alternatives = {
        "shape": torch.randn(8, 12),
        "stride": torch.randn(12, 7).t(),
        "offset": torch.randn(8, 12)[1:],
        "dtype": x.double(),
        "backing": torch.randn(8, 12)[:7],
    }
    changed = alternatives[change]
    graph = make_fx(torch.sin)(x)
    expected = analyze_lifetimes(graph, (changed,))
    with lifetime_analysis_scope() as cache:
        analyze_lifetimes(graph, (x,))
        assert analyze_lifetimes(graph, (changed,)) == expected
        assert cache.misses == 2 and cache.hits == 0


def test_graph_edits_and_metadata_return_trees_invalidate_reuse():
    x = torch.randn(7, 12)
    graph = make_fx(lambda x: (x.sin() * 2).sum(0))(x)
    with lifetime_analysis_scope() as cache:
        analyze_lifetimes(graph, (x,))
        multiply = next(
            n for n in graph.graph.nodes if n.target is torch.ops.aten.mul.Tensor
        )
        multiply.args = (multiply.args[0], 3)
        graph.recompile()
        analyze_lifetimes(graph, (x,))
        assert cache.misses == 2
        del multiply.meta["val"]
        with pytest.raises(CaptureError, match="missing captured lifetime geometry"):
            analyze_lifetimes(graph, (x,))


def test_effect_registration_is_rechecked_on_every_request(monkeypatch):
    import torch.fx.node as node_module

    x = torch.randn(7, 12)
    graph = make_fx(torch.sin)(x)
    node = next(n for n in graph.graph.nodes if n.target is torch.ops.aten.sin.default)
    with lifetime_analysis_scope() as cache:
        first = analyze_lifetimes(graph, (x,))
        assert node not in first.fences
        monkeypatch.setattr(
            node_module,
            "_side_effectful_functions",
            node_module._side_effectful_functions | {node.target},
        )
        changed = analyze_lifetimes(graph, (x,))
        assert node in changed.fences
        assert cache.misses == 2
        with pytest.raises(CaptureError, match="cannot override mutation/RNG/effects"):
            analyze_lifetimes(
                graph, (x,), readonly=(ReadonlyOperator(str(node.target), "v1"),)
            )


def test_readonly_contract_versions_and_refusals_are_not_hidden():
    x = torch.randn(7, 12)
    graph = make_fx(opaque_readonly)(x)
    target = "lifetime_analysis_test.readonly.default"
    with lifetime_analysis_scope() as cache:
        uncertified = analyze_lifetimes(graph, (x,))
        certified = [
            analyze_lifetimes(graph, (x,), readonly=(ReadonlyOperator(target, v),))
            for v in ("v1", "v2")
        ]
        assert uncertified.fences and all(not a.fences for a in certified)
        assert certified[0].readonly_contracts != certified[1].readonly_contracts
        assert cache.misses == 3 and cache.hits == 0
        with pytest.raises(ValueError, match="conflicting readonly"):
            analyze_lifetimes(
                graph,
                (x,),
                readonly=(
                    ReadonlyOperator(target, "v1"),
                    ReadonlyOperator(target, "v2"),
                ),
            )


def test_bounded_cache_releases_graphs_and_scopes_after_error():
    x = torch.randn(7, 12)
    with (
        pytest.raises(RuntimeError, match="end build"),
        lifetime_analysis_scope(max_entries=1) as cache,
    ):
        graph = make_fx(torch.sin)(x)
        reference = weakref.ref(graph)
        analyze_lifetimes(graph, (x,))
        del graph
        gc.collect()
        assert reference() is None
        other = make_fx(torch.cos)(x)
        analyze_lifetimes(other, (x,))
        analyze_lifetimes(make_fx(torch.sin)(x), (x,))
        assert cache.misses == 3
        with lifetime_analysis_scope() as nested:
            assert nested is cache
        raise RuntimeError("end build")
    assert current_analysis_cache.get() is None
    with lifetime_analysis_scope() as fresh:
        assert fresh is not cache and fresh.hits == fresh.misses == 0


def test_unrecognized_tensor_representation_uses_uncached_analysis():
    class TensorSubclass(torch.Tensor):
        pass

    x = torch.randn(7, 12)
    graph = make_fx(torch.sin)(x)
    wrapped = x.as_subclass(TensorSubclass)
    expected = analyze_lifetimes(graph, (wrapped,))
    with lifetime_analysis_scope() as cache:
        assert analyze_lifetimes(graph, (wrapped,)) == expected
        assert analyze_lifetimes(graph, (wrapped,)) == expected
        assert cache.bypasses == 2 and cache.hits == cache.misses == 0


def test_nested_tensor_arguments_do_not_reuse_unrepresented_aliases():
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx import symbolic_trace
    from torch.fx.passes.fake_tensor_prop import FakeTensorProp

    x = torch.randn(7, 12)
    graph = symbolic_trace(lambda pair: torch.ops.aten.add.Tensor(pair[0], pair[1]))
    FakeTensorProp(graph, mode=FakeTensorMode(allow_non_fake_inputs=True)).propagate(
        [x, x]
    )
    inputs = (([x, x],), ([x, x.clone()],))
    expected = [analyze_lifetimes(graph, args) for args in inputs]
    with lifetime_analysis_scope() as cache:
        actual = [analyze_lifetimes(graph, args) for args in inputs]
        assert actual == expected
        assert cache.bypasses == 2 and cache.hits == cache.misses == 0


def test_training_region_entry_reuses_analysis_across_actual_graph_pair_forms(
    monkeypatch,
):
    from shadowspill.build.chunking import chunk_training
    from shadowspill.build.training_capture import TrainingCapture
    from shadowspill.pytorch.capture.objective import capture_training_objective
    from shadowspill.pytorch.chunking import lifetimes
    from shadowspill.pytorch.graph_pairs import partition_training_capture
    from shadowspill.pytorch.guards import capture_input_signature

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(5, 5, dtype=torch.float64))

        def forward(self, x):
            normalized = x.sin()
            return (normalized @ self.weight).softmax(0) + normalized

    class Blocks(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([Block() for _ in range(3)])

        def forward(self, x):
            for block in self.blocks:
                x = block(x)
            return x

    class Checkpoints:
        def assign_stages(self, graph, module):
            result = {}
            for node in graph.graph.nodes:
                if node.op in {"placeholder", "get_attr", "output"}:
                    continue
                paths = [
                    path for path, _ in node.meta.get("nn_module_stack", {}).values()
                ]
                result[node.name] = next(
                    (
                        i
                        for i in range(3)
                        if any(path.endswith(f"blocks.{i}") for path in paths)
                    ),
                    3,
                )
            return result

    model = Blocks()
    inputs = (torch.randn(17, 5, dtype=torch.float64),)
    differentiated = partition_training_capture(
        capture_training_objective(model, lambda m, x: m(x).square().sum(), inputs),
        partition=Checkpoints(),
        accumulating=True,
    )
    assert len(differentiated.stages) == 4
    captured = TrainingCapture(
        model, (differentiated,), (capture_input_signature((inputs, {})),), ({},)
    )
    softmax = ("aten._softmax.default", "aten._softmax_backward_data.default")
    policy = ChunkRegions(
        IsolateOperations(softmax, base="whole"),
        RegionChunks(17),
        lifetimes=ExecutionLifetimes(
            rules=(
                LifetimeRule("aten.sin.default", replay=True),
                *(LifetimeRule(target, barrier=True) for target in softmax),
            )
        ),
    )
    plain = chunk_training.__wrapped__(captured, policy)
    observed = []
    apply = lifetimes.apply_lifetimes

    def observe(*args, **kwargs):
        observed.append(current_analysis_cache.get())
        return apply(*args, **kwargs)

    monkeypatch.setattr(lifetimes, "apply_lifetimes", observe)
    cached = chunk_training(captured, policy)
    assert observed and observed[0] is not None
    assert all(cache is observed[0] for cache in observed)
    assert observed[0].hits > 0
    assert current_analysis_cache.get() is None
    forms = 0
    for one, two in zip(
        plain.examples[0].stages, cached.examples[0].stages, strict=True
    ):
        for first, second in zip(
            one.graph_pairs.variants, two.graph_pairs.variants, strict=True
        ):
            forms += 1
            assert first.option_id == second.option_id
            for a, b in (
                (first.pair.forward, second.pair.forward),
                (first.pair.backward, second.pair.backward),
            ):
                assert a.compatibility_digest == b.compatibility_digest
                assert a.storage_contract == b.storage_contract
                if a.regions is not None:
                    assert a.regions.identity == b.regions.identity
                    assert a.regions.lifetime_report == b.regions.lifetime_report
    assert forms > 2 * len(differentiated.stages)


@pytest.mark.parametrize("save", [False, True])
@pytest.mark.parametrize("lora,batch", [(False, 1), (True, 2)])
def test_cached_region_policies_preserve_saved_cut_and_independent_vjp(
    save, lora, batch
):
    torch.manual_seed(941)
    inputs = (
        torch.randn(batch, 17, 5, dtype=torch.float64, requires_grad=True),
        torch.randn(5, 9, dtype=torch.float64, requires_grad=not lora),
        torch.randn(5, 3, dtype=torch.float64, requires_grad=True),
        torch.randn(3, 9, dtype=torch.float64, requires_grad=True),
        torch.randn(9, 5, dtype=torch.float64, requires_grad=True),
    )

    def layer(x, weight, a, b, out):
        normalized = x.sin()
        q = normalized @ weight + (normalized @ a) @ b
        return torch.nn.functional.silu(q.softmax(1) @ out) + normalized

    expected = layer(*inputs)
    tangent = torch.randn_like(expected)
    reference = torch.autograd.grad(
        expected, [v for v in inputs if v.requires_grad], tangent
    )
    pair = capture_graph_pair(
        make_fx(layer)(*inputs),
        inputs,
        original_output=expected,
        memory_budget=float(save),
    )
    observed = []
    chunks = []

    def select(info):
        observed.append((info.stage_id, info.variant, info.phase, info.module_paths))
        if info.target == "aten.sin.default":
            return LifetimeSelection(replay=info.stage_id == "first")
        return None

    class PerOccurrenceChunks(RegionChunks):
        def configure_stage(self, stage):
            chunks.append(stage.stage_id)
            config = super().configure_stage(stage)
            if config is not None:
                return AxisChunks(
                    3 if stage.stage_id.startswith("first/") else 5,
                    config.inputs,
                    domain=config.domain,
                )
            return None

    softmax = ("aten._softmax.default", "aten._softmax_backward_data.default")
    policy = ChunkRegions(
        IsolateOperations(softmax, base="whole"),
        PerOccurrenceChunks(17, batch),
        lifetimes=ExecutionLifetimes(
            rules=tuple(LifetimeRule(t, barrier=True) for t in softmax), select=select
        ),
    )

    def transform(stage):
        return tuple(
            region_artifact(
                a, policy, stage_id=stage, variant="save" if save else "recompute"
            )
            for a in (pair.forward, pair.backward)
        )

    plain = [transform(stage) for stage in ("first", "second")]
    expected_observed, expected_chunks = observed[:], chunks[:]
    observed.clear()
    chunks.clear()
    with lifetime_analysis_scope() as cache:
        cached = [transform(stage) for stage in ("first", "second")]
        assert cache.hits > 0
    assert observed == expected_observed and chunks == expected_chunks
    for before, after in zip(plain, cached, strict=True):
        for one, two in zip(before, after, strict=True):
            assert one.compatibility_digest == two.compatibility_digest
            assert one.regions.identity == two.regions.identity
            assert one.regions.lifetime_report == two.regions.lifetime_report
            assert one.storage_contract == two.storage_contract
        forward, backward = after
        with torch.no_grad():
            actual = region_executable(forward)(*inputs)
            saved = actual[-pair.saved_value_count :] if pair.saved_value_count else ()
            gradients = region_executable(backward)(*saved, tangent)
        torch.testing.assert_close(actual[0], expected, rtol=1e-11, atol=1e-11)
        torch.testing.assert_close(
            tuple(g for g in gradients if g is not None),
            reference,
            rtol=1e-10,
            atol=1e-10,
        )
