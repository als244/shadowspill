"""Split templates preserve current aliases, metadata and conservative fallback."""

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx

from shadowspill.pytorch.capture.torch_deprecations import copy_graph_module
from shadowspill.pytorch.chunking.observations import (
    SplitObservations,
    split_output_values,
)
from shadowspill.pytorch.chunking.transform_cache import execution_analysis_scope
from shadowspill.pytorch.partition.policy import resolve_partition_assignments


def assignments_for(graph):
    return resolve_partition_assignments(graph, graph, "whole")[0]


def test_observation_rebinds_mutated_input_views_and_current_graph_metadata(
    monkeypatch,
):
    from shadowspill.pytorch.chunking import observations

    def operation(x):
        value = x[:, 1:]
        value.add_(1)
        return value, value, x, 7

    with FakeTensorMode():
        x, y = torch.empty(5, 4), torch.empty(5, 4)
        graph = make_fx(operation)(x)
        assignments = assignments_for(graph)
        cache = SplitObservations(2)
        first = cache.observe(graph, assignments, (x,))
        current = copy_graph_module(graph)
        for node in current.graph.nodes:
            node.meta["nn_module_stack"] = {"block": ("second.block", torch.nn.Module)}
        assignments = assignments_for(current)

        def unexpected(*args, **kwargs):
            raise AssertionError("matching fake geometry was evaluated twice")

        monkeypatch.setattr(observations, "split_and_observe", unexpected)
        second = cache.observe(current, assignments, (y,))
        assert cache.hits == cache.misses == 1
        assert second.stages[0].inputs[0] is y
        assert first.stages[0].inputs[0] is x
        assert second.stages[0].graph_module is not first.stages[0].graph_module
        values = split_output_values(second, (y,))
        assert values[0] is values[1]
        assert values[2] is y and values[3] == 7
        assert values[0].untyped_storage()._cdata == y.untyped_storage()._cdata
        assert values[0].storage_offset() == 1
        assert tuple(values[0].stride()) == (4, 1)
        assert all(
            n.meta["nn_module_stack"]["block"][0] == "second.block"
            for n in second.stages[0].graph_module.graph.nodes
            if n.op != "output"
        )


def test_input_aliases_backing_extents_and_cuts_distinguish_templates():
    with FakeTensorMode():
        x = torch.empty(5, 4)
        graph = make_fx(lambda a, b: (a.sin() + b, a))(x, x)
        assignments = assignments_for(graph)
        cache = SplitObservations(4)
        cache.observe(graph, assignments, (x, x))
        cache.observe(graph, assignments, (x, torch.empty_like(x)))
        larger = torch.empty(6, 4)[:5]
        cache.observe(graph, assignments, (larger, larger))
        assignments = {node: index for index, node in enumerate(assignments)}
        cache.observe(graph, assignments, (x, x))
        assert cache.misses == 4 and cache.hits == 0


def test_rebound_wrapper_is_deferred_but_executes_and_tracks_graph_edits(monkeypatch):
    from torch.fx import GraphModule

    with FakeTensorMode():
        x, y = torch.empty(5, 4), torch.empty(5, 4)
        graph = make_fx(lambda x: x.sin())(x)
        assignments = assignments_for(graph)
        cache = SplitObservations(2)
        cache.observe(graph, assignments, (x,))
        calls = []
        recompile = GraphModule.recompile

        def record(self):
            calls.append(self)
            return recompile(self)

        monkeypatch.setattr(GraphModule, "recompile", record)
        rebound = cache.observe(graph, assignments, (y,))
        assert cache.hits == 1 and not calls
    child = rebound.stages[0].graph_module
    value = torch.randn(5, 4)
    torch.testing.assert_close(child(value), value.sin())
    assert len(calls) == 1
    operation = next(n for n in child.graph.nodes if n.op == "call_function")
    operation.target = torch.ops.aten.cos.default
    child.recompile()
    assert len(calls) == 1
    assert "cos" in child.code
    torch.testing.assert_close(child(value), value.cos())
    assert len(calls) == 2


def test_real_values_are_observed_and_cache_scope_releases_templates():
    x = torch.randn(5, 4)
    graph = make_fx(lambda x: (x.sin(), x))(x)
    assignments = assignments_for(graph)
    cache = SplitObservations(1)
    for value in (x, x + 2):
        result = cache.observe(graph, assignments, (value,))
        torch.testing.assert_close(
            split_output_values(result, (value,)), [value.sin(), value]
        )
    assert cache.bypasses == 2 and not cache.entries
    with pytest.raises(RuntimeError), execution_analysis_scope(max_entries=1) as scope:
        with FakeTensorMode():
            a, b = torch.empty(5, 4), torch.empty(6, 4)
            for value in (a, b, a):
                scope.observations.observe(graph, assignments, (value,))
            assert scope.observations.misses == 3
            assert len(scope.observations.entries) == 1
        raise RuntimeError("end build")
    assert not scope.observations.entries
