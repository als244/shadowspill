"""Storage reuse preserves aliases, offsets, mutation targets and graph names."""

import gc
import weakref

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from shadowspill.pytorch.capture.storage import (
    ExplicitMutation,
    capture_task_storage_contract,
)
from shadowspill.pytorch.capture.storage.cache import (
    current_storage_cache,
    storage_contract_scope,
)
from shadowspill.pytorch.capture.torch_deprecations import copy_graph_module


def test_hit_rebinds_producers_without_replaying_fake_operators(monkeypatch):
    from shadowspill.pytorch.capture.storage import capture

    x, destination = torch.randn(7, 12), torch.zeros(7)

    def function(x, destination):
        left, right = x.sin().split(6, -1)
        destination.add_(left.sum(-1))
        return right.cos(), destination

    graph = make_fx(function)(x, destination.clone())
    renamed = copy_graph_module(graph)
    for index, node in enumerate(renamed.graph.nodes):
        node.name = f"next_{index}"
    renamed.recompile()
    expected = capture_task_storage_contract(renamed, (x, destination))
    with storage_contract_scope() as cache:
        capture_task_storage_contract(graph, (x, destination))

        def unexpected(*args, **kwargs):
            raise AssertionError("matching storage evidence was recaptured")

        monkeypatch.setattr(capture, "_task_output_values", unexpected)
        assert (
            capture_task_storage_contract(renamed, (x.clone(), destination.clone()))
            == expected
        )
        assert cache.hits == cache.misses == 1
    assert current_storage_cache.get() is None and not cache._entries


def test_aliases_backings_graph_literals_and_explicit_mutations_are_keyed():
    x = torch.randn(7, 12)
    graph = make_fx(lambda x, y: (x.add(2), y))(x, x)
    cases = [
        ((x, x), ()),
        ((x, x.clone()), ()),
        ((torch.randn(8, 12)[:7], x), ()),
        ((x, x), (ExplicitMutation(0, 0, "state"),)),
        ((x, x), (ExplicitMutation(0, 0, "different_state"),)),
    ]
    expected = [
        capture_task_storage_contract(graph, values, explicit_mutations=mutations)
        for values, mutations in cases
    ]
    with storage_contract_scope() as cache:
        actual = [
            capture_task_storage_contract(graph, values, explicit_mutations=mutations)
            for values, mutations in cases
        ]
        assert actual == expected and cache.misses == len(cases)
        add = next(
            n for n in graph.graph.nodes if n.target is torch.ops.aten.add.Tensor
        )
        add.args = (add.args[0], 3)
        graph.recompile()
        capture_task_storage_contract(graph, (x, x))
        assert cache.misses == len(cases) + 1 and cache.hits == 0


def test_cache_holds_metadata_only_and_clears_on_error():
    x = torch.randn(7, 12)
    with (
        pytest.raises(RuntimeError, match="stop"),
        storage_contract_scope(max_entries=1) as cache,
    ):
        graph = make_fx(torch.sin)(x)
        reference = weakref.ref(graph)
        capture_task_storage_contract(graph, (x,))
        del graph
        gc.collect()
        assert reference() is None
        capture_task_storage_contract(make_fx(torch.cos)(x), (x,))
        capture_task_storage_contract(make_fx(torch.sin)(x), (x,))
        assert cache.misses == 3
        with storage_contract_scope() as nested:
            assert nested is cache
        raise RuntimeError("stop")
    assert current_storage_cache.get() is None and not cache._entries
