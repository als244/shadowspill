"""Semantic view preparation preserves storage, values and effect ordering."""

import json
import operator
from collections import Counter
from dataclasses import asdict

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from engine_tests.pytorch.chunking.test_lifetime_analysis import opaque_write
from shadowspill.errors import CaptureError
from shadowspill.pytorch.chunking.lifetime_analysis import analyze_lifetimes
from shadowspill.pytorch.chunking.lifetime_views import prepare_internal_views

VIEWS = {
    torch.ops.aten.slice.Tensor,
    torch.ops.aten.view.default,
    torch.ops.aten.split.Tensor,
    operator.getitem,
}


def prepare(graph, inputs, *, barriers=()):
    analysis = analyze_lifetimes(graph, inputs)
    schedule = prepare_internal_views(
        graph,
        analysis,
        selected=frozenset(n for n in graph.graph.nodes if n.target in VIEWS),
        barriers=frozenset(n for n in graph.graph.nodes if n.target in barriers),
    )
    return analysis, schedule


@pytest.mark.parametrize("length", [7, 17, 31])
@pytest.mark.parametrize("tuple_views", [False, True])
def test_sibling_views_can_be_prepared_together_with_one_projection(
    length, tuple_views
):
    torch.manual_seed(717)
    inputs = (
        torch.randn(length, 5, dtype=torch.float64, requires_grad=True),
        torch.randn(5, 12, dtype=torch.float64, requires_grad=True),
    )

    def layer(x, weight):
        packed = x @ weight
        parts = packed.split((8, 4), -1) if tuple_views else None
        attended = (parts[0] if tuple_views else packed[:, :8]).cumsum(0)
        gate = parts[1] if tuple_views else packed[:, 8:]
        return attended[:, :4] * gate.sigmoid()

    graph = make_fx(layer)(*inputs)
    nodes = tuple(graph.graph.nodes)
    counts = Counter(n.target for n in nodes)
    initial, schedule = prepare(
        graph, inputs, barriers=(torch.ops.aten.cumsum.default,)
    )
    assert Counter(n.target for n in graph.graph.nodes) == counts
    assert counts[torch.ops.aten.mm.default] == 1
    assert any(item.applied for item in schedule.decisions)
    assert all(after <= before for _, before, after in schedule.barrier_bytes)
    final = analyze_lifetimes(graph, inputs)
    assert initial.external_roots == final.external_roots
    expected, actual = layer(*inputs), graph(*inputs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    tangent = torch.randn_like(expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual, inputs, tangent),
        torch.autograd.grad(expected, inputs, tangent),
        rtol=1e-12,
        atol=1e-12,
    )
    json.dumps(asdict(schedule))


@pytest.mark.parametrize("external", [False, True])
def test_input_and_outer_alias_backings_are_unchanged(external):
    x = torch.randn(7, 12)

    def function(x):
        backing = x.sin() if external else x
        independent = x.cos()
        late = backing[:, :4]
        return late, independent

    graph = make_fx(function)(x)
    original = graph.code
    _, result = prepare(graph, (x,))
    assert graph.code == original
    assert result.decisions
    assert all(
        not item.applied and item.reason == "input or externally observable storage"
        for item in result.decisions
    )
    actual = graph(x)
    assert (
        actual[0].untyped_storage().data_ptr() == x.untyped_storage().data_ptr()
    ) is not external


def test_views_cannot_cross_an_opaque_writer_of_their_backing():
    x, delta = torch.randn(7, 12), torch.randn(7, 12)

    def function(x, delta):
        backing = x.sin()
        opaque_write(backing, delta)
        return backing[:, :4].cos()

    graph = make_fx(function)(x, delta)
    original = graph.code
    _, schedule = prepare(graph, (x, delta))
    assert graph.code == original
    assert any(
        not item.applied and item.reason == "crossed effect or overlapping writer"
        for item in schedule.decisions
    )
    torch.testing.assert_close(graph(x, delta), function(x, delta), rtol=0, atol=0)


def test_known_disjoint_write_can_be_crossed_without_changing_destination():
    x, prior = torch.randn(7, 12), torch.randn(12)

    def function(x, prior):
        backing = x.sin()
        prior.add_(x.sum(0))
        return backing[:, :4].cos(), prior

    graph = make_fx(function)(x, prior.clone())
    _, schedule = prepare(graph, (x, prior))
    assert any(item.applied and item.crossed_writes for item in schedule.decisions)
    destination = prior.clone()
    actual = graph(x, destination)
    expected = function(x, prior.clone())
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual[1].data_ptr() == destination.data_ptr()


def test_rng_remains_a_fence_for_view_preparation():
    x = torch.randn(7, 12)

    def function(x):
        backing = x.sin()
        noise = torch.rand_like(x)
        return backing[:, :4].cos() + noise[:, :4]

    graph = make_fx(function)(x)
    _, schedule = prepare(graph, (x,))
    assert any(not item.applied and item.obstructions for item in schedule.decisions)
    torch.manual_seed(123)
    expected = function(x)
    torch.manual_seed(123)
    torch.testing.assert_close(graph(x), expected, rtol=0, atol=0)


def test_view_preparation_never_replays_fresh_computation_or_moves_barriers():
    x = torch.randn(7, 12)
    graph = make_fx(lambda x: x.sin()[:, :4].cos())(x)
    analysis = analyze_lifetimes(graph, (x,))
    nodes = frozenset(n for n in graph.graph.nodes if n.op == "call_function")
    result = prepare_internal_views(graph, analysis, selected=nodes, barriers=nodes)
    assert all(
        not item.applied and item.reason == "protected memory barrier"
        for item in result.decisions
    )
    result = prepare_internal_views(
        graph, analysis, selected=nodes, barriers=frozenset()
    )
    assert any(
        item.reason == "operation is not a semantic alias view"
        for item in result.decisions
    )


def test_preparing_layout_sensitive_view_does_not_copy_or_change_its_strides():
    x = torch.arange(24, dtype=torch.float64).reshape(2, 12)

    def function(x):
        backing = x.sin()
        extra = x.cos().sum()
        late = backing[:, :4]
        return late.as_strided((2, 2), (4, 1)).sin() + extra

    graph = make_fx(function)(x)
    _, schedule = prepare(graph, (x,))
    assert any(item.applied for item in schedule.decisions)
    assert all(n.target is not torch.ops.aten.clone.default for n in graph.graph.nodes)
    torch.testing.assert_close(graph(x), function(x), rtol=0, atol=0)
    view = x[:, :4]
    # Readonly alone cannot justify a layout-changing copy: both views below
    # are in bounds, but the second reads different physical elements.
    assert not torch.equal(
        view.as_strided((2, 2), (4, 1)), view.clone().as_strided((2, 2), (4, 1))
    )


def test_stale_and_foreign_view_selections_fail():
    x = torch.randn(7)
    graph, other = make_fx(torch.sin)(x), make_fx(torch.cos)(x)
    analysis = analyze_lifetimes(graph, (x,))
    with pytest.raises(CaptureError, match="stale or belongs"):
        prepare_internal_views(
            other, analysis, selected=frozenset(), barriers=frozenset()
        )
    with pytest.raises(CaptureError, match="must belong"):
        prepare_internal_views(
            graph, analysis, selected=frozenset(other.graph.nodes), barriers=frozenset()
        )
