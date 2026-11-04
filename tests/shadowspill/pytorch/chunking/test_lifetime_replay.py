"""Explicit local replay preserves primal versions, aliases and arbitrary VJPs."""

import json
import operator
from collections import Counter
from dataclasses import asdict

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from shadowspill.errors import CaptureError
from shadowspill.pytorch.chunking.lifetime_analysis import analyze_lifetimes
from shadowspill.pytorch.chunking.lifetime_policy import ReadonlyOperator
from shadowspill.pytorch.chunking.lifetime_replay import replay_primal
from shadowspill.pytorch.partition import IsolateOperations
from shadowspill.pytorch.partition.policy import resolve_partition_assignments
from tests.shadowspill.pytorch.chunking._lifetime_ops import alias_observer

SOFTMAX = torch.ops.aten._softmax.default


def replay(
    graph,
    inputs,
    *,
    targets=(torch.ops.aten.sin.default,),
    readonly=(),
    primal=True,
    excluded=(),
    gradient_leaves=None,
):
    analysis = analyze_lifetimes(graph, inputs, readonly=readonly)
    regions, _ = resolve_partition_assignments(
        graph, graph, IsolateOperations((str(SOFTMAX),), base="whole")
    )
    return replay_primal(
        graph,
        analysis,
        inputs,
        selected=frozenset(n for n in graph.graph.nodes if n.target in targets),
        primal=frozenset(graph.graph.nodes) if primal else frozenset(),
        regions=regions,
        barriers=frozenset(n for n in graph.graph.nodes if n.target is SOFTMAX),
        readonly=readonly,
        excluded=frozenset(excluded),
        gradient_leaves=gradient_leaves,
    )


@pytest.mark.parametrize("batch,length", [(1, 7), (2, 17), (2, 31)])
@pytest.mark.parametrize("lora", [False, True])
def test_one_extra_primal_serves_all_late_consumers_with_independent_vjp(
    batch, length, lora
):
    torch.manual_seed(912)
    inputs = (
        torch.randn(batch, length, 5, dtype=torch.float64, requires_grad=True),
        torch.randn(5, 5, dtype=torch.float64, requires_grad=not lora),
        torch.randn(5, 2, dtype=torch.float64, requires_grad=True),
        torch.randn(2, 5, dtype=torch.float64, requires_grad=True),
    )

    def layer(x, weight, a, b):
        normalized = x.sin()
        projection = normalized @ weight + (normalized @ a) @ b
        attention = projection.softmax(1)
        return attention * (normalized.cos() + normalized.exp())

    graph = make_fx(layer)(*inputs)
    original_code = graph.code
    changed, report = replay(graph, inputs)
    counts = Counter(n.target for n in changed.graph.nodes)
    assert counts[torch.ops.aten.sin.default] == 2
    assert counts[torch.ops.aten.mm.default] == 3
    assert graph.code == original_code
    late = [
        n
        for n in changed.graph.nodes
        if n.target in (torch.ops.aten.cos.default, torch.ops.aten.exp.default)
    ]
    assert late[0].args[0] is late[1].args[0]
    assert any(d.applied for d in report.decisions)
    assert all(after <= before for _, before, after in report.barrier_bytes)
    assert any(after < before for _, before, after in report.barrier_bytes)
    expected, actual = layer(*inputs), changed(*inputs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    tangent = torch.randn_like(expected)
    trainable = [x for x in inputs if x.requires_grad]
    torch.testing.assert_close(
        torch.autograd.grad(actual, trainable, tangent),
        torch.autograd.grad(expected, trainable, tangent),
        rtol=1e-12,
        atol=1e-12,
    )
    json.dumps(asdict(report))


def test_replay_refuses_a_new_live_projection_across_the_barrier():
    x, weight = torch.randn(17, 5), torch.randn(5, 5)

    def layer(x, weight):
        normalized = (x @ weight).sin()
        return normalized.softmax(0) * normalized.cos()

    graph = make_fx(layer)(x, weight)
    changed, report = replay(graph, (x, weight))
    assert any(d.reason == "protected backing would increase" for d in report.decisions)
    assert not any(d.applied for d in report.decisions)
    assert changed.code == graph.code
    torch.testing.assert_close(changed(x, weight), layer(x, weight), rtol=0, atol=0)


def test_changed_input_version_cannot_be_recomputed_later():
    x = torch.randn(17, 5)

    def function(x):
        old = x.sin()
        x.add_(1)
        attention = x.softmax(0)
        return old.cos() + attention

    graph = make_fx(function)(x.clone())
    changed, report = replay(graph, (x,))
    assert any("changed dependency version" in d.reason for d in report.decisions)
    assert not any(d.applied for d in report.decisions)
    torch.testing.assert_close(changed(x.clone()), function(x.clone()), rtol=0, atol=0)


def test_disjoint_gradient_write_is_preserved_and_can_be_crossed():
    x, destination = torch.randn(17, 5), torch.randn(5)

    def function(x, destination):
        value = x.sin()
        projection = value * 3
        destination.add_(x.sum(0))
        attention = projection.softmax(0)
        return attention * value.cos(), destination

    graph = make_fx(function)(x, destination.clone())
    changed, report = replay(graph, (x, destination))
    assert any(d.applied for d in report.decisions)
    updated = destination.clone()
    actual = changed(x, updated)
    torch.testing.assert_close(actual, function(x, destination.clone()), rtol=0, atol=0)
    assert actual[1].data_ptr() == updated.data_ptr()
    assert sum(n.target is torch.ops.aten.add_.Tensor for n in changed.graph.nodes) == 1


def test_rng_is_a_replay_fence_and_has_unchanged_consumption():
    x = torch.randn(17, 5)

    def function(x):
        value = x.sin()
        noise = torch.rand_like(x)
        attention = (x + noise).softmax(0)
        return attention * value.cos()

    graph = make_fx(function)(x)
    changed, report = replay(graph, (x,))
    assert any("crossed effect" in d.reason for d in report.decisions)
    assert not any(d.applied for d in report.decisions)
    torch.manual_seed(912)
    expected = function(x)
    torch.manual_seed(912)
    torch.testing.assert_close(changed(x), expected, rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["external", "writer", "unknown"])
def test_external_outputs_written_outputs_and_missing_provenance_refuse(kind):
    x = torch.randn(17, 5)

    def function(x):
        value = x.sin()
        if kind == "writer":
            value.add_(1)
        attention = x.softmax(0)
        output = attention * value.cos()
        return (output, value) if kind == "external" else output

    graph = make_fx(function)(x)
    changed, report = replay(graph, (x,), primal=kind != "unknown")
    assert not any(d.applied for d in report.decisions)
    assert any(
        kind in d.reason or (kind == "unknown" and "provenance" in d.reason)
        for d in report.decisions
    )
    torch.testing.assert_close(changed(x), function(x), rtol=0, atol=0)


def test_readonly_does_not_justify_splitting_an_observed_alias_relationship():
    x = torch.randn(17, 5)

    def function(x):
        value = x.sin()
        alias = value.view_as(value)
        attention = x.softmax(0)
        return attention + alias_observer(value, alias)

    graph = make_fx(function)(x)
    target = "lifetime_copy_test.alias_observer.default"
    changed, report = replay(
        graph, (x,), readonly=(ReadonlyOperator(target, "effects.v1"),)
    )
    assert any(
        "layout-independent value semantics" in d.reason for d in report.decisions
    )
    assert not any(d.applied for d in report.decisions)
    torch.testing.assert_close(changed(x), function(x), rtol=0, atol=0)
    value = x.sin()
    assert not torch.equal(
        alias_observer(value, value), alias_observer(value.clone(), value)
    )


def test_replayed_readonly_consumer_still_needs_input_value_contracts():
    x = torch.randn(17, 5)

    def function(x):
        value = x.sin()
        alias = value.view_as(value)
        observed = alias_observer(value, alias)
        attention = (observed * 2).softmax(0)
        return attention + observed.cos()

    graph = make_fx(function)(x)
    target = torch.ops.lifetime_copy_test.alias_observer.default
    changed, report = replay(
        graph,
        (x,),
        targets=(torch.ops.aten.sin.default, target),
        readonly=(ReadonlyOperator(str(target), "effects.v1"),),
    )
    torch.testing.assert_close(changed(x), function(x), rtol=0, atol=0)
    original = tuple(graph.graph.nodes)
    assert any(
        "layout-independent value semantics" in d.reason
        and any(original[index].target is target for index in d.obstructions)
        for d in report.decisions
    )


def test_gradient_conversion_is_protected_even_with_primal_permission():
    x = torch.randn(17, 5)
    graph = make_fx(lambda x: (x.sin().double().sum(0),))(x)
    _, report = replay(
        graph, (x,), targets=(torch.ops.aten._to_copy.default,), gradient_leaves=(0,)
    )
    assert report.protected_conversions
    assert any("conversion before gradient" in d.reason for d in report.decisions)
    assert not any(d.applied for d in report.decisions)


@pytest.mark.parametrize("exclude_unpacking", [False, True])
def test_tuple_producer_is_replayed_once_with_its_structural_outputs(exclude_unpacking):
    x = torch.randn(17, 5, dtype=torch.float64)

    def function(x):
        value, mean, rstd = torch.ops.aten.native_layer_norm.default(
            x, [5], None, None, 1e-5
        )
        attention = (value * 2).softmax(0)
        return attention * (value.cos() + mean + rstd)

    graph = make_fx(function)(x)
    unpacking = [n for n in graph.graph.nodes if n.target is operator.getitem]
    changed, report = replay(
        graph,
        (x,),
        targets=(torch.ops.aten.native_layer_norm.default,),
        excluded=unpacking if exclude_unpacking else (),
    )
    count = sum(
        n.target is torch.ops.aten.native_layer_norm.default
        for n in changed.graph.nodes
    )
    assert count == (1 if exclude_unpacking else 2)
    assert any(d.applied for d in report.decisions) is not exclude_unpacking
    torch.testing.assert_close(changed(x), function(x), rtol=0, atol=0)


def test_explicitly_selected_sequence_barrier_is_never_replayed():
    x = torch.randn(17, 5)
    graph = make_fx(lambda x: x.softmax(0).sin())(x)
    changed, report = replay(graph, (x,), targets=(SOFTMAX,))
    assert any(d.reason == "protected memory barrier" for d in report.decisions)
    assert not any(d.applied for d in report.decisions)
    assert changed.code == graph.code


def test_stale_replay_analysis_is_refused():
    x = torch.randn(17, 5)
    graph, other = make_fx(torch.sin)(x), make_fx(torch.cos)(x)
    with pytest.raises(CaptureError, match="stale or belongs"):
        replay_primal(
            other,
            analyze_lifetimes(graph, (x,)),
            (x,),
            selected=frozenset(),
            primal=frozenset(),
            regions={},
            barriers=frozenset(),
        )
