"""Compaction proves complete sibling backing and transitive layout behavior."""

import json
from collections import Counter
from dataclasses import asdict

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from engine_tests.pytorch.chunking._lifetime_ops import (
    layout_observer,
    logical_consumer,
)
from shadowspill.errors import CaptureError
from shadowspill.pytorch.capture.storage import capture_task_storage_contract
from shadowspill.pytorch.chunking.lifetime_analysis import analyze_lifetimes
from shadowspill.pytorch.chunking.lifetime_compaction import (
    LayoutIndependentOperator,
    compact_internal_views,
)
from shadowspill.pytorch.chunking.lifetime_policy import ReadonlyOperator
from shadowspill.pytorch.chunking.lifetime_views import prepare_internal_views


def compact(graph, inputs, *, names=None, barriers=(), readonly=(), layout=()):
    analysis = analyze_lifetimes(graph, inputs, readonly=readonly)
    selected = frozenset(
        node
        for node in graph.graph.nodes
        if (names is None or node.name in names)
        and node.op == "call_function"
        and analysis.values[node]
        and analysis.values[node]
        <= frozenset().union(
            *(analysis.values[parent] for parent in node.all_input_nodes)
        )
    )
    pinned = frozenset(node for node in graph.graph.nodes if node.target in barriers)
    prepare_internal_views(graph, analysis, selected=selected, barriers=pinned)
    analysis = analyze_lifetimes(graph, inputs, readonly=readonly)
    # Alias nodes are producer-local; every arithmetic consumer is in a later
    # execution region. This isolates the copy proof from partition heuristics.
    regions = {
        node: 0 if node in selected else 1
        for node in graph.graph.nodes
        if node.op == "call_function"
    }
    return compact_internal_views(
        graph,
        analysis,
        inputs,
        selected=selected,
        regions=regions,
        barriers=pinned,
        readonly=readonly,
        layout=layout,
    )


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("length", [7, 17, 31])
@pytest.mark.parametrize("tuple_views", [False, True])
def test_joint_siblings_reduce_backing_without_replaying_projection(
    batch, length, tuple_views
):
    torch.manual_seed(511)
    inputs = (
        torch.randn(batch, length, 5, dtype=torch.float64, requires_grad=True),
        torch.randn(5, 12, dtype=torch.float64, requires_grad=True),
    )

    def layer(x, weight):
        packed = x @ weight
        if tuple_views:
            attended_input, gate = packed.split((8, 4), -1)
        else:
            attended_input, gate = packed[..., :8], packed[..., 8:]
        attended = attended_input.cumsum(1)
        return attended[..., :4] * gate.sin()

    graph = make_fx(layer)(*inputs)
    outer = capture_task_storage_contract(graph, inputs)
    sources = {
        n.args[0].name
        for n in graph.graph.nodes
        if n.target in (torch.ops.aten.cumsum.default, torch.ops.aten.sin.default)
    }
    changed, report = compact(
        graph,
        inputs,
        names=sources,
        barriers=(torch.ops.aten.cumsum.default, torch.ops.aten.sin.default),
    )
    counts = Counter(n.target for n in changed.graph.nodes)
    assert counts[torch.ops.aten.mm.default] == 1
    applied = [d for d in report.decisions if d.applied]
    assert len(applied) == 1 and len(applied[0].source_indices) == 2
    assert counts[torch.ops.aten.clone.default] == 2
    assert all(after <= before for _, before, after in report.barrier_bytes)
    assert any(after < before for _, before, after in report.barrier_bytes)
    assert capture_task_storage_contract(changed, inputs) == outer
    assert all(n.target is not torch.ops.aten.clone.default for n in graph.graph.nodes)
    actual, expected = changed(*inputs), layer(*inputs)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    tangent = torch.randn_like(expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual, inputs, tangent),
        torch.autograd.grad(expected, inputs, tangent),
        rtol=1e-12,
        atol=1e-12,
    )
    json.dumps(asdict(report))


def test_one_sibling_cannot_hide_the_original_backing_kept_by_another():
    x = torch.randn(7, 12)

    def function(x):
        packed = x * 2
        early, late = packed[:, :8], packed[:, 8:]
        attended = early.cumsum(0)
        return attended[:, :4] * late.sin()

    graph = make_fx(function)(x)
    source = next(
        n for n in graph.graph.nodes if n.target is torch.ops.aten.cumsum.default
    ).args[0]
    changed, report = compact(
        graph,
        (x,),
        names={source.name},
        barriers=(torch.ops.aten.cumsum.default, torch.ops.aten.sin.default),
    )
    assert not any(d.applied for d in report.decisions)
    assert any(d.reason == "protected storage would increase" for d in report.decisions)
    assert changed.code == graph.code


@pytest.mark.parametrize("transitive", [False, True])
def test_raw_and_transitive_layout_observers_are_refused(transitive):
    x = torch.arange(48, dtype=torch.float64).reshape(4, 12)

    def function(x):
        backing = x * 2
        if transitive:
            view = backing.T[:6, :]
            return layout_observer(view + 2)
        return backing[:, :4].as_strided((2, 2), (4, 1)).sin()

    graph = make_fx(function)(x)
    target = "lifetime_copy_test.layout_observer.default"
    source = next(
        n
        for n in graph.graph.nodes
        if n.target
        is (
            torch.ops.aten.add.Tensor
            if transitive
            else torch.ops.aten.as_strided.default
        )
    ).args[0]
    changed, report = compact(
        graph,
        (x,),
        names={source.name},
        readonly=(ReadonlyOperator(target, "effects.v1"),),
        barriers=(
            torch.ops.aten.sin.default,
            torch.ops.lifetime_copy_test.layout_observer.default,
        ),
    )
    assert not any(d.applied for d in report.decisions)
    assert any(
        d.reason == "changed input lacks layout-independent value semantics"
        for d in report.decisions
    )
    torch.testing.assert_close(changed(x), function(x), rtol=0, atol=0)


@pytest.mark.parametrize("contract_ports", [None, (0,), (1,)])
def test_opaque_contract_is_separate_from_readonly_and_argument_specific(
    contract_ports,
):
    x = torch.randn(7, 12)

    def function(x):
        backing = x * 2
        view = backing[:, :4]
        return logical_consumer(x[:, :4], offset=view)

    graph = make_fx(function)(x)
    consumer = next(
        n
        for n in graph.graph.nodes
        if n.target is torch.ops.lifetime_copy_test.logical_consumer.default
    )
    consumer.kwargs = {"offset": consumer.args[1]}
    consumer.args = (consumer.args[0],)
    graph.recompile()
    target = "lifetime_copy_test.logical_consumer.default"
    changed, report = compact(
        graph,
        (x,),
        barriers=(torch.ops.lifetime_copy_test.logical_consumer.default,),
        readonly=(ReadonlyOperator(target, "effects.v1"),),
        layout=()
        if contract_ports is None
        else (LayoutIndependentOperator(target, "values.v1", contract_ports),),
    )
    assert any(d.applied for d in report.decisions) is (contract_ports == (1,))
    if contract_ports == (1,):
        assert any(
            d.layout_contracts == ((target, "values.v1", (1,)),)
            for d in report.decisions
        )
    torch.testing.assert_close(changed(x), function(x), rtol=0, atol=0)


def test_layout_contract_without_effect_contract_does_not_bypass_fence():
    x = torch.randn(7, 12)
    graph = make_fx(lambda x: logical_consumer((x * 2)[:, :4], offset=x[:, :4]))(x)
    target = "lifetime_copy_test.logical_consumer.default"
    changed, report = compact(
        graph,
        (x,),
        barriers=(torch.ops.lifetime_copy_test.logical_consumer.default,),
        layout=(LayoutIndependentOperator(target, "values.v1", (0, 1)),),
    )
    assert not any(d.applied for d in report.decisions)
    assert any("effect or uncertified" in d.reason for d in report.decisions)
    torch.testing.assert_close(changed(x), graph(x), rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["input", "external", "writer"])
def test_input_external_and_written_backings_are_never_compacted(mode):
    x = torch.randn(7, 12)

    def function(x):
        backing = x if mode == "input" else x * 2
        view = backing[:, :4]
        if mode == "writer":
            backing.add_(1)
        result = view.sin()
        return (result, backing) if mode == "external" else result

    graph = make_fx(function)(x)
    changed, report = compact(graph, (x,), barriers=(torch.ops.aten.sin.default,))
    assert not any(d.applied for d in report.decisions)
    assert any("storage" in d.reason for d in report.decisions)
    torch.testing.assert_close(changed(x), function(x), rtol=0, atol=0)


def test_unchanged_dense_fresh_result_stops_layout_propagation():
    x = torch.randn(7, 12)

    def function(x):
        view = (x * 2)[:, :4]
        return layout_observer(view.sin())

    graph = make_fx(function)(x)
    # No contract is needed for the observer: sin returns an identically laid
    # out complete fresh dense tensor, independent of the candidate copy.
    changed, report = compact(graph, (x,), barriers=(torch.ops.aten.sin.default,))
    assert any(d.applied for d in report.decisions)
    torch.testing.assert_close(changed(x), function(x), rtol=0, atol=0)


def test_changed_outer_layout_is_not_silently_promoted():
    x = torch.randn(4, 12)
    graph = make_fx(lambda x: (x * 2).T[:6, :] + 2)(x)
    changed, report = compact(graph, (x,), barriers=(torch.ops.aten.add.Tensor,))
    assert not any(d.applied for d in report.decisions)
    assert any(
        d.reason == "changed layout reaches a task output" for d in report.decisions
    )
    assert changed.code == graph.code


def test_incompatible_downstream_view_rolls_back_the_entire_candidate():
    x = torch.randn(8, 4)

    def function(x):
        view = (x * 2).T[:, :4]
        return view.T.view(-1).sin()

    graph = make_fx(function)(x)
    source = [
        n for n in graph.graph.nodes if n.target is torch.ops.aten.permute.default
    ][-1].args[0]
    changed, report = compact(
        graph, (x,), names={source.name}, barriers=(torch.ops.aten.sin.default,)
    )
    assert not any(d.applied for d in report.decisions)
    assert any(
        "candidate geometry is unsupported" in d.reason for d in report.decisions
    )
    assert changed.code == graph.code
    torch.testing.assert_close(changed(x), function(x), rtol=0, atol=0)


def test_foreign_or_stale_analysis_and_contract_indices_fail():
    x = torch.randn(4, 12)
    graph, other = make_fx(torch.sin)(x), make_fx(torch.cos)(x)
    analysis = analyze_lifetimes(graph, (x,))
    with pytest.raises(CaptureError, match="stale or belongs"):
        compact_internal_views(
            other,
            analysis,
            (x,),
            selected=frozenset(),
            regions={},
            barriers=frozenset(),
        )
    for inputs in ((), (0, 0), (-1,), (True,)):
        with pytest.raises(ValueError, match="unique nonnegative"):
            LayoutIndependentOperator("test.op.default", "v1", inputs)
