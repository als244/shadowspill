"""Independent value, VJP, alias and effect checks for lifetime movement."""

import json
from collections import Counter
from dataclasses import asdict

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from shadowspill.errors import CaptureError
from shadowspill.pytorch.capture.storage import capture_task_storage_contract
from shadowspill.pytorch.chunking.lifetime_analysis import (
    ReadonlyOperator,
    analyze_lifetimes,
)
from shadowspill.pytorch.chunking.lifetime_schedule import sink_readonly


def operation(graph, target):
    return next(node for node in graph.graph.nodes if node.target is target)


def selected_ancestors(node):
    result, pending = set(), [node]
    while pending:
        current = pending.pop()
        if current in result or current.op == "placeholder":
            continue
        result.add(current)
        pending.extend(current.all_input_nodes)
    return frozenset(result)


def test_statistics_can_cross_disjoint_gradient_writes_with_exact_vjps() -> None:
    inputs = (
        torch.randn(3, dtype=torch.float64),
        torch.randn(3, dtype=torch.float64),
        torch.randn(7, 4, dtype=torch.float64),
        torch.randn(4, 5, dtype=torch.float64),
        torch.randn(5, dtype=torch.float64),
    )

    def function(counts, previous, x, weight, prior):
        routed = x @ weight
        total = torch.stack((counts, previous)).sum(0).float()
        coefficient = total.div(17).unsqueeze(0).expand(2, -1)[0]
        prior.add_(routed.square().sum(0))
        return routed, prior, coefficient

    graph = make_fx(function)(*(value.clone() for value in inputs))
    output = next(node for node in graph.graph.nodes if node.op == "output")
    statistics = selected_ancestors(output.args[0][2])
    writer = operation(graph, torch.ops.aten.add_.Tensor)
    stack = operation(graph, torch.ops.aten.stack.default)
    original = tuple(graph.graph.nodes)
    counts = Counter(node.target for node in original)
    assert original.index(stack) < original.index(writer)
    result = sink_readonly(
        graph,
        analyze_lifetimes(graph, inputs),
        movable=statistics,
        barriers=frozenset(),
    )
    changed = tuple(graph.graph.nodes)
    assert changed.index(stack) > changed.index(writer)
    assert Counter(node.target for node in changed) == counts
    assert counts[torch.ops.aten.mm.default] == 1
    assert any(item.applied and item.crossed_writes for item in result.decisions)
    json.dumps(asdict(result))  # No live tensors, FX nodes or callable objects.

    left = tuple(value.clone().requires_grad_() for value in inputs[:4])
    right = tuple(value.clone().requires_grad_() for value in inputs[:4])
    expected = function(*left, inputs[4].clone())
    actual = graph(*right, inputs[4].clone())
    cotangents = tuple(torch.randn_like(value) for value in expected)
    expected_grads = torch.autograd.grad(expected, left, cotangents)
    actual_grads = torch.autograd.grad(actual, right, cotangents)
    for a, b in zip(
        (*expected, *expected_grads), (*actual, *actual_grads), strict=True
    ):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("aliases", [False, True])
def test_read_cannot_cross_a_write_through_an_aliased_input(aliases) -> None:
    x = torch.randn(7, 12)
    other = x[:, :6] if aliases else torch.randn(7, 6)

    def function(x, other):
        before = x.sin()
        other.add_(2)
        return before.cos(), other

    graph = make_fx(function)(x.clone(), other.clone())
    source = operation(graph, torch.ops.aten.sin.default)
    writer = operation(graph, torch.ops.aten.add_.Tensor)
    result = sink_readonly(
        graph,
        analyze_lifetimes(graph, (x, other)),
        movable=frozenset({source}),
        barriers=frozenset(),
    )
    order = tuple(graph.graph.nodes)
    assert (order.index(source) < order.index(writer)) is aliases
    assert result.decisions[0].applied is not aliases
    left, right = x.clone(), x.clone()
    expected = function(left, left[:, :6] if aliases else other.clone())
    actual = graph(right, right[:, :6] if aliases else other.clone())
    for a, b in zip(expected, actual, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_small_reduction_cannot_keep_large_packed_backing_across_barrier() -> None:
    x, weight, control = torch.randn(7, 4), torch.randn(4, 12), torch.randn(7)

    def function(x, weight, control):
        left, _ = (x @ weight).split(6, -1)
        reduced = left.sum(-1)
        barrier = control.cos()
        return reduced + barrier

    graph = make_fx(function)(x, weight, control)
    code = graph.code
    reduced = operation(graph, torch.ops.aten.sum.dim_IntList)
    barrier = operation(graph, torch.ops.aten.cos.default)
    result = sink_readonly(
        graph,
        analyze_lifetimes(graph, (x, weight, control)),
        movable=frozenset({reduced}),
        barriers=frozenset({barrier}),
    )
    assert graph.code == code
    assert result.decisions[0].reason == "protected storage would increase"
    assert result.decisions[0].barrier_bytes[0][1:] == (28, 336)
    assert result.barrier_bytes[0][1:] == (28, 28)


def test_pointwise_sinking_releases_backing_at_every_protected_barrier() -> None:
    x, control = torch.randn(7, 4), torch.randn(7, 4)

    def function(x, control):
        early = x.sin()
        first = control.cos()
        second = first.square()
        return early + second

    graph = make_fx(function)(x, control)
    original = graph(x, control)
    early = operation(graph, torch.ops.aten.sin.default)
    barriers = frozenset(
        node
        for node in graph.graph.nodes
        if node.target in {torch.ops.aten.cos.default, torch.ops.aten.pow.Tensor_Scalar}
    )
    result = sink_readonly(
        graph,
        analyze_lifetimes(graph, (x, control)),
        movable=frozenset({early}) | barriers,
        barriers=barriers,
    )
    assert all(after == before - 112 for _, before, after in result.barrier_bytes)
    assert sum(item.applied for item in result.decisions) == 1
    assert (
        sum(item.reason == "protected memory barrier" for item in result.decisions) == 2
    )
    torch.testing.assert_close(graph(x, control), original, rtol=0, atol=0)


@torch.library.custom_op("lifetime_schedule_test::opaque", mutates_args=())
def opaque(x: torch.Tensor) -> torch.Tensor:
    return x.cos()


@opaque.register_fake
def opaque_fake(x):
    return torch.empty_like(x)


@pytest.mark.parametrize("certified", [False, True])
def test_crossing_opaque_provider_requires_effect_contract(certified) -> None:
    x = torch.randn(7)

    def function(x):
        early = x.sin()
        barrier = opaque(x)
        return early + barrier

    graph = make_fx(function)(x)
    source = operation(graph, torch.ops.aten.sin.default)
    provider = torch.ops.lifetime_schedule_test.opaque.default
    readonly = (ReadonlyOperator(str(provider), "v1"),) if certified else ()
    result = sink_readonly(
        graph,
        analyze_lifetimes(graph, (x,), readonly=readonly),
        movable=frozenset({source}),
        barriers=frozenset(),
    )
    assert result.decisions[0].applied is certified
    torch.testing.assert_close(graph(x), function(x), rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["rng", "opaque", "writer"])
def test_effectful_movement_selection_is_explicitly_refused(kind) -> None:
    x = torch.randn(7)
    call = dict(rng=torch.rand_like, opaque=opaque, writer=lambda x: x.add_(2))[kind]
    graph = make_fx(lambda x: call(x).sin())(x.clone())
    selected = next(node for node in graph.graph.nodes if node.op == "call_function")
    with pytest.raises(CaptureError, match="cannot move"):
        sink_readonly(
            graph,
            analyze_lifetimes(graph, (x,)),
            movable=frozenset({selected}),
            barriers=frozenset(),
        )


def test_rng_fences_nonrandom_movement_without_consuming_state() -> None:
    x = torch.randn(7)

    def function(x):
        early = x.sin()
        random = torch.rand_like(x)
        return early + random

    graph = make_fx(function)(x)
    rng = torch.random.get_rng_state().clone()
    result = sink_readonly(
        graph,
        analyze_lifetimes(graph, (x,)),
        movable=frozenset({operation(graph, torch.ops.aten.sin.default)}),
        barriers=frozenset(),
    )
    assert not result.decisions[0].applied
    assert torch.equal(rng, torch.random.get_rng_state())


@pytest.mark.parametrize("effect_kind", ["registered", "legacy"])
def test_known_disjoint_writer_with_additional_effect_is_still_a_fence(
    effect_kind,
) -> None:
    from torch._higher_order_ops.effects import _EffectType, _register_effectful_op
    from torch.fx.node import _side_effectful_functions

    x, prior = torch.randn(7), torch.randn(7)

    def function(x, prior):
        early = x.sin()
        prior.add_(1)
        return early.cos(), prior

    graph = make_fx(function)(x, prior.clone())
    target = torch.ops.aten.add_.Tensor
    handle = None
    if effect_kind == "registered":
        handle = _register_effectful_op(target, _EffectType.ORDERED)
    else:
        assert target not in _side_effectful_functions
        _side_effectful_functions.add(target)
    try:
        result = sink_readonly(
            graph,
            analyze_lifetimes(graph, (x, prior)),
            movable=frozenset({operation(graph, torch.ops.aten.sin.default)}),
            barriers=frozenset(),
        )
        assert not result.decisions[0].applied
    finally:
        if handle is not None:
            handle.destroy()
        else:
            _side_effectful_functions.remove(target)


def test_selected_gradient_widening_stays_before_reduction() -> None:
    x, control = torch.randn(7, 4, dtype=torch.bfloat16), torch.randn(7, 4)

    def function(x, control):
        widened = x.float()
        barrier = control.sin()
        gradient = widened.sum(0)
        return barrier, gradient

    graph = make_fx(function)(x, control)
    conversion = operation(graph, torch.ops.aten._to_copy.default)
    original = graph.code
    result = sink_readonly(
        graph,
        analyze_lifetimes(graph, (x, control)),
        movable=frozenset({conversion}),
        barriers=frozenset(),
        gradient_leaves=(1,),
    )
    assert graph.code == original
    assert (
        result.decisions[0].reason
        == "preserve conversion before gradient chunk reduction"
    )
    assert result.protected_conversions == (result.decisions[0].node_index,)


def test_movement_keeps_external_alias_and_mutation_order() -> None:
    x, first, second = torch.randn(7, 12), torch.randn(7), torch.randn(7)

    def function(x, first, second):
        view = x[:, :6]
        first.add_(1)
        second.copy_(first)
        return view, first, second

    graph = make_fx(function)(x, first.clone(), second.clone())
    before = analyze_lifetimes(graph, (x, first, second))
    contract = capture_task_storage_contract(graph, (x, first, second))
    output = next(node for node in graph.graph.nodes if node.op == "output")
    sink_readonly(
        graph,
        before,
        movable=selected_ancestors(output.args[0][0]),
        barriers=frozenset(),
    )
    after = analyze_lifetimes(graph, (x, first, second))
    assert before.external_roots == after.external_roots
    assert capture_task_storage_contract(graph, (x, first, second)) == contract
    actual = graph(x, first.clone(), second.clone())
    expected = function(x, first.clone(), second.clone())
    assert actual[0].untyped_storage().data_ptr() == x.untyped_storage().data_ptr()
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_foreign_selections_and_stale_analysis_are_rejected() -> None:
    x = torch.randn(7)
    graph = make_fx(lambda x: x.sin() + x.cos())(x)
    other = make_fx(torch.sin)(x)
    analysis = analyze_lifetimes(graph, (x,))
    with pytest.raises(CaptureError, match="selections must belong"):
        sink_readonly(
            graph,
            analysis,
            movable=frozenset({operation(other, torch.ops.aten.sin.default)}),
            barriers=frozenset(),
        )
    operation(graph, torch.ops.aten.sin.default).prepend(
        operation(graph, torch.ops.aten.cos.default)
    )
    with pytest.raises(CaptureError, match="stale"):
        sink_readonly(graph, analysis, movable=frozenset(), barriers=frozenset())
