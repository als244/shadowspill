"""Ownership/effect counterexamples for execution lifetime analysis."""

from __future__ import annotations

import operator

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from shadowspill.errors import CaptureError
from shadowspill.pytorch.capture.torch_deprecations import copy_graph_module
from shadowspill.pytorch.chunking.lifetime_analysis import (
    ReadonlyOperator,
    analyze_lifetimes,
)
from shadowspill.task.storage import StorageRootKind


def operation(graph, target):
    return next(node for node in graph.graph.nodes if node.target is target)


def test_packed_backing_is_preserved_when_small_reduction_moves() -> None:
    x, weight, control = torch.randn(7, 4), torch.randn(4, 12), torch.randn(7)

    def function(x, weight, control):
        left, _right = (x @ weight).split(6, -1)
        reduced = left.sum(-1)
        barrier = control.cos()
        return reduced + barrier

    graph = make_fx(function)(x, weight, control)
    before = analyze_lifetimes(graph, (x, weight, control))
    barrier = operation(graph, torch.ops.aten.cos.default)
    assert before.live_bytes(barrier) == 28
    changed = copy_graph_module(graph)
    reduced = operation(changed, torch.ops.aten.sum.dim_IntList)
    next(iter(reduced.users)).prepend(reduced)
    changed.recompile()
    after = analyze_lifetimes(changed, (x, weight, control))
    assert after.live_bytes(operation(changed, torch.ops.aten.cos.default)) == 336
    torch.testing.assert_close(graph(x, weight, control), changed(x, weight, control))


def test_intermediate_metadata_addresses_do_not_define_ownership() -> None:
    x = torch.randn(7, 12)
    graph = make_fx(lambda x: (x.cos().split(6, -1)[1] + 2).sin())(x)
    first = analyze_lifetimes(graph, (x,))

    def unrelated(value):
        if isinstance(value, torch.Tensor):
            return torch.empty_strided(
                value.shape, value.stride(), dtype=value.dtype, device="meta"
            )
        if isinstance(value, (tuple, list)):
            return type(value)(unrelated(child) for child in value)
        return value

    for node in graph.graph.nodes:
        if "val" in node.meta:
            node.meta["val"] = unrelated(node.meta["val"])
    second = analyze_lifetimes(graph, (x,))
    assert first.storage == second.storage
    assert first.values == second.values
    assert first.reads == second.reads


def test_unsafe_view_alias_is_normalized_only_in_analysis_copy() -> None:
    x = torch.randn(7, 12)
    graph = make_fx(lambda x: x.sin().view(7, 2, 6))(x)
    view = operation(graph, torch.ops.aten.view.default)
    view.target = torch.ops.aten._unsafe_view.default
    graph.recompile()
    original_code = graph.code
    result = analyze_lifetimes(graph, (x,))
    sin = operation(graph, torch.ops.aten.sin.default)
    assert result.values[sin] == result.values[view] == result.external_roots
    assert len(result.values[sin]) == 1
    assert graph.code == original_code
    assert view.target is torch.ops.aten._unsafe_view.default


@pytest.mark.parametrize(
    "target", [torch.ops.aten.reshape.default, torch.ops.aten.contiguous.default]
)
def test_conditional_alias_is_refused_even_with_readonly_contract(target) -> None:
    x = torch.randn(7, 12)
    graph = make_fx(lambda x: x.view(7, 12))(x)
    view = operation(graph, torch.ops.aten.view.default)
    view.target = target
    if target is torch.ops.aten.contiguous.default:
        view.args = (view.args[0],)
    graph.recompile()
    with pytest.raises(CaptureError, match="always-alias storage contract"):
        analyze_lifetimes(
            graph, (x,), readonly=(ReadonlyOperator(str(target), "test-v1"),)
        )


def test_input_alias_groups_and_writes_through_views_are_preserved() -> None:
    base = torch.randn(7, 12)
    x, alias = base[:, :6], base[:, 6:]

    def function(x, alias):
        selected = alias[:, :3]
        selected.add_(2)
        return x, selected

    graph = make_fx(function)(x.clone(), alias.clone())
    original_base = base.clone()
    result = analyze_lifetimes(graph, (x, alias))
    inputs = [node for node in graph.graph.nodes if node.op == "placeholder"]
    write = operation(graph, torch.ops.aten.add_.Tensor)
    assert result.values[inputs[0]] == result.values[inputs[1]]
    assert result.writes[write] == result.values[inputs[0]]
    assert result.external_roots == result.values[inputs[0]]
    assert result.fences[write] == "ordered mutation"
    assert all(
        result.storage.roots[i].kind is StorageRootKind.INPUT
        for i in result.external_roots
    )
    assert torch.equal(base, original_base), "analysis executed a real input mutation"


def test_tuple_selection_does_not_read_other_return_backings() -> None:
    x = torch.randn(7, 12)

    def function(x):
        values, indices = torch.max(x, dim=1)
        return values.sin(), indices

    graph = make_fx(function)(x)
    result = analyze_lifetimes(graph, (x,))
    selections = [node for node in graph.graph.nodes if node.target is operator.getitem]
    assert len(selections) == 2
    assert not result.values[selections[0]] & result.values[selections[1]]
    assert all(result.reads[node] == result.values[node] for node in selections)


def test_list_views_share_one_semantic_backing() -> None:
    x = torch.randn(7, 12)
    graph = make_fx(lambda x: x.cos().split(6, -1))(x)
    result = analyze_lifetimes(graph, (x,))
    source = operation(graph, torch.ops.aten.cos.default)
    selections = [node for node in graph.graph.nodes if node.target is operator.getitem]
    assert all(result.values[node] == result.values[source] for node in selections)
    root = result.storage.roots[next(iter(result.values[source]))]
    assert root.minimum_span_bytes == 7 * 12 * 4


@pytest.mark.parametrize("kind", ["mutation", "rng"])
def test_readonly_assertion_cannot_override_observable_effects(kind) -> None:
    x = torch.randn(7, 12)
    graph = make_fx(lambda x: x.add_(1) if kind == "mutation" else torch.rand_like(x))(
        x
    )
    target = (
        torch.ops.aten.add_.Tensor
        if kind == "mutation"
        else torch.ops.aten.rand_like.default
    )
    result = analyze_lifetimes(graph, (x,))
    assert operation(graph, target) in result.fences
    with pytest.raises(CaptureError, match="cannot override mutation/RNG/effects"):
        analyze_lifetimes(
            graph, (x,), readonly=(ReadonlyOperator(str(target), "test-v1"),)
        )


@torch.library.custom_op("lifetime_analysis_test::readonly", mutates_args=())
def opaque_readonly(x: torch.Tensor) -> torch.Tensor:
    return x.sin()


@opaque_readonly.register_fake
def opaque_readonly_fake(x):
    return torch.empty_like(x)


def test_unknown_opaque_effects_fence_until_versioned_certification() -> None:
    x = torch.randn(7, 12)
    graph = make_fx(lambda x: opaque_readonly(x).cos())(x)
    target = torch.ops.lifetime_analysis_test.readonly.default
    node = operation(graph, target)
    before = analyze_lifetimes(graph, (x,))
    assert node in before.fences
    after = analyze_lifetimes(
        graph, (x,), readonly=(ReadonlyOperator(str(target), "v1"),)
    )
    assert node not in after.fences
    assert after.readonly_contracts == {str(target): "v1"}
    assert before.storage == after.storage
    assert before.external_roots == after.external_roots
    with pytest.raises(ValueError, match="conflicting readonly"):
        analyze_lifetimes(
            graph,
            (x,),
            readonly=(
                ReadonlyOperator(str(target), "v1"),
                ReadonlyOperator(str(target), "v2"),
            ),
        )


def test_analysis_cannot_mutate_records_or_accept_foreign_barriers() -> None:
    x = torch.randn(7)
    first, second = make_fx(torch.sin)(x), make_fx(torch.cos)(x)
    result = analyze_lifetimes(first, (x,))
    with pytest.raises(TypeError):
        result.values[operation(first, torch.ops.aten.sin.default)] = frozenset()
    with pytest.raises(ValueError, match="does not belong"):
        result.live_bytes(operation(second, torch.ops.aten.cos.default))


def test_unused_pure_values_do_not_extend_storage_lifetimes() -> None:
    x = torch.randn(7, 12)

    def function(x):
        unused = x.sin()
        barrier = x.cos()
        unused.square()
        return barrier

    graph = make_fx(function)(x)
    result = analyze_lifetimes(graph, (x,))
    assert result.live_bytes(operation(graph, torch.ops.aten.cos.default)) == 0


def test_unknown_python_is_not_evaluated_to_obtain_geometry() -> None:
    x = torch.randn(7)
    graph = make_fx(torch.sin)(x)
    invoked = []

    def unknown(value):
        invoked.append(True)
        return value.sin()

    node = operation(graph, torch.ops.aten.sin.default)
    node.target = unknown
    graph.recompile()
    with pytest.raises(CaptureError, match="requires dispatcher operations"):
        analyze_lifetimes(graph, (x,))
    assert not invoked


def test_readonly_cannot_override_registered_ordered_effect() -> None:
    from torch._higher_order_ops.effects import _EffectType, _register_effectful_op

    x = torch.randn(7)
    graph = make_fx(lambda x: opaque_readonly(x))(x)
    target = torch.ops.lifetime_analysis_test.readonly.default
    handle = _register_effectful_op(target, _EffectType.ORDERED)
    try:
        result = analyze_lifetimes(graph, (x,))
        assert (
            result.fences[operation(graph, target)] == "RNG or other observable effect"
        )
        with pytest.raises(CaptureError, match="cannot override mutation/RNG/effects"):
            analyze_lifetimes(
                graph, (x,), readonly=(ReadonlyOperator(str(target), "v1"),)
            )
    finally:
        handle.destroy()


def test_readonly_does_not_certify_unknown_opaque_alias() -> None:
    from torch.library import _scoped_library

    x = torch.randn(7)
    graph = make_fx(lambda x: x.view(7))(x)
    with _scoped_library("lifetime_analysis_alias_test", "FRAGMENT") as library:
        library.define("alias(Tensor(a) x) -> Tensor(a)")
        target = torch.ops.lifetime_analysis_alias_test.alias.default
        node = operation(graph, torch.ops.aten.view.default)
        node.target, node.args = target, (node.args[0],)
        graph.recompile()
        with pytest.raises(CaptureError, match="always-alias storage contract"):
            analyze_lifetimes(
                graph, (x,), readonly=(ReadonlyOperator(str(target), "v1"),)
            )


def test_negative_tuple_index_keeps_the_correct_return_root() -> None:
    x = torch.randn(7, 12)
    graph = make_fx(lambda x: torch.max(x, dim=1)[1])(x)
    selected = next(
        node
        for node in graph.graph.nodes
        if node.target is operator.getitem and node.args[1] == 1
    )
    before = analyze_lifetimes(graph, (x,))
    selected.args = (selected.args[0], -1)
    graph.recompile()
    after = analyze_lifetimes(graph, (x,))
    assert before.values == after.values
    assert before.external_roots == after.external_roots
    assert selected.args[1] == -1


def test_analysis_does_not_consume_real_rng_state() -> None:
    x = torch.randn(7)
    graph = make_fx(lambda x: torch.rand_like(x))(x)
    state = torch.random.get_rng_state().clone()
    analyze_lifetimes(graph, (x,))
    assert torch.equal(torch.random.get_rng_state(), state)


def test_nested_fresh_returns_are_not_born_at_a_later_unpack() -> None:
    from torch.library import _scoped_library

    x = torch.randn(7)
    graph = make_fx(torch.sin)(x)
    with _scoped_library("lifetime_analysis_nested_test", "FRAGMENT") as library:
        library.define("produce(Tensor x) -> (Tensor[], Tensor)")
        target = torch.ops.lifetime_analysis_nested_test.produce.default
        node = operation(graph, torch.ops.aten.sin.default)
        node.target = target
        node.meta["val"] = ([torch.empty_like(x)], torch.empty_like(x))
        graph.recompile()
        with pytest.raises(CaptureError, match="nested lifetime return trees"):
            analyze_lifetimes(graph, (x,))
