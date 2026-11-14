"""Speculative compaction of related internal views, with layout contracts.

This internal pass returns a new graph. Effect certification alone cannot
authorize a copy: layout changes must be followed through fresh intermediates,
and all selected views of one semantic storage are considered together.
"""

from __future__ import annotations

import operator
from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx import GraphModule, Node
from torch.fx.passes.fake_tensor_prop import FakeTensorProp
from torch.utils._pytree import tree_flatten

from shadowspill.errors import CaptureError
from shadowspill.pytorch.capture.storage import capture_task_storage_contract
from shadowspill.pytorch.capture.storage.symbolic import _fresh_symbolic_arguments
from shadowspill.pytorch.capture.torch_deprecations import copy_graph_module
from shadowspill.task.storage import StorageRootKind

from .lifetime_analysis import LifetimeGraph, analyze_lifetimes
from .lifetime_policy import LayoutIndependentOperator as LayoutIndependentOperator
from .lifetime_policy import ReadonlyOperator

# These exact overloads compute logical values without reading storage gaps or
# interpreting explicit physical strides. Layout may still propagate through
# their outputs. In particular, a fresh pointwise result is not a clean boundary
# unless its complete dense geometry is unchanged. No raw as_strided/dtype-view,
# arbitrary dispatcher overload, mutation or opaque provider is inferred here.
_VALUE_INPUTS = {
    **{
        "aten." + target: frozenset({0})
        for target in (
            "alias.default",
            "detach.default",
            "diagonal.default",
            "expand.default",
            "permute.default",
            "select.int",
            "slice.Tensor",
            "split.Tensor",
            "split_with_sizes.default",
            "squeeze.default",
            "squeeze.dim",
            "squeeze.dims",
            "t.default",
            "transpose.int",
            "unbind.int",
            "unsqueeze.default",
            "view.default",
            "_unsafe_view.default",
            "clone.default",
            "_to_copy.default",
            "sin.default",
            "cos.default",
            "sigmoid.default",
            "tanh.default",
            "silu.default",
            "relu.default",
            "neg.default",
            "exp.default",
            "log.default",
            "sqrt.default",
            "rsqrt.default",
            "pow.Tensor_Scalar",
            "mean.dim",
            "sum.default",
            "sum.dim_IntList",
            "cumsum.default",
            "_softmax.default",
            "cat.default",
            "stack.default",
        )
    },
    **{
        "aten." + target: frozenset({0, 1})
        for target in (
            "add.Tensor",
            "sub.Tensor",
            "mul.Tensor",
            "mul.Scalar",
            "div.Tensor",
            "div.Scalar",
            "mm.default",
            "bmm.default",
            "_softmax_backward_data.default",
        )
    },
    **{
        "aten." + target: frozenset({0, 1, 2})
        for target in ("addmm.default", "baddbmm.default", "where.self")
    },
}


@dataclass(frozen=True, slots=True)
class CompactionDecision:
    source_indices: tuple[int, ...]
    applied: bool
    reason: str
    copy_bytes: int = 0
    obstruction_index: int | None = None
    barrier_bytes: tuple[tuple[int, int, int], ...] = ()
    layout_contracts: tuple[tuple[str, str, tuple[int, ...]], ...] = ()


@dataclass(frozen=True, slots=True)
class CompactionSchedule:
    decisions: tuple[CompactionDecision, ...]
    barrier_bytes: tuple[tuple[int, int, int], ...]


def _refresh(graph: GraphModule, inputs: tuple[object, ...]) -> None:
    # Never run real tensor kernels to inspect a candidate, including on CPU.
    mode = FakeTensorMode(allow_non_fake_inputs=True)
    with mode, torch.no_grad():
        FakeTensorProp(graph, mode=mode).propagate(*_fresh_symbolic_arguments(inputs))


def _geometry(value: torch.Tensor) -> tuple[object, ...]:
    return (
        tuple(value.shape),
        tuple(value.stride()),
        value.storage_offset(),
        value.dtype,
        value.device,
        value.layout,
        value.is_conj(),
        value.is_neg(),
    )


def _dense(value: torch.Tensor) -> bool:
    if value.layout is not torch.strided or value.storage_offset() != 0:
        return False
    extent = 1
    for stride, size in sorted(zip(value.stride(), value.shape, strict=True)):
        if size > 1:
            if stride != extent:
                return False
            extent *= size
    return True


def _clean_fresh(
    old: Node, new: Node, before: LifetimeGraph, after: LifetimeGraph
) -> bool:
    old_values, spec = tree_flatten(old.meta.get("val"))
    new_values, new_spec = tree_flatten(new.meta.get("val"))
    if spec != new_spec or not old_values:
        return False
    for left, right in zip(old_values, new_values, strict=True):
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            return False
        if _geometry(left) != _geometry(right) or not _dense(left):
            return False
    # Only an unchanged, complete, newly allocated dense result resets taint.
    # Alias views and packed/subset outputs cannot hide changed storage gaps.
    for analysis, node in ((before, old), (after, new)):
        roots = analysis.values[node]
        if not roots or len(roots) != len(old_values):
            return False
        if any(
            analysis.storage.roots[root].kind is not StorageRootKind.FRESH
            or analysis.storage.roots[root].producer_node != node.name
            for root in roots
        ):
            return False
        logical = sum(value.numel() * value.element_size() for value in old_values)
        if (
            sum(analysis.storage.roots[root].minimum_span_bytes for root in roots)
            != logical
        ):
            return False
    return True


def _layout_check(
    before: LifetimeGraph,
    after: LifetimeGraph,
    copies: frozenset[Node],
    contracts: Mapping[str, LayoutIndependentOperator],
    *,
    check_copy_inputs: bool = False,
) -> tuple[str | None, Node | None, tuple[tuple[str, str, tuple[int, ...]], ...]]:
    originals = {node.name: node for node in before.order}
    tainted = set(copies)
    used: dict[str, LayoutIndependentOperator] = {}
    for node in after.order:
        # A compaction seed is an explicit logical clone. A replay seed can
        # itself consume another replayed value: readonly permission alone
        # does not justify changing those inputs' alias relationships.
        if (node in copies and not check_copy_inputs) or not (
            tainted & set(node.all_input_nodes)
        ):
            continue
        if node.op == "output":
            return "changed layout reaches a task output", node, ()
        if node in after.fences:
            return "changed layout reaches an effect or uncertified provider", node, ()
        if node.target is operator.getitem:
            # Structural extraction preserves the producer's outstanding taint.
            tainted.add(node)
            continue
        if not isinstance(node.target, torch._ops.OpOverload):
            return "changed layout reaches an unqualified operation", node, ()
        schema = node.target._schema
        changed = set()
        for index, argument in enumerate(schema.arguments):
            value = (
                node.args[index]
                if index < len(node.args)
                else node.kwargs.get(argument.name)
            )
            leaves, _ = tree_flatten(value)
            if any(isinstance(leaf, Node) and leaf in tainted for leaf in leaves):
                changed.add(index)
        target = str(node.target)
        allowed = _VALUE_INPUTS.get(target, frozenset())
        if node.target.namespace != "aten" and target in contracts:
            contract = contracts[target]
            if any(index >= len(schema.arguments) for index in contract.inputs):
                return (
                    "layout contract argument index exceeds operator schema",
                    node,
                    (),
                )
            allowed = frozenset(contract.inputs)
            used[target] = contract
        if not changed <= allowed:
            return "changed input lacks layout-independent value semantics", node, ()
        if node in copies or not _clean_fresh(
            originals[node.name], node, before, after
        ):
            tainted.add(node)
    return (
        None,
        None,
        tuple(
            (target, contract.version, tuple(sorted(contract.inputs)))
            for target, contract in sorted(used.items())
        ),
    )


def compact_internal_views(
    graph: GraphModule,
    analysis: LifetimeGraph,
    example_inputs: tuple[object, ...],
    *,
    selected: frozenset[Node],
    regions: Mapping[Node, int],
    barriers: frozenset[Node],
    readonly: tuple[ReadonlyOperator, ...] = (),
    layout: tuple[LayoutIndependentOperator, ...] = (),
) -> tuple[GraphModule, CompactionSchedule]:
    """Copy selected escaping aliases jointly by storage, or record refusal.

    Input/external roots and any written storage are ineligible. Only uses in
    another execution region are redirected. Copies are producer-local; no
    arithmetic is replayed or moved. Call view preparation first if appropriate.
    A proposal must preserve the outer storage contract, not increase any
    selected barrier's complete internal live span and improve at least one.
    The input graph remains untouched, including on failed fake evaluation.
    """
    original = tuple(graph.graph.nodes)
    if analysis.order != original:
        raise CaptureError(
            "lifetime compaction analysis is stale or belongs to another graph"
        )
    indices = {node: index for index, node in enumerate(original)}
    if not (selected | barriers | regions.keys()) <= indices.keys():
        raise CaptureError(
            "lifetime compaction selections must belong to the analyzed graph"
        )
    contracts = {item.target: item for item in layout}
    if len(contracts) != len(layout):
        raise ValueError("layout operator targets must be unique")
    groups: dict[int, list[Node]] = {}
    decisions = []
    written = frozenset().union(*analysis.writes.values())
    for node in original:
        if node not in selected:
            continue
        roots = analysis.values[node]
        value = node.meta.get("val")
        parents = frozenset().union(*(analysis.values[p] for p in node.all_input_nodes))
        reason = None
        if node in barriers:
            reason = "protected memory barrier"
        elif node in analysis.fences or node.op != "call_function":
            reason = "compaction requires readonly alias operations"
        elif (
            not isinstance(value, torch.Tensor)
            or len(roots) != 1
            or not roots <= parents
        ):
            reason = "compaction requires a single tensor alias view"
        elif roots & analysis.external_roots or any(
            analysis.storage.roots[root].kind is not StorageRootKind.FRESH
            for root in roots
        ):
            reason = "input or externally observable storage"
        elif roots & written:
            reason = "storage has an ordered writer"
        elif (
            value.numel() == 0
            or value.numel() * value.element_size()
            >= analysis.storage.roots[next(iter(roots))].minimum_span_bytes
        ):
            reason = "view does not compact its semantic storage"
        elif node not in regions or not any(
            user in regions and regions[user] != regions[node] for user in node.users
        ):
            continue
        if reason:
            decisions.append(CompactionDecision((indices[node],), False, reason))
        else:
            groups.setdefault(next(iter(roots)), []).append(node)

    current = copy_graph_module(graph)
    _refresh(current, example_inputs)
    current_analysis = analyze_lifetimes(current, example_inputs, readonly=readonly)
    outer = capture_task_storage_contract(current, example_inputs)
    barrier_names = tuple(node.name for node in original if node in barriers)
    original_by_name = {node.name: node for node in original}
    initial_bytes = {node.name: analysis.live_bytes(node) for node in barriers}
    for nodes in groups.values():
        trial = copy_graph_module(current)
        by_name = {node.name: node for node in trial.graph.nodes}
        copies = set()
        copy_bytes = 0
        for source in nodes:
            node = by_name[source.name]
            consumers = tuple(
                by_name[user.name]
                for user in source.users
                if user in regions and regions[user] != regions[source]
            )
            with trial.graph.inserting_after(node):
                copied = trial.graph.call_function(
                    torch.ops.aten.clone.default,
                    (node,),
                    {"memory_format": torch.contiguous_format},
                )
            copied.meta = dict(node.meta)
            for consumer in consumers:
                consumer.replace_input_with(node, copied)
            copies.add(copied)
            value = source.meta["val"]
            copy_bytes += value.numel() * value.element_size()
        trial.graph.lint()
        trial.recompile()
        reason, obstruction = None, None
        used: tuple[tuple[str, str, tuple[int, ...]], ...] = ()
        inventory: tuple[tuple[int, int, int], ...] = ()
        try:
            _refresh(trial, example_inputs)
            proposed = analyze_lifetimes(trial, example_inputs, readonly=readonly)
            reason, obstruction, used = _layout_check(
                current_analysis, proposed, frozenset(copies), contracts
            )
            if (
                reason is None
                and capture_task_storage_contract(trial, example_inputs) != outer
            ):
                reason = "outer storage contract would change"
            if reason is None:
                old_nodes = {node.name: node for node in current_analysis.order}
                inventory = tuple(
                    (
                        indices[original_by_name[name]],
                        current_analysis.live_bytes(old_nodes[name]),
                        proposed.live_bytes(by_name[name]),
                    )
                    for name in barrier_names
                )
                if any(after > before for _, before, after in inventory):
                    reason = "protected storage would increase"
                elif not any(after < before for _, before, after in inventory):
                    reason = "no protected storage reduction"
        except (CaptureError, RuntimeError, ValueError) as exc:
            reason = f"candidate geometry is unsupported: {type(exc).__name__}"
        decisions.append(
            CompactionDecision(
                tuple(indices[node] for node in nodes),
                reason is None,
                reason or "compacted related internal views",
                copy_bytes,
                indices.get(original_by_name.get(obstruction.name))
                if obstruction
                else None,
                inventory,
                used,
            )
        )
        if reason is None:
            current, current_analysis = trial, proposed
    final_nodes = {node.name: node for node in current_analysis.order}
    return current, CompactionSchedule(
        tuple(decisions),
        tuple(
            (
                indices[original_by_name[name]],
                initial_bytes[name],
                current_analysis.live_bytes(final_nodes[name]),
            )
            for name in barrier_names
        ),
    )
