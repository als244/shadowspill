"""Explicit primal replay with dependency-version and backing guards.

No operation is implicitly classified as cheap. A caller supplies permissions
and primal provenance; only structural tuple extraction accompanies a selected
producer automatically. This pass leaves the input graph untouched.
"""

from __future__ import annotations

import operator
from collections.abc import Mapping
from dataclasses import dataclass

from torch.fx import GraphModule, Node
from torch.fx.node import map_arg

from shadowspill.errors import CaptureError
from shadowspill.pytorch.capture.storage import capture_task_storage_contract
from shadowspill.pytorch.capture.torch_deprecations import copy_graph_module

from .lifetime_analysis import LifetimeGraph, analyze_lifetimes
from .lifetime_compaction import _layout_check, _refresh
from .lifetime_policy import LayoutIndependentOperator, ReadonlyOperator
from .lifetime_schedule import _gradient_conversions, _only_tensor_write


@dataclass(frozen=True, slots=True)
class ReplayDecision:
    consumer_index: int | None
    destination_region: int | None
    source_indices: tuple[int, ...]
    applied: bool
    reason: str
    obstructions: tuple[int, ...] = ()
    barrier_bytes: tuple[tuple[int, int, int], ...] = ()
    layout_contracts: tuple[tuple[str, str, tuple[int, ...]], ...] = ()


@dataclass(frozen=True, slots=True)
class ReplayCopy:
    source_index: int
    result_index: int
    destination_region: int


@dataclass(frozen=True, slots=True)
class ReplaySchedule:
    decisions: tuple[ReplayDecision, ...]
    barrier_bytes: tuple[tuple[int, int, int], ...]
    protected_conversions: tuple[int, ...]
    copies: tuple[ReplayCopy, ...]


@dataclass(frozen=True)
class _Proposal:
    graph: GraphModule
    memo: dict[tuple[Node, int], str]
    added: tuple[Node, ...]
    refusals: dict[Node, tuple[int, ...]]
    changed: bool


def _propose(
    current: GraphModule,
    analysis: LifetimeGraph,
    consumer: Node,
    indices: Mapping[Node, int],
    eligible: set[Node],
    regions: Mapping[Node, int],
    memo: Mapping[tuple[Node, int], str],
) -> _Proposal:
    trial = copy_graph_module(current)
    by_name: dict[str, Node] = {node.name: node for node in trial.graph.nodes}
    destination = regions[consumer]
    destination_node = by_name[consumer.name]
    proposed_memo = dict(memo)
    added: list[Node] = []
    refusals: dict[Node, tuple[int, ...]] = {}

    def localize(source: Node) -> Node:
        if (
            source not in eligible
            or source not in regions
            or regions[source] == destination
        ):
            return by_name[source.name]
        key = source, destination
        if key in proposed_memo:
            return by_name[proposed_memo[key]]
        # Compare the original read to the proposed new read. An immutable
        # output does not imply that its source backing still has that version.
        guarded = analysis.reads[source] | analysis.values[source]
        crossed = analysis.order[indices[source] + 1 : indices[consumer]]
        blocked = tuple(
            indices[node]
            for node in crossed
            if node in analysis.fences
            and not (
                _only_tensor_write(node)
                and analysis.writes[node]
                and not analysis.writes[node] & guarded
            )
        )
        if blocked:
            refusals[source] = blocked
            return by_name[source.name]
        parents = {parent: localize(parent) for parent in source.all_input_nodes}
        if source.target is operator.getitem:
            producer = source.args[0]
            assert isinstance(producer, Node)  # Eligibility requires a producer.
            if parents[producer] is by_name[producer.name]:
                # Do not send the original opaque tuple across a region only
                # to duplicate its extraction. Replay the producer as a unit.
                return by_name[source.name]
        with trial.graph.inserting_before(destination_node):
            duplicate = trial.graph.node_copy(source, parents.__getitem__)
        duplicate.meta = dict(source.meta)
        by_name[duplicate.name] = duplicate
        proposed_memo[key] = duplicate.name
        added.append(source)
        return duplicate

    # Original arguments identify permissions/provenance. Current arguments
    # may already refer to previously accepted copies.
    args = map_arg(consumer.args, localize)
    kwargs = map_arg(consumer.kwargs, localize)
    changed = args != destination_node.args or kwargs != destination_node.kwargs
    destination_node.args, destination_node.kwargs = args, kwargs
    trial.graph.lint()
    trial.recompile()
    return _Proposal(trial, proposed_memo, tuple(added), refusals, changed)


def replay_primal(
    graph: GraphModule,
    analysis: LifetimeGraph,
    example_inputs: tuple[object, ...],
    *,
    selected: frozenset[Node],
    primal: frozenset[Node],
    regions: Mapping[Node, int],
    barriers: frozenset[Node],
    excluded: frozenset[Node] = frozenset(),
    readonly: tuple[ReadonlyOperator, ...] = (),
    layout: tuple[LayoutIndependentOperator, ...] = (),
    gradient_leaves: tuple[int, ...] | None = None,
) -> tuple[GraphModule, ReplaySchedule]:
    """Replay permitted immutable primal dependencies in consumer regions.

    Copies are memoized per original operation and destination region. Regions
    containing a protected barrier never receive replay. Selected operations
    need primal provenance, readonly effects and unobserved/unwritten outputs.
    Their input versions must survive until the copy, crossing only certified
    disjoint tensor writes. The conversion-before-gradient-reduction invariant
    applies independently of caller permissions or provenance.

    Value/layout contracts protect consumers from changed alias relationships.
    Each proposal preserves the outer storage contract and cannot increase any
    selected barrier's internal live span. Nonincreasing intermediate proposals
    are permitted so the last redirected consumer can release an old backing.
    Physical cost and admission still require profiling the final composition.
    """
    original = tuple(graph.graph.nodes)
    if analysis.order != original:
        raise CaptureError(
            "lifetime replay analysis is stale or belongs to another graph"
        )
    indices = {node: index for index, node in enumerate(original)}
    if not (selected | primal | barriers | excluded | regions.keys()) <= indices.keys():
        raise CaptureError(
            "lifetime replay selections must belong to the analyzed graph"
        )
    contracts = {item.target: item for item in layout}
    if len(contracts) != len(layout):
        raise ValueError("layout operator targets must be unique")
    protected = _gradient_conversions(graph, gradient_leaves)
    written = frozenset().union(*analysis.writes.values())
    eligible: set[Node] = set()
    decisions: list[ReplayDecision] = []
    for node in original:
        structural = (
            node.target is operator.getitem
            and isinstance(node.args[0], Node)
            and node.args[0] in eligible
            and node not in excluded
        )
        if node not in selected and not structural:
            continue
        reason = None
        if node in barriers:
            reason = "protected memory barrier"
        elif node in excluded:
            reason = "replay explicitly excluded"
        elif node in protected:
            reason = "preserve conversion before gradient chunk reduction"
        elif node not in primal:
            reason = "replay requires captured primal provenance"
        elif node.op != "call_function" or node in analysis.fences:
            reason = "replay requires certified readonly effects"
        elif node.target is operator.getitem and not isinstance(node.args[0], Node):
            reason = "replay requires an explicit tuple producer"
        elif analysis.values[node] & analysis.external_roots:
            reason = "externally observable output backing"
        elif analysis.values[node] & written:
            reason = "output backing has an ordered writer"
        if reason is None:
            eligible.add(node)
        else:
            decisions.append(
                ReplayDecision(None, None, (indices[node],), False, reason)
            )

    current = copy_graph_module(graph)
    _refresh(current, example_inputs)
    current_analysis = analyze_lifetimes(current, example_inputs, readonly=readonly)
    outer = capture_task_storage_contract(current, example_inputs)
    barrier_nodes = tuple(node for node in original if node in barriers)
    blocked_regions = {regions[node] for node in barriers if node in regions}
    memo: dict[tuple[Node, int], str] = {}
    for consumer in original:
        if consumer not in regions or regions[consumer] in blocked_regions:
            continue
        destination = regions[consumer]
        if not any(
            source in eligible and source in regions and regions[source] != destination
            for source in consumer.all_input_nodes
        ):
            continue
        proposal = _propose(
            current, analysis, consumer, indices, eligible, regions, memo
        )
        for source, obstructions in proposal.refusals.items():
            decisions.append(
                ReplayDecision(
                    indices[consumer],
                    destination,
                    (indices[source],),
                    False,
                    "crossed effect or changed dependency version",
                    obstructions,
                )
            )
        if not proposal.changed:
            continue
        trial, proposed_memo, added = proposal.graph, proposal.memo, proposal.added
        by_name = {node.name: node for node in trial.graph.nodes}
        reason, obstruction = None, None
        inventory: tuple[tuple[int, int, int], ...] = ()
        used: tuple[tuple[str, str, tuple[int, ...]], ...] = ()
        try:
            _refresh(trial, example_inputs)
            proposed = analyze_lifetimes(trial, example_inputs, readonly=readonly)
            reason, obstruction, used = _layout_check(
                current_analysis,
                proposed,
                frozenset(by_name[name] for name in proposed_memo.values()),
                contracts,
                check_copy_inputs=True,
            )
            if (
                reason is None
                and capture_task_storage_contract(trial, example_inputs) != outer
            ):
                reason = "outer storage contract would change"
            old_nodes = {node.name: node for node in current_analysis.order}
            inventory = tuple(
                (
                    indices[node],
                    current_analysis.live_bytes(old_nodes[node.name]),
                    proposed.live_bytes(by_name[node.name]),
                )
                for node in barrier_nodes
            )
            if reason is None and any(after > before for _, before, after in inventory):
                reason = "protected backing would increase"
        except (CaptureError, RuntimeError, ValueError) as exc:
            reason = f"candidate geometry is unsupported: {type(exc).__name__}"
        # An obstruction can be a newly replayed consumer. Report its original
        # occurrence rather than requiring every trial node to exist already.
        original_index = {node.name: indices[node] for node in original}
        original_index.update(
            (name, indices[source]) for (source, _), name in proposed_memo.items()
        )
        decisions.append(
            ReplayDecision(
                indices[consumer],
                destination,
                tuple(indices[node] for node in added),
                reason is None,
                reason or "replayed primal dependencies",
                (original_index[obstruction.name],) if obstruction else (),
                inventory,
                used,
            )
        )
        if reason is None:
            current, current_analysis, memo = trial, proposed, proposed_memo

    # Keep all unknown effects and protected barriers, even when FX regards an
    # opaque dispatcher call as pure. Only dead certified readonly work drops.
    keep = {node.name for node in current_analysis.fences} | {
        node.name for node in barriers
    }
    if memo:
        current.graph.eliminate_dead_code(
            is_impure_node=lambda node: (
                node.is_impure(impure_random=True) or node.name in keep
            )
        )
    current.graph.lint()
    current.recompile()
    final = analyze_lifetimes(current, example_inputs, readonly=readonly)
    final_nodes = {node.name: node for node in final.order}
    final_indices = {node: index for index, node in enumerate(final.order)}
    if capture_task_storage_contract(current, example_inputs) != outer:
        raise CaptureError("primal replay cleanup changed the outer storage contract")
    return current, ReplaySchedule(
        tuple(decisions),
        tuple(
            (
                indices[node],
                analysis.live_bytes(node),
                final.live_bytes(final_nodes[node.name]),
            )
            for node in barrier_nodes
        ),
        tuple(indices[node] for node in original if node in protected),
        tuple(
            ReplayCopy(indices[source], final_indices[final_nodes[name]], destination)
            for (source, destination), name in memo.items()
            if name in final_nodes
        ),
    )
