"""Prepare internal alias views without changing storage or computation.

This is the semantic prerequisite for considering sibling views together during
compaction. It does not copy values, compact storage, or replay producers.
"""

from __future__ import annotations

from dataclasses import replace

from torch.fx import GraphModule, Node

from shadowspill.errors import CaptureError
from shadowspill.task.storage import StorageRootKind

from .lifetime_analysis import LifetimeGraph
from .lifetime_schedule import LifetimeSchedule, MovementDecision, _only_tensor_write


def prepare_internal_views(
    graph: GraphModule,
    analysis: LifetimeGraph,
    *,
    selected: frozenset[Node],
    barriers: frozenset[Node],
) -> LifetimeSchedule:
    """Place selected internal aliases just after their last dependency.

    Semantic roots, rather than FakeTensor storage addresses, establish that a
    node only exposes an existing backing. Views of inputs and externally
    observable backings are left unchanged. Effects and overlapping writes
    constrain movement; every protected backing inventory must not increase.
    Decisions' destination indices identify the dependency *after* which a
    view is placed. All indices refer to this pass's original graph.
    """
    original = tuple(graph.graph.nodes)
    if analysis.order != original:
        raise CaptureError(
            "lifetime view analysis is stale or belongs to another graph"
        )
    indices = {node: index for index, node in enumerate(original)}
    if not selected <= indices.keys() or not barriers <= indices.keys():
        raise CaptureError("lifetime view selections must belong to the analyzed graph")
    ordered_barriers = tuple(node for node in original if node in barriers)
    initial = {node: analysis.live_bytes(node) for node in ordered_barriers}
    current = analysis
    decisions = []
    for node in original:
        if node not in selected:
            continue
        roots = analysis.values[node]
        parents = frozenset().union(
            *(analysis.values[parent] for parent in node.all_input_nodes)
        )
        reason = None
        if node in barriers:
            reason = "protected memory barrier"
        elif node.op != "call_function" or node in analysis.fences:
            reason = "view preparation requires readonly alias operations"
        elif not roots or not roots <= parents:
            reason = "operation is not a semantic alias view"
        elif roots & analysis.external_roots or any(
            analysis.storage.roots[root].kind is not StorageRootKind.FRESH
            for root in roots
        ):
            reason = "input or externally observable backing"
        if reason is not None:
            decisions.append(MovementDecision(indices[node], None, False, reason))
            continue
        order = current.order
        rank = {item: position for position, item in enumerate(order)}
        dependency = max(node.all_input_nodes, key=rank.__getitem__)
        start, stop = rank[dependency], rank[node]
        crossed = order[start + 1 : stop]
        if not crossed:
            continue
        guarded = analysis.reads[node] | roots
        blocked = tuple(
            item
            for item in crossed
            if item in analysis.fences
            and not (
                _only_tensor_write(item)
                and analysis.writes[item]
                and not analysis.writes[item] & guarded
            )
        )
        if blocked:
            decisions.append(
                MovementDecision(
                    indices[node],
                    indices[dependency],
                    False,
                    "crossed effect or overlapping writer",
                    obstructions=tuple(indices[item] for item in blocked),
                )
            )
            continue
        proposed = replace(
            current,
            order=(*order[: start + 1], node, *crossed, *order[stop + 1 :]),
        )
        inventory = tuple(
            (
                indices[barrier],
                current.live_bytes(barrier),
                proposed.live_bytes(barrier),
            )
            for barrier in ordered_barriers
        )
        increases = tuple(index for index, before, after in inventory if after > before)
        decisions.append(
            MovementDecision(
                indices[node],
                indices[dependency],
                not increases,
                "protected backing would increase"
                if increases
                else "prepared internal view",
                obstructions=increases,
                crossed_writes=tuple(
                    indices[item] for item in crossed if analysis.writes[item]
                ),
                barrier_bytes=inventory,
            )
        )
        if not increases:
            dependency.append(node)
            current = proposed
    graph.graph.lint()
    graph.recompile()
    return LifetimeSchedule(
        tuple(decisions),
        tuple(indices[node] for node in current.order),
        tuple(
            (indices[node], initial[node], current.live_bytes(node))
            for node in ordered_barriers
        ),
        (),
    )
