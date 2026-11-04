"""Conservative movement within one complete checkpoint task.

This pass only changes operation order. It neither replays work nor changes
storage or arithmetic. Public policy resolution and build integration remain
separate from the semantic checks here.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from torch.fx import GraphModule, Node
from torch.utils._pytree import tree_flatten

from shadowspill.errors import CaptureError
from shadowspill.pytorch.capture.gradients.nodes import _CONVERTING, _MOVING_ONLY

from .lifetime_analysis import LifetimeGraph

_GRADIENT_ASSEMBLY = (
    _CONVERTING
    | _MOVING_ONLY
    | {
        torch.ops.aten.add.Tensor,
        torch.ops.aten.add_.Tensor,
        torch.ops.aten.cat.default,
        torch.ops.aten.sum.default,
        torch.ops.aten.sum.dim_IntList,
    }
)
_TENSOR_WRITES = frozenset(
    {
        torch.ops.aten.add_.Tensor,
        torch.ops.aten.copy_.default,
        torch.ops.aten.zero_.default,
    }
)


@dataclass(frozen=True, slots=True)
class MovementDecision:
    """Indices refer to the original graph, without occurrence-specific names."""

    node_index: int
    destination_index: int | None
    applied: bool
    reason: str
    obstructions: tuple[int, ...] = ()
    crossed_writes: tuple[int, ...] = ()
    barrier_bytes: tuple[tuple[int, int, int], ...] = ()


@dataclass(frozen=True, slots=True)
class LifetimeSchedule:
    decisions: tuple[MovementDecision, ...]
    order: tuple[int, ...]
    barrier_bytes: tuple[tuple[int, int, int], ...]
    protected_conversions: tuple[int, ...]


def _gradient_conversions(
    graph: GraphModule, output_leaves: tuple[int, ...] | None
) -> frozenset[Node]:
    output = next(node for node in graph.graph.nodes if node.op == "output")
    leaves, _ = tree_flatten(output.args[0])
    if output_leaves is not None:
        if any(index < 0 or index >= len(leaves) for index in output_leaves):
            raise CaptureError("gradient lifetime output index is outside the task")
        leaves = [leaves[index] for index in output_leaves]
    pending, seen, protected = list(leaves), set(), set()
    while pending:
        node = pending.pop()
        if (
            not isinstance(node, Node)
            or node in seen
            or node.target not in _GRADIENT_ASSEMBLY
        ):
            continue
        seen.add(node)
        if node.target in _CONVERTING:
            protected.add(node)
        pending.extend(node.all_input_nodes)
    return frozenset(protected)


def _only_tensor_write(node: Node) -> bool:
    # A mutable schema is not proof that mutation is the *only* effect. Unknown
    # custom writers remain fences, and explicit effect registrations win even
    # for these known ATen operations. is_impure() alone cannot distinguish the
    # cases because it returns immediately for every mutable schema.
    from torch._higher_order_ops.effects import _get_effect
    from torch.fx.node import _side_effectful_functions

    target = node.target
    return (
        target in _TENSOR_WRITES
        and isinstance(target, torch._ops.OpOverload)
        and _get_effect(target) is None
        and target not in _side_effectful_functions
        and not getattr(target, "_nondeterministic_seeded", False)
    )


def sink_readonly(
    graph: GraphModule,
    analysis: LifetimeGraph,
    *,
    movable: frozenset[Node],
    barriers: frozenset[Node],
    gradient_leaves: tuple[int, ...] | None = None,
) -> LifetimeSchedule:
    """Move selected readonly nodes toward first use with no duplicated work.

    The caller must analyze this exact graph immediately before this pass.
    Known tensor writes can be crossed only if their backing roots are disjoint
    from the moved operation's reads and outputs. Writers themselves never move.
    Unknown effects, RNG and uncertified opaque providers remain fences.

    Every candidate is checked at *all* selected memory barriers using complete
    internal backing lifetimes, including siblings of partial views. These
    barriers are not execution cuts or checkpoint retention choices. Semantic
    nonincrease does not replace profiling and physical memory admission.
    """
    original = tuple(graph.graph.nodes)
    if analysis.order != original:
        raise CaptureError("lifetime analysis is stale or belongs to another graph")
    indices = {node: index for index, node in enumerate(original)}
    if not movable <= indices.keys() or not barriers <= indices.keys():
        raise CaptureError("lifetime selections must belong to the analyzed graph")
    for node in movable:
        if node.op != "call_function" or node in analysis.fences:
            raise CaptureError(
                f"cannot move {node.target}: "
                f"{analysis.fences.get(node, 'not a readonly tensor operation')}"
            )
    protected = _gradient_conversions(graph, gradient_leaves)
    ordered_barriers = tuple(node for node in original if node in barriers)
    initial_bytes = {node: analysis.live_bytes(node) for node in ordered_barriers}
    decisions: list[MovementDecision] = []
    current = analysis
    for node in reversed(original):
        if node not in movable:
            continue
        index = indices[node]
        if node in barriers or node in protected:
            decisions.append(
                MovementDecision(
                    index,
                    None,
                    False,
                    "protected memory barrier"
                    if node in barriers
                    else "preserve conversion before gradient chunk reduction",
                )
            )
            continue
        if not node.users:
            continue
        order = current.order
        rank = {item: position for position, item in enumerate(order)}
        first = min(node.users, key=lambda user: rank[user])
        start, stop = rank[node], rank[first]
        crossed = order[start + 1 : stop]
        if not crossed:
            continue
        guarded_roots = analysis.reads[node] | analysis.values[node]
        writes = tuple(item for item in crossed if analysis.writes[item])
        blocked = tuple(
            item
            for item in crossed
            if item in analysis.fences
            and not (
                _only_tensor_write(item)
                and analysis.writes[item]
                and not analysis.writes[item] & guarded_roots
            )
        )
        if blocked:
            decisions.append(
                MovementDecision(
                    index,
                    indices[first],
                    False,
                    "crossed effect or overlapping writer",
                    obstructions=tuple(indices[item] for item in blocked),
                )
            )
            continue
        proposed = replace(
            current,
            order=order[:start] + crossed + (node,) + order[stop:],
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
                index,
                indices[first],
                not increases,
                "protected backing would increase" if increases else "moved",
                obstructions=increases,
                crossed_writes=tuple(indices[item] for item in writes),
                barrier_bytes=inventory,
            )
        )
        if not increases:
            first.prepend(node)
            current = proposed
    graph.graph.lint()
    graph.recompile()
    return LifetimeSchedule(
        tuple(decisions),
        tuple(indices[node] for node in current.order),
        tuple(
            (indices[barrier], initial_bytes[barrier], current.live_bytes(barrier))
            for barrier in ordered_barriers
        ),
        tuple(indices[node] for node in original if node in protected),
    )
