"""Carry gradient destinations through execution cuts before slicing tasks."""

from __future__ import annotations

import torch
from torch._guards import detect_fake_mode
from torch._prims_common import get_computation_dtype
from torch.fx import GraphModule, Node
from torch.utils._pytree import tree_flatten

from shadowspill.pytorch.capture.artifacts import GraphArtifact, TaskInputProvenance
from shadowspill.pytorch.capture.gradients.accumulation import _add_in_multiply
from shadowspill.pytorch.capture.gradients.nodes import _CONVERTING, _MOVING_ONLY
from shadowspill.task.inputs import TaskInputRole


def _same_geometry(left: Node, right: Node) -> bool:
    a, b = left.meta.get("val"), right.meta.get("val")
    return (
        isinstance(a, torch.Tensor)
        and isinstance(b, torch.Tensor)
        and a.shape == b.shape
        and a.dtype == b.dtype
    )


def _sum(node: Node) -> bool:
    alpha = node.kwargs.get("alpha", 1)
    return (
        node.target is torch.ops.aten.add.Tensor
        and isinstance(alpha, (int, float))
        and alpha == 1
        and len(node.args) == 2
        and all(
            isinstance(item, Node) and _same_geometry(node, item) for item in node.args
        )
    )


def _terms(node: Node) -> tuple[list[Node], list[Node]]:
    if not _sum(node) or len(node.users) != 1:
        return [], [node]
    joins, terms = [node], []
    for operand in node.args:
        assert isinstance(operand, Node)
        nested, leaves = _terms(operand)
        joins.extend(nested)
        terms.extend(leaves)
    return joins, terms


def _owned(node: Node) -> bool:
    """A single-use fresh value, possibly behind single-use views."""
    if node.op != "call_function" or len(node.users) != 1:
        return False
    if node.target in _MOVING_ONLY and node.target is not torch.ops.aten.clone.default:
        return isinstance(node.args[0], Node) and _owned(node.args[0])
    schema = getattr(node.target, "_schema", None)
    return (
        schema is not None
        and len(schema.returns) == 1
        and schema.returns[0].alias_info is None
        and isinstance(node.meta.get("val"), torch.Tensor)
    )


def route_gradient_destinations(
    graph: GraphModule,
    artifact: GraphArtifact,
    *,
    fuse_narrow_gradient_accumulation: bool,
    gradient_leaves: tuple[int, ...] | None = None,
) -> None:
    """Make eligible VJP joins write the same buffer across execution regions.

    Only single-use gradient expressions are consumed. Shared forward values,
    externally aliased priors and expressions read elsewhere retain their graph.
    Narrow sum trees retain their association unless fusion is permitted.
    """
    if artifact.kind != "backward":
        return
    mode = detect_fake_mode(artifact.example_arguments)
    provenance = TaskInputProvenance(TaskInputRole.GRADIENT)
    positions = {node: i for i, node in enumerate(graph.graph.nodes)}
    outputs, _ = tree_flatten(
        next(n for n in graph.graph.nodes if n.op == "output").args[0]
    )
    queue = [
        outputs[i]
        for i in (range(len(outputs)) if gradient_leaves is None else gradient_leaves)
    ]
    expressions: set[Node] = set()
    # Follow only gradient assembly, stopping at each contribution producer.
    # Recomputed forward values and internal activation adjoints are not
    # destinations for a parameter-gradient accumulation rewrite.
    assembly = (
        _MOVING_ONLY
        | _CONVERTING
        | {
            torch.ops.aten.add.Tensor,
            torch.ops.aten.add_.Tensor,
            torch.ops.aten.cat.default,
        }
    )
    while queue:
        value = queue.pop()
        if (
            not isinstance(value, Node)
            or value in expressions
            or value.target not in assembly
        ):
            continue
        expressions.add(value)
        queue.extend(value.all_input_nodes)

    def accumulate(prior: Node, produced: Node) -> list[Node]:
        multiply = produced
        while multiply.target in _MOVING_ONLY:
            multiply = multiply.args[0]  # type: ignore[assignment]
        # A destination must exist when the producer runs. Otherwise preserve
        # the producer and add after it; never introduce a use before definition.
        removed = None
        if positions.get(prior, -1) < positions.get(multiply, -1):
            removed = _add_in_multiply(
                graph.graph,
                produced,
                prior,
                [produced],
                mode,
                fuse_narrow_gradient_accumulation,
            )
        if removed is None:
            with graph.graph.inserting_after(produced):
                update = graph.graph.call_function(
                    torch.ops.aten.add_.Tensor, (prior, produced)
                )
            update.meta = dict(prior.meta)
            return []
        return removed

    placeholders = [n for n in graph.graph.nodes if n.op == "placeholder"]
    aliases = artifact.tensor_argument_alias_groups
    unique_inputs = {
        node
        for node, group in zip(placeholders, aliases, strict=True)
        if aliases.count(group) == 1
    }
    for node in tuple(graph.graph.nodes):
        if (
            node not in expressions
            or node.target is not torch.ops.aten.add_.Tensor
            or len(node.args) != 2
        ):
            continue
        prior, produced = node.args
        if (
            not isinstance(prior, Node)
            or not isinstance(produced, Node)
            or prior not in unique_inputs
            or len(prior.users) != 1
            or prior.meta.get("execution_provenance", provenance).role
            is not TaskInputRole.GRADIENT
            or node.kwargs.get("alpha", 1) != 1
            or not _same_geometry(prior, produced)
        ):
            continue
        value = prior.meta["val"]
        if (
            not fuse_narrow_gradient_accumulation
            and value.dtype != get_computation_dtype(value.dtype)
        ):
            continue
        joins, terms = _terms(produced)
        if not joins:
            continue
        removed = []
        for term in terms:
            removed.extend(accumulate(prior, term))
        node.replace_all_uses_with(prior)
        graph.graph.erase_node(node)
        for old in (*joins, *removed):
            graph.graph.erase_node(old)

    for node in tuple(graph.graph.nodes):
        if node not in expressions or not _sum(node):
            continue
        operands = sorted(node.args, key=lambda item: positions[item])
        prior, produced = operands
        if not _owned(prior):
            continue
        # The accumulator's geometry must also be writable by the producer.
        # Its public output is normalized by the composed task's boundary.
        prior.meta["execution_provenance"] = provenance
        removed = accumulate(prior, produced)
        node.replace_all_uses_with(prior)
        graph.graph.erase_node(node)
        for old in removed:
            graph.graph.erase_node(old)
    graph.graph.lint()
    graph.recompile()
