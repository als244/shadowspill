"""Whether a partitioned stage has work of its own to compile."""

from __future__ import annotations

from torch.fx import GraphModule, Node


def computes_nothing(graph_module: GraphModule) -> bool:
    """Whether every operation in this stage only renames what it was given.

    A stage becomes one compiled kernel graph. An operation whose every
    result is a view of an argument writes no storage, and one that returns
    nothing writes no value at all; a stage made only of those has no kernel
    to generate, and a compiler asked for one exposes no root graph.

    What counts as writing comes from each operator's own schema rather than
    a list kept here, so an operator this module has never heard of is
    classified by what it declares: a result carrying alias information came
    from an argument rather than from new storage.
    """

    return all(
        _writes_nothing(node)
        for node in graph_module.graph.nodes
        if node.op not in {"placeholder", "output", "get_attr"}
    )


def _writes_nothing(node: Node) -> bool:
    schema = getattr(node.target, "_schema", None)
    if schema is None:
        return False
    # Vacuously true for an operator that returns nothing, which is what it
    # should be: an assertion writes no value either.
    return all(result.alias_info is not None for result in schema.returns)


__all__ = ["computes_nothing"]
