"""Source-operation provenance independent of a repeated stage's occurrence."""

from collections.abc import Mapping
from dataclasses import dataclass

from torch.fx import GraphModule, Node

from shadowspill.errors import CaptureError

from .torch_deprecations import copy_graph_module

_SOURCE_KEY = "shadowspill_source_operation"


@dataclass(frozen=True, slots=True)
class SourceOperation:
    index: int
    op: str
    target: str | None


def source_operation(node: Node) -> SourceOperation | None:
    custom = node.meta.get("custom", {})
    if not isinstance(custom, Mapping):
        raise CaptureError("FX custom metadata must be a mapping")
    value = custom.get(_SOURCE_KEY)
    if value is None:
        return None
    if (
        not isinstance(value, tuple)
        or len(value) != 3
        or isinstance(value[0], bool)
        or not isinstance(value[0], int)
        or value[0] < 0
        or not isinstance(value[1], str)
        or (value[2] is not None and not isinstance(value[2], str))
    ):
        raise CaptureError("FX source-operation metadata is invalid")
    return SourceOperation(*value)


def set_source_operation(node: Node, source: SourceOperation) -> None:
    node.meta["custom"] = {
        **node.meta.get("custom", {}),
        _SOURCE_KEY: (source.index, source.op, source.target),
    }


def _source_nodes(graph: GraphModule) -> tuple[Node, ...]:
    # Static argument specialization can remove placeholders without changing
    # the structural contract. They must not shift an operation's identity.
    return tuple(
        node for node in graph.graph.nodes if node.op not in {"placeholder", "output"}
    )


def traceable_source_graph(graph: GraphModule) -> GraphModule:
    """Tag a copy so tracing can preserve structural source indices."""
    copied = copy_graph_module(graph)
    for node in copied.graph.nodes:
        custom = dict(node.meta.get("custom", {}))
        custom.pop(_SOURCE_KEY, None)
        node.meta["custom"] = custom
    for index, node in enumerate(_source_nodes(copied)):
        # Source indices belong to this capture, not to a graph it was derived
        # from. The structurally keyed source graph fixes their interpretation.
        set_source_operation(node, SourceOperation(index, node.op, str(node.target)))
    return copied


def bind_source_metadata(graph: GraphModule, source: GraphModule) -> GraphModule:
    """Resolve source indices against one occurrence without mutating a cache."""
    nodes = _source_nodes(source)
    bound = copy_graph_module(graph)
    for node in bound.graph.nodes:
        reference = source_operation(node)
        node.meta.pop("nn_module_stack", None)
        node.meta.pop("fwd_nn_module_stack", None)
        if reference is None:
            continue
        if reference.index >= len(nodes):
            raise CaptureError("source-operation index is outside the source graph")
        original = nodes[reference.index]
        if (reference.op, reference.target) != (original.op, str(original.target)):
            raise CaptureError("source-operation contract differs from its occurrence")
        stack = original.meta.get("nn_module_stack")
        if stack is not None:
            node.meta["nn_module_stack"] = dict(stack)
    return bound
