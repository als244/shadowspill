"""Align repeated shared adjoints with an established chunk domain."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch.fx import Graph, Node

from .geometry import ChunkAxis, reshape_axis


def align_broadcasts(
    graph: Graph, layouts: Mapping[Node, Any], rows: int
) -> dict[Node, ChunkAxis]:
    """Shrink only broadcasts proved constant along a consumer's chunk axis.

    The VJP of a row sum expands a shared cotangent. Its chunk domain is supplied
    by the consuming pointwise operation, rather than guessed from an extent
    that happens to equal the sequence length. Placeholders are never sliced
    by this inference: the caller owns their contracts.
    """
    result: dict[Node, ChunkAxis] = {}

    def repeated(node: Node, axis: ChunkAxis) -> dict[Node, ChunkAxis]:
        if node.op != "call_function" or layouts[node] is not None:
            return {}
        value = node.meta.get("val")
        if (
            not isinstance(value, torch.Tensor)
            or value.shape[axis.dim] != axis.before * rows * axis.after
        ):
            return {}
        source = node.args[0] if node.args else None
        if not isinstance(source, Node) or layouts[source] is not None:
            return {}
        original = source.meta.get("val")
        if not isinstance(original, torch.Tensor):
            return {}
        name = str(node.target)
        if name == "aten.expand.default":
            dim = axis.dim - value.ndim + original.ndim
            if dim < 0 or original.shape[dim] == 1:
                return {node: axis}
        if name in {
            "aten.clone.default",
            "aten.detach.default",
            "aten.alias.default",
            "aten._to_copy.default",
            "prims.convert_element_type.default",
        }:
            inner = repeated(source, axis)
            if inner:
                return {**inner, node: axis}
        if name in {
            "aten.view.default",
            "aten.reshape.default",
            "aten._unsafe_view.default",
        }:
            from shadowspill.errors import CaptureError

            try:
                old_axis = reshape_axis(
                    axis, tuple(value.shape), tuple(original.shape), rows
                )
            except CaptureError:
                return {}
            inner = repeated(source, old_axis)
            if inner:
                return {**inner, node: axis}
        return {}

    for node in graph.nodes:
        axis = layouts.get(node)
        if not isinstance(axis, ChunkAxis) or node.op != "call_function":
            continue
        target = str(node.target).split(".")
        if len(target) < 2 or target[0] not in {"aten", "prims"}:
            continue
        # These consumers align every full-sized tensor operand elementwise.
        # Matrix, indexing and opaque operators have different operand domains.
        if target[1] not in {
            "add",
            "sub",
            "mul",
            "div",
            "where",
            "silu_backward",
            "gelu_backward",
            "threshold_backward",
            "tanh_backward",
            "sigmoid_backward",
            "_softmax_backward_data",
            "_log_softmax_backward_data",
        }:
            continue
        output = node.meta["val"]
        for operand in node.args:
            if not isinstance(operand, Node):
                continue
            value = operand.meta.get("val")
            if not isinstance(value, torch.Tensor):
                continue
            dim = axis.dim - output.ndim + value.ndim
            if dim < 0 or dim >= value.ndim:
                continue
            found = repeated(operand, ChunkAxis(dim, axis.before, axis.after))
            for selected, contract in found.items():
                if selected in result and result[selected] != contract:
                    from shadowspill.errors import CaptureError

                    raise CaptureError(
                        "a shared broadcast has incompatible row consumers"
                    )
                result[selected] = contract
    return result
