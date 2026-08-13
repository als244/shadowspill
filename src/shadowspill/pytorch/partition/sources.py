"""Resolve tensor provenance through a split FX root graph."""

from __future__ import annotations

import operator

from torch.fx import Node

from ..contracts import CaptureError
from .artifacts import StageValueSource


def stage_value_source(
    node: object,
    *,
    placeholder_index: dict[Node, int],
    stage_index_by_node: dict[Node, int],
) -> StageValueSource:
    """Resolve one split-root tensor node to a root input or stage output."""

    if not isinstance(node, Node):
        raise CaptureError("tensor stage input has no split-root FX provenance")
    root_index = placeholder_index.get(node)
    if root_index is not None:
        return StageValueSource(root_input_index=root_index)
    stage_index = stage_index_by_node.get(node)
    if stage_index is not None:
        return StageValueSource(
            producer_stage_index=stage_index,
            producer_output_index=0,
        )
    if node.op == "call_function" and node.target is operator.getitem:
        producer, output_index = node.args[:2]
        if isinstance(producer, Node) and isinstance(output_index, int):
            stage_index = stage_index_by_node.get(producer)
            if stage_index is not None:
                return StageValueSource(
                    producer_stage_index=stage_index,
                    producer_output_index=output_index,
                )
    raise CaptureError(
        "tensor stage input has unsupported split-root provenance: "
        f"node={node.name}, op={node.op}, target={node.target}"
    )


__all__ = ["stage_value_source"]
