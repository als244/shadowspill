"""Mechanical FX graph splitting and stage-call observation.

The public partition orchestrator delegates here after a policy has selected
one stage label for every executable node.  This module owns the mechanical
work of splitting and observing the stage ABI. Semantic projection lives in
``provenance.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.fx import GraphModule, Interpreter, Node
from torch.fx.passes.split_module import split_module
from torch.utils._pytree import tree_flatten

from ..aot import ExportCapture
from ..contracts import CaptureError
from .artifacts import StageRecord
from .sources import stage_value_source


@dataclass(frozen=True, slots=True)
class SplitExportGraph:
    """One split root plus the topology needed for semantic projection."""

    root: GraphModule
    stages: tuple[StageRecord, ...]
    placeholder_index: dict[Node, int]
    stage_index_by_node: dict[Node, int]


class _StageRecorder(Interpreter):
    """Observe exact stage inputs and outputs through the split root."""

    def __init__(self, module: GraphModule) -> None:
        super().__init__(module)
        self.calls: list[tuple[str, tuple[object, ...], object]] = []

    def call_module(
        self, target: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Any:
        if kwargs:
            raise CaptureError("automatic stage calls require a positional ABI")
        output = super().call_module(target, args, kwargs)
        self.calls.append((str(target), args, output))
        return output


def split_export_graph(
    capture: ExportCapture,
    assignments: dict[Node, int],
) -> SplitExportGraph:
    """Split and observe one exported graph under validated assignments."""

    graph_module = capture.exported_program.graph_module
    try:
        root = split_module(
            graph_module,
            graph_module,
            lambda node: assignments[node],
            keep_original_order=True,
        )
        recorder = _StageRecorder(root)
        recorder.run(*capture.flat_inputs)
    except CaptureError:
        raise
    except Exception as exc:
        raise CaptureError(f"stage partition failed: {exc}") from exc

    call_nodes = tuple(node for node in root.graph.nodes if node.op == "call_module")
    if len(call_nodes) != len(recorder.calls):
        raise CaptureError("split root task topology differs from recorded calls")
    placeholders = tuple(node for node in root.graph.nodes if node.op == "placeholder")
    placeholder_index = {node: index for index, node in enumerate(placeholders)}
    stage_index_by_node = {node: index for index, node in enumerate(call_nodes)}
    stages = tuple(
        _stage_record(
            root,
            call_node,
            recorded,
            placeholder_index=placeholder_index,
            stage_index_by_node=stage_index_by_node,
        )
        for recorded, call_node in zip(recorder.calls, call_nodes, strict=True)
    )
    if not stages:
        raise CaptureError("partitioning produced no executable stage")
    return SplitExportGraph(root, stages, placeholder_index, stage_index_by_node)


def _stage_record(
    root: GraphModule,
    call_node: Node,
    recorded: tuple[str, tuple[object, ...], object],
    *,
    placeholder_index: dict[Node, int],
    stage_index_by_node: dict[Node, int],
) -> StageRecord:
    target, inputs, output = recorded
    child = root.get_submodule(target)
    if not isinstance(child, GraphModule):
        raise CaptureError(f"partition {target!r} is not an FX GraphModule")
    argument_leaves, _ = tree_flatten(call_node.args)
    input_leaves, _ = tree_flatten(inputs)
    if len(argument_leaves) != len(input_leaves):
        raise CaptureError("stage input structure differs from split-root topology")
    sources = tuple(
        stage_value_source(
            leaf,
            placeholder_index=placeholder_index,
            stage_index_by_node=stage_index_by_node,
        )
        if isinstance(value, torch.Tensor)
        else None
        for leaf, value in zip(argument_leaves, input_leaves, strict=True)
    )
    return StageRecord(target, child, inputs, sources, output)


__all__ = ["SplitExportGraph", "split_export_graph"]
