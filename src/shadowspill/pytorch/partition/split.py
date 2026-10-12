"""Mechanical FX graph splitting and stage-call observation.

The public partition orchestrator delegates here after a policy has selected
one stage label for every executable node.  This module owns the mechanical
work of splitting and observing the stage contract. Semantic projection lives in
``provenance.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.fx import GraphModule, Interpreter, Node
from torch.fx.passes.split_module import split_module
from torch.utils._python_dispatch import is_traceable_wrapper_subclass
from torch.utils._pytree import tree_flatten, tree_map

from shadowspill.errors import CaptureError
from shadowspill.pytorch.capture.aot import ExportCapture
from shadowspill.pytorch.representations import tensor_components

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
            raise CaptureError("automatic stage calls require a positional signature")
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
    except CaptureError:
        raise
    except Exception as exc:
        raise CaptureError(f"stage partition failed: {exc}") from exc
    return observe_split_root(root, capture.flat_inputs)


def observe_split_root(
    root: GraphModule, flat_inputs: tuple[object, ...]
) -> SplitExportGraph:
    """Run a split root on one position's inputs and record what each stage
    was given and gave back.

    The root is a function of the exported program and the stage
    assignments alone, so every position of one exported program can be
    observed through the root another position was split into; only what
    the stages see is the position's own.
    """

    try:
        recorder = _StageRecorder(root)
        recorder.run(*flat_inputs)
    except CaptureError:
        raise
    except Exception as exc:
        raise CaptureError(f"stage partition failed: {exc}") from exc

    call_nodes, placeholder_index, stage_index_by_node = _root_topology(root)
    if len(call_nodes) != len(recorder.calls):
        raise CaptureError("split root task topology differs from recorded calls")
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


def _root_topology(
    root: GraphModule,
) -> tuple[tuple[Node, ...], dict[Node, int], dict[Node, int]]:
    """A split root's stage calls in order, and the index of each placeholder
    and each stage call."""

    call_nodes = tuple(node for node in root.graph.nodes if node.op == "call_module")
    placeholders = tuple(node for node in root.graph.nodes if node.op == "placeholder")
    return (
        call_nodes,
        {node: index for index, node in enumerate(placeholders)},
        {node: index for index, node in enumerate(call_nodes)},
    )


def rebind_split_records(
    root: GraphModule,
    records: tuple[StageRecord, ...],
    template_inputs: tuple[object, ...],
    flat_inputs: tuple[object, ...],
) -> SplitExportGraph | None:
    """Another position's stage records, built from a template's without
    running the graph, or ``None`` where that cannot be done exactly.

    A position of the same exported program, whose inputs have the
    template's geometry, gives each stage the same geometry the template's
    gave it; only which tensors those are is its own. So every tensor the
    template's stages were given or gave back is mapped once: a root input
    to this position's input at the same index, and anything else to a
    tensor of the same geometry on storage of this position's own. Storage
    is mapped as a whole, so tensors that shared storage in the template --
    a view of an input, two outputs of one buffer -- share it here as well,
    which is what the lowering reads aliasing from. A tensor this cannot
    reproduce exactly, a wrapper of components or a non-strided layout,
    returns ``None`` and the caller runs the graph instead.
    """

    if len(template_inputs) != len(flat_inputs):
        return None
    tensors: dict[int, torch.Tensor] = {}
    storages: dict[int, torch.UntypedStorage] = {}
    for template, value in zip(template_inputs, flat_inputs, strict=True):
        if isinstance(template, torch.Tensor) != isinstance(value, torch.Tensor):
            return None
        if not isinstance(template, torch.Tensor):
            continue
        assert isinstance(value, torch.Tensor)
        tensors[id(template)] = value
        # A wrapper owns no storage; its components do, and a stage output
        # may view one, so each component's storage is mapped to the one in
        # the same place of this position's input.
        pairs = (
            zip(
                (component for _, component in tensor_components(template)),
                (component for _, component in tensor_components(value)),
                strict=True,
            )
            if is_traceable_wrapper_subclass(template)
            else ((template, value),)
        )
        for before, after in pairs:
            storages[_storage_key(before)] = after.untyped_storage()

    def mapped(leaf: object) -> object:
        if not isinstance(leaf, torch.Tensor):
            return leaf
        found = tensors.get(id(leaf))
        if found is not None:
            return found
        if is_traceable_wrapper_subclass(leaf) or leaf.layout is not torch.strided:
            raise _CannotRebind
        key = _storage_key(leaf)
        storage = storages.get(key)
        if storage is None:
            storage = torch.empty(
                leaf.untyped_storage().nbytes(), dtype=torch.uint8, device=leaf.device
            ).untyped_storage()
            storages[key] = storage
        result = torch.empty(0, dtype=leaf.dtype, device=leaf.device).set_(
            storage, leaf.storage_offset(), tuple(leaf.shape), tuple(leaf.stride())
        )
        if leaf.requires_grad:
            result.requires_grad_(True)
        tensors[id(leaf)] = result
        return result

    try:
        rebound = tuple(
            StageRecord(
                record.module_target,
                record.graph_module,
                tree_map(mapped, record.inputs),
                record.input_sources,
                tree_map(mapped, record.output),
            )
            for record in records
        )
    except _CannotRebind:
        return None
    _, placeholder_index, stage_index_by_node = _root_topology(root)
    return SplitExportGraph(root, rebound, placeholder_index, stage_index_by_node)


class _CannotRebind(Exception):
    """A template tensor whose counterpart cannot be built exactly."""


def _storage_key(tensor: torch.Tensor) -> int:
    return int(tensor.untyped_storage()._cdata)


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


__all__ = [
    "SplitExportGraph",
    "observe_split_root",
    "rebind_split_records",
    "split_export_graph",
]
