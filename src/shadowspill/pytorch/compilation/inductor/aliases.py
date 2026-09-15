"""Outputs that alias an input, made explicit as views of that input."""

from __future__ import annotations

import torch
from torch.fx import GraphModule, Node
from torch.utils._pytree import tree_flatten, tree_unflatten

from shadowspill.errors import CaptureError
from shadowspill.pytorch.capture.storage import (
    OutputView,
    StorageRoot,
    StorageRootKind,
    TaskStorageContract,
)


def _canonicalize_input_alias_outputs(
    graph_module: GraphModule,
    contract: TaskStorageContract,
    example_inputs: tuple[object, ...],
) -> None:
    """Make every declared input-alias return explicit in the FX output contract.

    In-place operators may return their mutated argument, but direct Inductor
    lowering can otherwise materialize that return as a second output buffer.
    Publishing the input (or its exact view) expresses the already-declared
    alias without adding a copy or changing the mutating operation itself.
    """

    output_node = next(node for node in graph_module.graph.nodes if node.op == "output")
    leaves, spec = tree_flatten(output_node.args[0])
    placeholders = tuple(
        node for node in graph_module.graph.nodes if node.op == "placeholder"
    )
    root_by_id = {root.root_id: root for root in contract.roots}
    view_by_leaf = {view.leaf_index: view for view in contract.output_views}
    changed = False
    for leaf_index, leaf in enumerate(leaves):
        view = view_by_leaf.get(leaf_index)
        if view is None:
            continue
        root = root_by_id[view.root_id]
        if root.kind is not StorageRootKind.INPUT:
            continue
        replacement = _input_alias_output(
            graph_module,
            output_node,
            placeholders,
            example_inputs,
            root,
            view,
        )
        if replacement is not leaf:
            leaves[leaf_index] = replacement
            changed = True
    if changed:
        output_node.args = (tree_unflatten(leaves, spec),)
        graph_module.graph.lint()
        graph_module.recompile()


def _input_alias_output(
    graph_module: GraphModule,
    output_node: Node,
    placeholders: tuple[Node, ...],
    example_inputs: tuple[object, ...],
    root: StorageRoot,
    view: OutputView,
) -> Node:
    source_position = root.source_input
    if source_position is None:
        raise AssertionError("input storage root omitted its source position")
    source = example_inputs[source_position]
    if not isinstance(source, torch.Tensor):
        raise CaptureError("compiled input alias refers to a non-tensor argument")
    _validate_input_alias_view(view, source, source_position)
    itemsize = source.element_size()
    placeholder = placeholders[source_position]
    source_offset_bytes = int(source.storage_offset()) * itemsize
    if (
        view.shape == tuple(int(value) for value in source.shape)
        and view.stride == tuple(int(value) for value in source.stride())
        and view.offset_bytes == source_offset_bytes
    ):
        return placeholder
    with graph_module.graph.inserting_before(output_node):
        return graph_module.graph.call_function(
            torch.ops.aten.as_strided.default,
            args=(
                placeholder,
                view.shape,
                view.stride,
                view.offset_bytes // itemsize,
            ),
        )


def _validate_input_alias_view(
    view: OutputView,
    source: torch.Tensor,
    source_position: int,
) -> None:
    if view.dtype != str(source.dtype):
        raise CaptureError(
            "direct compilation cannot express a dtype-changing input view: "
            f"leaf={view.leaf_index}, input={source_position}, "
            f"source={source.dtype}, output={view.dtype}"
        )
    itemsize = source.element_size()
    if view.offset_bytes % itemsize:
        raise CaptureError(
            "input-alias output offset is not element aligned: "
            f"leaf={view.leaf_index}, bytes={view.offset_bytes}, itemsize={itemsize}"
        )
