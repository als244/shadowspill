"""Which lowered outputs are visible, and the geometry each one has."""

from __future__ import annotations

from typing import Any

import torch
from torch._inductor.graph import GraphLowering
from torch.fx import GraphModule

from shadowspill.errors import CaptureError
from shadowspill.pytorch.capture.storage import (
    OutputView,
    StorageRoot,
    TaskStorageContract,
)

from .manifest import _LoweredOutput
from .roots import _visible_output_indices
from .values import _span_bytes, _static_int


def _select_graph_outputs(
    graph: GraphLowering,
    optimized_graph: GraphModule,
    inner_contract: TaskStorageContract,
    semantic_contract: TaskStorageContract,
) -> tuple[tuple[int, ...], tuple[Any, ...]]:
    visible = _visible_output_indices(optimized_graph)
    if graph.graph_outputs is None:
        raise CaptureError("Inductor GraphLowering omitted task outputs")
    if any(index >= len(graph.graph_outputs) for index in visible):
        raise CaptureError(
            "Inductor callable-visible output index exceeds GraphLowering contract"
        )
    inner_leaves = {view.leaf_index for view in inner_contract.output_views}
    outputs = tuple(
        graph.graph_outputs[index] for index in visible if index in inner_leaves
    )
    if len(outputs) != len(semantic_contract.output_views):
        raise CaptureError(
            "GraphLowering callable-visible tensor output count changed: "
            f"semantic={len(semantic_contract.output_views)}, "
            f"executable={len(outputs)}"
        )
    return visible, outputs


def _graph_input_positions(
    graph: GraphLowering,
    optimized_graph: GraphModule,
) -> dict[str, int]:
    placeholders = tuple(
        node for node in optimized_graph.graph.nodes if node.op == "placeholder"
    )
    return {
        node.name: index
        for index, node in enumerate(placeholders)
        if node.name in graph.graph_inputs
    }


def _lower_graph_outputs(
    graph: GraphLowering,
    outputs: tuple[Any, ...],
    visible: tuple[int, ...],
    inner_contract: TaskStorageContract,
    semantic_contract: TaskStorageContract,
) -> tuple[_LoweredOutput, ...]:
    inner_view_by_leaf = {view.leaf_index: view for view in inner_contract.output_views}
    optimized_views = tuple(
        inner_view_by_leaf[index] for index in visible if index in inner_view_by_leaf
    )
    semantic_views = tuple(
        sorted(semantic_contract.output_views, key=lambda view: view.leaf_index)
    )
    root_by_id = {root.root_id: root for root in inner_contract.roots}
    return tuple(
        _lower_graph_output(
            graph,
            semantic_view,
            optimized_view,
            output,
            root_by_id[optimized_view.root_id],
        )
        for semantic_view, optimized_view, output in zip(
            semantic_views, optimized_views, outputs, strict=True
        )
    )


def _lower_graph_output(
    graph: GraphLowering,
    semantic_view: OutputView,
    optimized_view: OutputView,
    output: Any,
    provenance: StorageRoot,
) -> _LoweredOutput:
    if not output.has_tensor_output():
        raise CaptureError(
            "GraphLowering replaced a tensor output with a non-tensor value: "
            f"leaf={semantic_view.leaf_index}"
        )
    root_name, shape, stride, dtype, offset_bytes, span_bytes = _graph_output_geometry(
        graph, semantic_view, output
    )
    _validate_graph_output_geometry(semantic_view, shape, stride, dtype)
    return _LoweredOutput(
        semantic_view=semantic_view,
        optimized_view=optimized_view,
        provenance=provenance,
        root_name=root_name,
        offset_bytes=offset_bytes,
        span_bytes=span_bytes,
        shape=shape,
        stride=stride,
        dtype=str(dtype),
    )


def _graph_output_geometry(
    graph: GraphLowering,
    semantic_view: OutputView,
    output: Any,
) -> tuple[str, tuple[int, ...], tuple[int, ...], torch.dtype, int, int]:
    try:
        root_name = str(output.get_name())
        layout = output.get_layout()
        dtype = output.get_dtype()
        shape = tuple(
            _static_int(graph, value, "output shape") for value in output.get_size()
        )
        stride = tuple(
            _static_int(graph, value, "output stride") for value in output.get_stride()
        )
        offset_elements = _static_int(graph, layout.offset, "output offset")
    except (AttributeError, NotImplementedError, TypeError) as error:
        raise CaptureError(
            "Inductor GraphLowering output has no concrete strided layout: "
            f"leaf={semantic_view.leaf_index}, type={type(output).__name__}"
        ) from error
    if offset_elements < 0 or any(value < 0 for value in (*shape, *stride)):
        raise CaptureError(
            "Inductor GraphLowering produced a negative output geometry: "
            f"leaf={semantic_view.leaf_index}"
        )
    item_size = torch.empty((), device="meta", dtype=dtype).element_size()
    return (
        root_name,
        shape,
        stride,
        dtype,
        offset_elements * item_size,
        _span_bytes(shape, stride, item_size),
    )


def _validate_graph_output_geometry(
    expected: OutputView,
    shape: tuple[int, ...],
    stride: tuple[int, ...],
    dtype: torch.dtype,
) -> None:
    significant_strides_match = all(
        extent <= 1 or actual == expected_stride
        for extent, actual, expected_stride in zip(
            shape, stride, expected.stride, strict=True
        )
    )
    expected_geometry = (
        expected.shape,
        expected.stride,
        expected.dtype,
        expected.layout,
    )
    actual_geometry = (shape, stride, str(dtype), str(torch.strided))
    if (
        shape != expected.shape
        or str(dtype) != expected.dtype
        or str(torch.strided) != expected.layout
        or not significant_strides_match
    ):
        raise CaptureError(
            "GraphLowering changed task output geometry: "
            f"leaf={expected.leaf_index}, expected={expected_geometry}, "
            f"actual={actual_geometry}"
        )
