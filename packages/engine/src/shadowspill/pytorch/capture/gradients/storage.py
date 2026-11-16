"""Give gradient outputs independent dense storage and equal-shard padding."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import nullcontext

import torch
from torch._guards import detect_fake_mode

from shadowspill.pytorch.capture.artifacts import GraphArtifact
from shadowspill.pytorch.operators.materialize import MATERIALIZE_GRADIENT

from ..torch_deprecations import copy_graph_module
from .nodes import (
    _record_value,
)


def _dense_gradient_layout(value: torch.Tensor) -> bool:
    """Static geometry check, including transposed dense tensors."""
    if value.numel() == 0:
        return True
    expected = 1
    for stride, extent in sorted(zip(value.stride(), value.shape, strict=True)):
        if extent == 1:
            continue
        if stride != expected:
            return False
        expected *= extent
    return True


def materialize_gradient_outputs(
    backward: GraphArtifact,
    leaf_indices: Sequence[int],
    *,
    memory_formats: Mapping[int, torch.memory_format] | None = None,
) -> GraphArtifact:
    """Give boundary gradients independent dense writable storage.

    Autograd may return a scalar expanded across a weight. The optimizer and
    later accumulation consume a full gradient tensor. Materialize that layout
    inside the backward task, before its allocation profile is measured.
    A fused backward may also return slices of one larger gradient allocation.
    Each parameter gradient and activation cotangent has its own canonical
    lifetime, so shared outputs are materialized inside the measured task too.
    Parameter gradients may retain transposed-dense layouts. Activation
    cotangents must use the canonical memory format expected by the preceding
    task; callers supply those formats explicitly.
    """
    memory_formats = memory_formats or {}
    output = next(
        node for node in backward.graph_module.graph.nodes if node.op == "output"
    )
    leaves = list(output.args[0])
    views = {view.leaf_index: view for view in backward.storage_contract.output_views}
    roots = {root.root_id: root for root in backward.storage_contract.roots}
    root_uses: dict[int, int] = {}
    for view in views.values():
        root_uses[view.root_id] = root_uses.get(view.root_id, 0) + 1
    # An input-root contract describes only the span reached by its outputs.
    # A single prefix slice can therefore have offset zero and a small minimum
    # span while borrowing a larger incoming cotangent. Its canonical gradient
    # cannot receive that larger storage lease. Include every declared view of
    # the input alias group, using geometry rather than sampled output storage.
    input_spans: dict[int, int] = {}
    for geometry, alias in zip(
        backward.tensor_inputs, backward.tensor_argument_alias_groups, strict=True
    ):
        elements = (
            0
            if 0 in geometry.shape
            else 1
            + sum(
                (extent - 1) * stride
                for extent, stride in zip(geometry.shape, geometry.stride, strict=True)
            )
        )
        span = (geometry.storage_offset + elements) * geometry.dtype.itemsize
        input_spans[alias] = max(input_spans.get(alias, 0), span)
    root_spans = {
        root.root_id: max(
            root.minimum_span_bytes,
            0
            if root.source_input is None
            else input_spans[backward.tensor_argument_alias_groups[root.source_input]],
        )
        for root in roots.values()
    }
    independent_leaves = {
        index
        for index, view in views.items()
        if root_uses[view.root_id] > 1
        or view.offset_bytes != 0
        or root_spans[view.root_id] > view.span_bytes
    }
    needed = [
        index
        for index in leaf_indices
        if not _dense_gradient_layout(leaves[index].meta["val"])
        or (
            index in memory_formats
            and not leaves[index]
            .meta["val"]
            .is_contiguous(memory_format=memory_formats[index])
        )
        or index in independent_leaves
    ]
    if not needed:
        return backward
    graph_module = copy_graph_module(backward.graph_module)
    graph = graph_module.graph
    output = next(node for node in graph.nodes if node.op == "output")
    leaves = list(output.args[0])
    mode = detect_fake_mode(backward.example_arguments)
    for index in needed:
        produced = leaves[index]
        memory_format = memory_formats.get(index, torch.contiguous_format)
        # Preserve an explicit allocation for slices of a shared gradient bank.
        # An ordinary broadcast-layout clone can still fuse with its producer.
        independent = index in independent_leaves
        with graph.inserting_before(output):
            node = graph.call_function(
                MATERIALIZE_GRADIENT if independent else torch.ops.aten.clone.default,
                args=(produced,),
                kwargs={"memory_format": str(memory_format).removeprefix("torch.")}
                if independent
                else {"memory_format": memory_format},
            )
        node.meta = dict(produced.meta)
        with mode if mode is not None else nullcontext():
            _record_value(node, produced.meta["val"].clone(memory_format=memory_format))
        leaves[index] = node
    output.args = (tuple(leaves),)
    graph.lint()
    graph_module.recompile()
    return GraphArtifact.capture(
        kind="backward",
        graph_module=graph_module,
        example_inputs=backward.example_arguments,
        input_provenance=backward.input_provenance,
    )
