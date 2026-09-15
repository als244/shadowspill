"""The allocations a compiled task makes, and the views onto them."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch._inductor.graph import GraphLowering
from torch.fx import GraphModule

from shadowspill.errors import CaptureError
from shadowspill.pytorch.capture.storage import (
    MutationBinding,
    OutputView,
    StorageRoot,
    StorageRootKind,
    TaskStorageContract,
)

from .manifest import ExecutableRootAllocation, _LoweredOutput
from .values import _static_int


def _build_executable_roots(
    graph: GraphLowering,
    records: tuple[_LoweredOutput, ...],
    input_position_by_name: Mapping[str, int],
) -> tuple[tuple[StorageRoot, ...], tuple[ExecutableRootAllocation, ...]]:
    root_names = tuple(dict.fromkeys(record.root_name for record in records))
    root_id_by_name = {name: index for index, name in enumerate(root_names)}
    roots: list[StorageRoot] = []
    allocations: list[ExecutableRootAllocation] = []
    for name in root_names:
        members = tuple(record for record in records if record.root_name == name)
        root, allocation = _build_executable_root(
            graph,
            name,
            root_id_by_name[name],
            members,
            input_position_by_name.get(name),
        )
        roots.append(root)
        allocations.append(allocation)
    return tuple(roots), tuple(allocations)


def _build_executable_root(
    graph: GraphLowering,
    name: str,
    root_id: int,
    members: tuple[_LoweredOutput, ...],
    source_input: int | None,
) -> tuple[StorageRoot, ExecutableRootAllocation]:
    minimum_span = max(record.offset_bytes + record.span_bytes for record in members)
    if source_input is not None:
        return (
            StorageRoot(
                root_id,
                StorageRootKind.INPUT,
                source_input,
                None,
                None,
                None,
                minimum_span,
            ),
            ExecutableRootAllocation(root_id, 0),
        )
    allocation_bytes = _graph_buffer_extent(graph, name)
    if allocation_bytes < minimum_span:
        raise CaptureError(
            "Inductor output allocation is smaller than its returned views: "
            f"root={name}, allocation={allocation_bytes}, "
            f"minimum_span={minimum_span}"
        )
    provenance = members[0].provenance
    return (
        StorageRoot(
            root_id,
            StorageRootKind.FRESH,
            None,
            provenance.producer_node or f"inductor_{name}",
            provenance.producer_target or "inductor.output_buffer",
            provenance.producer_result or 0,
            minimum_span,
        ),
        ExecutableRootAllocation(root_id, allocation_bytes),
    )


def _graph_buffer_extent(graph: GraphLowering, name: str) -> int:
    try:
        buffer: Any = graph.get_buffer(name)
        elements = _static_int(
            graph,
            graph.get_allocation_storage_size(buffer),
            "output allocation storage length",
        )
        item_size = torch.empty(
            (), device="meta", dtype=buffer.get_dtype()
        ).element_size()
    except (AttributeError, NotImplementedError, RuntimeError, TypeError) as error:
        raise CaptureError(
            f"Inductor GraphLowering output has no allocation extent: root={name}"
        ) from error
    return elements * item_size


def _lowered_output_views(
    records: tuple[_LoweredOutput, ...],
    roots: tuple[StorageRoot, ...],
) -> tuple[OutputView, ...]:
    root_id_by_name = {
        name: root.root_id
        for name, root in zip(
            dict.fromkeys(record.root_name for record in records),
            roots,
            strict=True,
        )
    }
    return tuple(
        OutputView(
            leaf_index=record.semantic_view.leaf_index,
            root_id=root_id_by_name[record.root_name],
            offset_bytes=record.offset_bytes,
            span_bytes=record.span_bytes,
            shape=record.shape,
            stride=record.stride,
            dtype=record.dtype,
            layout=record.semantic_view.layout,
        )
        for record in records
    )


def _visible_output_indices(graph: GraphModule) -> tuple[int, ...]:
    output = next(
        (node for node in graph.graph.nodes if node.op == "output"),
        None,
    )
    if output is None:
        raise CaptureError("optimized Inductor graph has no output node")
    raw_visible = output.meta.get("user_visible_output_idxs")
    if not isinstance(raw_visible, tuple | list) or any(
        not isinstance(index, int) or index < 0 for index in raw_visible
    ):
        raise CaptureError("Inductor did not publish callable-visible output indices")
    return tuple(raw_visible)


def _project_mutations(
    semantic_contract: TaskStorageContract,
    roots: tuple[StorageRoot, ...],
    output_views: tuple[OutputView, ...],
) -> tuple[MutationBinding, ...]:
    view_by_leaf = {view.leaf_index: view for view in output_views}
    root_by_id = {root.root_id: root for root in roots}
    for mutation in semantic_contract.mutations:
        leaf = mutation.replacement_output_leaf
        if leaf is None:
            continue
        view = view_by_leaf.get(leaf)
        if view is None:
            raise CaptureError(
                "Inductor removed a functional mutation replacement: "
                f"leaf={leaf}, target={mutation.argument_name}"
            )
        root = root_by_id[view.root_id]
        if (
            root.kind is StorageRootKind.INPUT
            and root.source_input != mutation.input_position
        ):
            raise CaptureError(
                "functional mutation replacement aliases another input: "
                f"leaf={leaf}, expected={mutation.input_position}, "
                f"actual={root.source_input}"
            )
    return semantic_contract.mutations
