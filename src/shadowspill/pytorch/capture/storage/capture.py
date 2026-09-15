"""The contract captured from a task that has run on fake tensors."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch.fx import GraphModule, Node
from torch.utils._pytree import tree_flatten

from shadowspill.errors import CaptureError
from shadowspill.pytorch.capture.live_storage import (
    live_storage_bytes,
)

from .records import (
    MutationBinding,
    OutputView,
    StorageRoot,
    StorageRootKind,
    TaskStorageContract,
)
from .roots import _InputRoot, _SemanticRoot
from .symbolic import (
    _canonical_input_positions,
    _capture_mutations,
    _raw_offset_bytes,
    _reject_tensor_getattrs,
    _storage_root,
    _symbolic_output_values,
    _view_span_bytes,
    make_storage_contract,
)


@dataclass(frozen=True, slots=True)
class _TensorOutput:
    leaf_index: int
    node: Node
    value: torch.Tensor
    root: _SemanticRoot


@dataclass(frozen=True, slots=True)
class _RootCatalog:
    roots: tuple[StorageRoot, ...]
    root_ids: Mapping[_SemanticRoot, int]
    base_offsets: Mapping[_SemanticRoot, int]


@dataclass(frozen=True, slots=True)
class ExplicitMutation:
    """Signature-level functional mutation supplied by Export normalization."""

    input_position: int
    output_leaf_index: int
    target: str

    def __post_init__(self) -> None:
        if self.input_position < 0 or self.output_leaf_index < 0:
            raise ValueError("explicit mutation positions must be non-negative")
        if not self.target:
            raise ValueError("explicit mutation target must be non-empty")


def capture_task_storage_contract(
    graph_module: GraphModule,
    example_inputs: tuple[object, ...],
    *,
    explicit_mutations: tuple[ExplicitMutation, ...] = (),
) -> TaskStorageContract:
    """Derive output ownership and mutations without compiled execution.

    Provenance and dispatcher schemas determine semantic roots. A fresh
    FakeTensor execution supplies only tensor geometry and cannot merge or
    split roots through its synthetic storage identities.
    """

    _reject_tensor_getattrs(graph_module)
    leaves, values = _task_output_values(graph_module, example_inputs)
    canonical_inputs, placeholder_positions = _input_positions(
        graph_module, example_inputs
    )
    root_cache: dict[Node, _SemanticRoot] = {}
    tensor_outputs = _tensor_outputs(leaves, values, placeholder_positions, root_cache)
    catalog = _build_root_catalog(tensor_outputs, example_inputs)
    output_views = _build_output_views(tensor_outputs, catalog)
    schema_mutations = _capture_mutations(
        graph_module,
        placeholder_positions,
        root_cache,
    )
    mutation_bindings = _explicit_mutation_bindings(
        explicit_mutations,
        example_inputs,
        canonical_inputs,
        leaves,
        output_views,
        catalog.roots,
    )
    return make_storage_contract(
        catalog.roots,
        output_views,
        (*schema_mutations, *mutation_bindings),
    )


def _task_output_values(
    graph_module: GraphModule,
    example_inputs: tuple[object, ...],
) -> tuple[tuple[object, ...], tuple[object, ...]]:
    output_node = next(
        (node for node in graph_module.graph.nodes if node.op == "output"), None
    )
    if output_node is None:
        raise CaptureError("task graph has no output node")
    leaves, _ = tree_flatten(output_node.args[0])
    values = _symbolic_output_values(graph_module, example_inputs)
    if len(values) != len(leaves):
        raise CaptureError("task output geometry arity differs from its FX graph")
    return tuple(leaves), values


def _input_positions(
    graph_module: GraphModule,
    example_inputs: tuple[object, ...],
) -> tuple[tuple[int, ...], dict[Node, int]]:
    placeholders = tuple(
        node for node in graph_module.graph.nodes if node.op == "placeholder"
    )
    if len(placeholders) != len(example_inputs):
        raise CaptureError("task placeholder count differs from example arguments")
    canonical = _canonical_input_positions(example_inputs)
    return canonical, {
        node: canonical[index] for index, node in enumerate(placeholders)
    }


def _tensor_outputs(
    leaves: tuple[object, ...],
    values: tuple[object, ...],
    placeholder_positions: Mapping[Node, int],
    root_cache: dict[Node, _SemanticRoot],
) -> tuple[_TensorOutput, ...]:
    outputs: list[_TensorOutput] = []
    for leaf_index, (leaf, value) in enumerate(zip(leaves, values, strict=True)):
        if not isinstance(value, torch.Tensor):
            continue
        if not isinstance(leaf, Node):
            raise CaptureError(
                f"tensor output leaf {leaf_index} is not represented by an FX node"
            )
        outputs.append(
            _TensorOutput(
                leaf_index,
                leaf,
                value,
                _storage_root(leaf, placeholder_positions, root_cache),
            )
        )
    return tuple(outputs)


def _build_root_catalog(
    outputs: tuple[_TensorOutput, ...],
    example_inputs: tuple[object, ...],
) -> _RootCatalog:
    grouped = _group_tensor_outputs(outputs)
    root_ids = {root: index for index, root in enumerate(grouped)}
    roots: list[StorageRoot] = []
    base_offsets: dict[_SemanticRoot, int] = {}
    for root, values in grouped.items():
        offsets = [_raw_offset_bytes(item.value) for item in values]
        origin = 0 if isinstance(root, _InputRoot) else min(offsets)
        minimum_span = max(
            offset - origin + _view_span_bytes(item.value)
            for offset, item in zip(offsets, values, strict=True)
        )
        base_offsets[root] = origin
        roots.append(
            _storage_root_spec(
                root,
                root_ids[root],
                minimum_span,
                example_inputs,
            )
        )
    return _RootCatalog(tuple(roots), root_ids, base_offsets)


def _group_tensor_outputs(
    outputs: tuple[_TensorOutput, ...],
) -> dict[_SemanticRoot, list[_TensorOutput]]:
    grouped: dict[_SemanticRoot, list[_TensorOutput]] = {}
    for output in outputs:
        grouped.setdefault(output.root, []).append(output)
    return grouped


def _storage_root_spec(
    root: _SemanticRoot,
    root_id: int,
    minimum_span: int,
    example_inputs: tuple[object, ...],
) -> StorageRoot:
    if isinstance(root, _InputRoot):
        source = example_inputs[root.position]
        if not isinstance(source, torch.Tensor):
            raise CaptureError("output aliases a non-tensor task input")
        if minimum_span > live_storage_bytes(source):
            raise CaptureError("output view exceeds its input storage")
        return StorageRoot(
            root_id,
            StorageRootKind.INPUT,
            root.position,
            None,
            None,
            None,
            minimum_span,
        )
    return StorageRoot(
        root_id,
        StorageRootKind.FRESH,
        None,
        root.node.name,
        str(root.node.target),
        root.result_index,
        minimum_span,
    )


def _build_output_views(
    outputs: tuple[_TensorOutput, ...],
    catalog: _RootCatalog,
) -> tuple[OutputView, ...]:
    return tuple(
        OutputView(
            leaf_index=output.leaf_index,
            root_id=catalog.root_ids[output.root],
            offset_bytes=(
                _raw_offset_bytes(output.value) - catalog.base_offsets[output.root]
            ),
            span_bytes=_view_span_bytes(output.value),
            shape=tuple(int(extent) for extent in output.value.shape),
            stride=tuple(int(stride) for stride in output.value.stride()),
            dtype=str(output.value.dtype),
            layout=str(output.value.layout),
        )
        for output in outputs
    )


def _explicit_mutation_bindings(
    mutations: tuple[ExplicitMutation, ...],
    example_inputs: tuple[object, ...],
    canonical_inputs: tuple[int, ...],
    leaves: tuple[object, ...],
    output_views: tuple[OutputView, ...],
    roots: tuple[StorageRoot, ...],
) -> tuple[MutationBinding, ...]:
    output_view_by_leaf = {view.leaf_index: view for view in output_views}
    root_by_id = {root.root_id: root for root in roots}
    return tuple(
        _explicit_mutation_binding(
            mutation,
            example_inputs,
            canonical_inputs,
            leaves,
            output_view_by_leaf,
            root_by_id,
        )
        for mutation in mutations
    )


def _explicit_mutation_binding(
    mutation: ExplicitMutation,
    example_inputs: tuple[object, ...],
    canonical_inputs: tuple[int, ...],
    leaves: tuple[object, ...],
    output_views: Mapping[int, OutputView],
    roots: Mapping[int, StorageRoot],
) -> MutationBinding:
    if mutation.input_position >= len(example_inputs):
        raise CaptureError(
            "functional mutation references an unavailable task input: "
            f"input={mutation.input_position}, target={mutation.target}"
        )
    source_position = canonical_inputs[mutation.input_position]
    if not isinstance(example_inputs[source_position], torch.Tensor):
        raise CaptureError(
            "functional mutation target is not a tensor task input: "
            f"input={mutation.input_position}, target={mutation.target}"
        )
    view = output_views.get(mutation.output_leaf_index)
    if view is None:
        raise CaptureError(
            "functional mutation replacement is not a tensor output: "
            f"leaf={mutation.output_leaf_index}, target={mutation.target}"
        )
    _validate_mutation_root(mutation, source_position, roots[view.root_id])
    leaf = leaves[mutation.output_leaf_index]
    if not isinstance(leaf, Node):
        raise CaptureError("functional mutation output has no FX provenance")
    return MutationBinding(
        source_position,
        mutation.output_leaf_index,
        leaf.name,
        str(leaf.target),
        mutation.target,
    )


def _validate_mutation_root(
    mutation: ExplicitMutation,
    source_position: int,
    root: StorageRoot,
) -> None:
    if root.kind is StorageRootKind.INPUT and root.source_input != source_position:
        raise CaptureError(
            "functional mutation replacement aliases a different task input: "
            f"leaf={mutation.output_leaf_index}, target={mutation.target}, "
            f"expected_input={source_position}, actual_input={root.source_input}"
        )
