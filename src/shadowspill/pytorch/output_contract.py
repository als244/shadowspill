"""Offline storage semantics for one normalized functional FX task."""

from __future__ import annotations

import hashlib
import json
import operator
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx import GraphModule, Node
from torch.utils._pytree import tree_flatten

from ._live_storage import live_storage_bytes, live_storage_identity
from ._schema_adapter import operator_alias_contract
from .contracts import CaptureError


class StorageRootKind(StrEnum):
    """Semantic origin of one output-reachable task storage."""

    INPUT = "input"
    FRESH = "fresh"


@dataclass(frozen=True, slots=True)
class StorageRoot:
    """One normalized semantic storage root.

    Input roots name the canonical ABI argument position for an input alias
    group. Fresh roots name one FX producer and result index. Physical
    allocation sizes deliberately do not appear in this record.
    """

    root_id: int
    kind: StorageRootKind
    source_input: int | None
    producer_node: str | None
    producer_target: str | None
    producer_result: int | None
    minimum_span_bytes: int

    def __post_init__(self) -> None:
        if self.root_id < 0 or self.minimum_span_bytes < 0:
            raise ValueError("storage-root fields must be non-negative")
        if not isinstance(self.kind, StorageRootKind):
            raise TypeError("storage-root kind has an invalid type")
        if self.kind is StorageRootKind.INPUT:
            if self.source_input is None or self.source_input < 0:
                raise ValueError("input root requires an ABI input position")
            if any(
                value is not None
                for value in (
                    self.producer_node,
                    self.producer_target,
                    self.producer_result,
                )
            ):
                raise ValueError("input root cannot name an FX producer")
        else:
            if self.source_input is not None:
                raise ValueError("fresh root cannot name an ABI input")
            if not self.producer_node or not self.producer_target:
                raise ValueError("fresh root requires FX producer provenance")
            if self.producer_result is None or self.producer_result < 0:
                raise ValueError("fresh root requires a producer result index")

    def identity(self) -> dict[str, object]:
        return {
            "root_id": self.root_id,
            "kind": self.kind.value,
            "source_input": self.source_input,
            "producer_node": self.producer_node,
            "producer_target": self.producer_target,
            "producer_result": self.producer_result,
            "minimum_span_bytes": self.minimum_span_bytes,
        }


@dataclass(frozen=True, slots=True)
class OutputView:
    """One flattened tensor output view of a semantic storage root."""

    leaf_index: int
    root_id: int
    offset_bytes: int
    span_bytes: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: str
    layout: str

    def __post_init__(self) -> None:
        if min(self.leaf_index, self.root_id, self.offset_bytes, self.span_bytes) < 0:
            raise ValueError("output-view fields must be non-negative")
        if len(self.shape) != len(self.stride):
            raise ValueError("output-view shape and stride ranks differ")
        if any(extent < 0 for extent in self.shape):
            raise ValueError("output-view shape has a negative extent")
        if any(stride < 0 for stride in self.stride):
            raise ValueError("output-view stride is negative")
        if not self.dtype or not self.layout:
            raise ValueError("output-view dtype and layout must be non-empty")

    def identity(self) -> dict[str, object]:
        return {
            "leaf_index": self.leaf_index,
            "root_id": self.root_id,
            "offset_bytes": self.offset_bytes,
            "span_bytes": self.span_bytes,
            "shape": list(self.shape),
            "stride": list(self.stride),
            "dtype": self.dtype,
            "layout": self.layout,
        }


@dataclass(frozen=True, slots=True)
class MutationBinding:
    """One task operation that updates an ABI input storage.

    ``replacement_output_leaf`` distinguishes Export's functional mutation
    form from a dispatcher-schema write.  The executable replacement normally
    has fresh storage and becomes the input object's next authoritative
    generation.  Inductor may prove a no-op update and return the target input
    itself; that preserves the mutation ABI without requiring a generation
    replacement.  A schema write has no replacement output because the
    compiled operation writes in place.
    """

    input_position: int
    replacement_output_leaf: int | None
    producer_node: str
    producer_target: str
    argument_name: str

    def __post_init__(self) -> None:
        if self.input_position < 0:
            raise ValueError("mutation input position must be non-negative")
        if (
            self.replacement_output_leaf is not None
            and self.replacement_output_leaf < 0
        ):
            raise ValueError("mutation output leaf must be non-negative")
        if not self.producer_node or not self.producer_target or not self.argument_name:
            raise ValueError("mutation provenance must be non-empty")

    def identity(self) -> dict[str, object]:
        return {
            "input_position": self.input_position,
            "replacement_output_leaf": self.replacement_output_leaf,
            "producer_node": self.producer_node,
            "producer_target": self.producer_target,
            "argument_name": self.argument_name,
        }


@dataclass(frozen=True, slots=True)
class TaskStorageContract:
    """Deterministic semantic storage contract for one functional task ABI."""

    roots: tuple[StorageRoot, ...]
    output_views: tuple[OutputView, ...]
    mutations: tuple[MutationBinding, ...]
    compatibility_digest: str

    def __post_init__(self) -> None:
        if tuple(root.root_id for root in self.roots) != tuple(range(len(self.roots))):
            raise ValueError("storage roots must have dense IDs")
        root_by_id = {root.root_id: root for root in self.roots}
        leaves: set[int] = set()
        used_roots: set[int] = set()
        for view in self.output_views:
            if view.leaf_index in leaves:
                raise ValueError("one output leaf has multiple storage bindings")
            leaves.add(view.leaf_index)
            root = root_by_id.get(view.root_id)
            if root is None:
                raise ValueError("output view references an unknown storage root")
            if view.offset_bytes + view.span_bytes > root.minimum_span_bytes:
                raise ValueError("output view exceeds its semantic storage span")
            used_roots.add(view.root_id)
        if used_roots != set(root_by_id):
            raise ValueError("semantic storage root is not referenced by an output")
        mutation_keys = [
            (
                item.input_position,
                item.producer_node,
                item.argument_name,
            )
            for item in self.mutations
        ]
        if len(set(mutation_keys)) != len(mutation_keys):
            raise ValueError("task storage contract contains duplicate mutations")
        if len(self.compatibility_digest) != 64:
            raise ValueError("task storage contract digest must be SHA-256")

    def identity(self) -> dict[str, object]:
        return {
            "roots": [root.identity() for root in self.roots],
            "output_views": [view.identity() for view in self.output_views],
            "mutations": [mutation.identity() for mutation in self.mutations],
        }

    def to_json(self) -> str:
        """Return deterministic standalone diagnostic serialization."""

        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def to_dict(self) -> dict[str, object]:
        """Return the versioned JSON-compatible contract record."""

        return {
            "schema": "shadowspill.task_storage_contract/v1",
            "compatibility_digest": self.compatibility_digest,
            **self.identity(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> TaskStorageContract:
        """Validate and restore one versioned contract record."""

        expected_keys = {
            "schema",
            "compatibility_digest",
            "roots",
            "output_views",
            "mutations",
        }
        if set(payload) != expected_keys:
            raise ValueError(
                "task storage contract fields differ from schema: "
                f"expected={sorted(expected_keys)}, actual={sorted(payload)}"
            )
        if payload["schema"] != "shadowspill.task_storage_contract/v1":
            raise ValueError("unsupported task storage contract schema")
        roots = tuple(
            _storage_root_from_record(item) for item in _records(payload, "roots")
        )
        output_views = tuple(
            _output_view_from_record(item) for item in _records(payload, "output_views")
        )
        mutations = tuple(
            _mutation_from_record(item) for item in _records(payload, "mutations")
        )
        identity = {
            "roots": [root.identity() for root in roots],
            "output_views": [view.identity() for view in output_views],
            "mutations": [mutation.identity() for mutation in mutations],
        }
        encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        calculated = hashlib.sha256(encoded.encode()).hexdigest()
        declared = payload["compatibility_digest"]
        if not isinstance(declared, str) or declared != calculated:
            raise ValueError("task storage contract digest does not match its contents")
        return cls(roots, output_views, mutations, calculated)

    @classmethod
    def from_json(cls, encoded: str) -> TaskStorageContract:
        """Validate and restore deterministic standalone JSON."""

        try:
            payload = json.loads(encoded)
        except json.JSONDecodeError as exc:
            raise ValueError("task storage contract is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("task storage contract JSON must contain an object")
        return cls.from_dict(payload)


def _records(
    payload: Mapping[str, object], field: str
) -> tuple[Mapping[str, object], ...]:
    value = payload[field]
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError(f"task storage contract {field!r} must be a list of objects")
    return tuple(value)


def _integer(record: Mapping[str, object], field: str) -> int:
    value = record.get(field)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"task storage contract field {field!r} must be an integer")
    return value


def _optional_integer(record: Mapping[str, object], field: str) -> int | None:
    value = record.get(field)
    if value is None:
        return None
    return _integer(record, field)


def _optional_string(record: Mapping[str, object], field: str) -> str | None:
    value = record.get(field)
    if value is None or isinstance(value, str):
        return value
    raise ValueError(f"task storage contract field {field!r} must be a string or null")


def _string(record: Mapping[str, object], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str):
        raise ValueError(f"task storage contract field {field!r} must be a string")
    return value


def _integer_tuple(record: Mapping[str, object], field: str) -> tuple[int, ...]:
    value = record.get(field)
    if not isinstance(value, list):
        raise ValueError(f"task storage contract field {field!r} must be a list")
    result: list[int] = []
    for item in value:
        if not isinstance(item, int) or isinstance(item, bool):
            raise ValueError(
                f"task storage contract field {field!r} must contain integers"
            )
        result.append(item)
    return tuple(result)


def _storage_root_from_record(record: Mapping[str, object]) -> StorageRoot:
    expected = {
        "root_id",
        "kind",
        "source_input",
        "producer_node",
        "producer_target",
        "producer_result",
        "minimum_span_bytes",
    }
    if set(record) != expected:
        raise ValueError("storage-root record fields differ from schema")
    try:
        kind = StorageRootKind(_string(record, "kind"))
    except ValueError as exc:
        raise ValueError("storage-root kind is unknown") from exc
    return StorageRoot(
        _integer(record, "root_id"),
        kind,
        _optional_integer(record, "source_input"),
        _optional_string(record, "producer_node"),
        _optional_string(record, "producer_target"),
        _optional_integer(record, "producer_result"),
        _integer(record, "minimum_span_bytes"),
    )


def _output_view_from_record(record: Mapping[str, object]) -> OutputView:
    expected = {
        "leaf_index",
        "root_id",
        "offset_bytes",
        "span_bytes",
        "shape",
        "stride",
        "dtype",
        "layout",
    }
    if set(record) != expected:
        raise ValueError("output-view record fields differ from schema")
    return OutputView(
        _integer(record, "leaf_index"),
        _integer(record, "root_id"),
        _integer(record, "offset_bytes"),
        _integer(record, "span_bytes"),
        _integer_tuple(record, "shape"),
        _integer_tuple(record, "stride"),
        _string(record, "dtype"),
        _string(record, "layout"),
    )


def _mutation_from_record(record: Mapping[str, object]) -> MutationBinding:
    expected = {
        "input_position",
        "replacement_output_leaf",
        "producer_node",
        "producer_target",
        "argument_name",
    }
    if set(record) != expected:
        raise ValueError("mutation record fields differ from schema")
    return MutationBinding(
        _integer(record, "input_position"),
        _optional_integer(record, "replacement_output_leaf"),
        _string(record, "producer_node"),
        _string(record, "producer_target"),
        _string(record, "argument_name"),
    )


@dataclass(frozen=True, slots=True)
class _InputRoot:
    position: int


@dataclass(frozen=True, slots=True)
class _FreshRoot:
    node: Node
    result_index: int


_SemanticRoot = _InputRoot | _FreshRoot


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

    output_node = next(
        (node for node in graph_module.graph.nodes if node.op == "output"), None
    )
    if output_node is None:
        raise CaptureError("task graph has no output node")
    _reject_tensor_getattrs(graph_module)
    leaves, _ = tree_flatten(output_node.args[0])
    values = _symbolic_output_values(graph_module, example_inputs)
    if len(values) != len(leaves):
        raise CaptureError("task output geometry arity differs from its FX graph")

    placeholders = tuple(
        node for node in graph_module.graph.nodes if node.op == "placeholder"
    )
    if len(placeholders) != len(example_inputs):
        raise CaptureError("task placeholder count differs from example arguments")
    canonical_inputs = _canonical_input_positions(example_inputs)
    placeholder_positions = {
        node: canonical_inputs[index] for index, node in enumerate(placeholders)
    }
    root_cache: dict[Node, _SemanticRoot] = {}

    tensor_outputs: list[tuple[int, Node, torch.Tensor, _SemanticRoot]] = []
    for leaf_index, (leaf, value) in enumerate(zip(leaves, values, strict=True)):
        if not isinstance(value, torch.Tensor):
            continue
        if not isinstance(leaf, Node):
            raise CaptureError(
                f"tensor output leaf {leaf_index} is not represented by an FX node"
            )
        root = _storage_root(leaf, placeholder_positions, root_cache)
        tensor_outputs.append((leaf_index, leaf, value, root))

    by_root: dict[_SemanticRoot, list[tuple[int, Node, torch.Tensor]]] = {}
    root_order: list[_SemanticRoot] = []
    for leaf_index, leaf, value, root in tensor_outputs:
        if root not in by_root:
            by_root[root] = []
            root_order.append(root)
        by_root[root].append((leaf_index, leaf, value))

    root_id = {root: index for index, root in enumerate(root_order)}
    base_offset: dict[_SemanticRoot, int] = {}
    spans: dict[_SemanticRoot, int] = {}
    roots: list[StorageRoot] = []
    for root in root_order:
        outputs = by_root[root]
        offsets = [_raw_offset_bytes(value) for _, _, value in outputs]
        origin = 0 if isinstance(root, _InputRoot) else min(offsets)
        minimum_span = max(
            offset - origin + _view_span_bytes(value)
            for offset, (_, _, value) in zip(offsets, outputs, strict=True)
        )
        base_offset[root] = origin
        spans[root] = minimum_span
        if isinstance(root, _InputRoot):
            source = example_inputs[root.position]
            if not isinstance(source, torch.Tensor):
                raise CaptureError("output aliases a non-tensor task input")
            if minimum_span > live_storage_bytes(source):
                raise CaptureError("output view exceeds its input storage")
            roots.append(
                StorageRoot(
                    root_id[root],
                    StorageRootKind.INPUT,
                    root.position,
                    None,
                    None,
                    None,
                    minimum_span,
                )
            )
        else:
            roots.append(
                StorageRoot(
                    root_id[root],
                    StorageRootKind.FRESH,
                    None,
                    root.node.name,
                    str(root.node.target),
                    root.result_index,
                    minimum_span,
                )
            )

    output_views = tuple(
        OutputView(
            leaf_index=leaf_index,
            root_id=root_id[root],
            offset_bytes=_raw_offset_bytes(value) - base_offset[root],
            span_bytes=_view_span_bytes(value),
            shape=tuple(int(extent) for extent in value.shape),
            stride=tuple(int(stride) for stride in value.stride()),
            dtype=str(value.dtype),
            layout=str(value.layout),
        )
        for leaf_index, _leaf, value, root in tensor_outputs
    )
    mutations = _capture_mutations(
        graph_module,
        placeholder_positions,
        root_cache,
    )
    explicit_bindings: list[MutationBinding] = []
    output_view_by_leaf = {view.leaf_index: view for view in output_views}
    root_by_id = {root.root_id: root for root in roots}
    for mutation in explicit_mutations:
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
        view = output_view_by_leaf.get(mutation.output_leaf_index)
        if view is None:
            raise CaptureError(
                "functional mutation replacement is not a tensor output: "
                f"leaf={mutation.output_leaf_index}, target={mutation.target}"
            )
        semantic_root = root_by_id[view.root_id]
        if (
            semantic_root.kind is StorageRootKind.INPUT
            and semantic_root.source_input != source_position
        ):
            raise CaptureError(
                "functional mutation replacement aliases a different task input: "
                f"leaf={mutation.output_leaf_index}, target={mutation.target}, "
                f"expected_input={source_position}, "
                f"actual_input={semantic_root.source_input}"
            )
        leaf = leaves[mutation.output_leaf_index]
        if not isinstance(leaf, Node):
            raise CaptureError("functional mutation output has no FX provenance")
        explicit_bindings.append(
            MutationBinding(
                source_position,
                mutation.output_leaf_index,
                leaf.name,
                str(leaf.target),
                mutation.target,
            )
        )
    mutations = tuple((*mutations, *explicit_bindings))
    identity = {
        "roots": [root.identity() for root in roots],
        "output_views": [view.identity() for view in output_views],
        "mutations": [mutation.identity() for mutation in mutations],
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return TaskStorageContract(
        roots=tuple(roots),
        output_views=output_views,
        mutations=mutations,
        compatibility_digest=hashlib.sha256(encoded.encode()).hexdigest(),
    )


def _symbolic_output_values(
    graph_module: GraphModule,
    example_inputs: tuple[object, ...],
) -> tuple[object, ...]:
    mode = FakeTensorMode(allow_non_fake_inputs=True)
    try:
        with mode, torch.no_grad():
            arguments = _fresh_symbolic_arguments(example_inputs)
            values, _ = tree_flatten(graph_module(*arguments))
    except BaseException as exc:
        raise CaptureError(
            "fresh symbolic task evaluation failed while deriving output geometry: "
            f"{exc}"
        ) from exc
    return tuple(values)


def _fresh_symbolic_arguments(
    example_inputs: tuple[object, ...],
) -> tuple[object, ...]:
    bases: dict[int, tuple[torch.Tensor, torch.device]] = {}
    result: list[object] = []
    for value in example_inputs:
        if not isinstance(value, torch.Tensor):
            result.append(value)
            continue
        if value.layout is not torch.strided:
            raise CaptureError("task storage contract requires strided tensor inputs")
        storage_identity = live_storage_identity(value)
        existing = bases.get(storage_identity)
        if existing is None:
            storage_bytes = live_storage_bytes(value)
            base = torch.empty(
                storage_bytes,
                dtype=torch.uint8,
                device=value.device,
            )
            existing = (base, value.device)
            bases[storage_identity] = existing
        base, device = existing
        if value.device != device:
            raise CaptureError("one aliased input storage spans multiple devices")
        symbolic = torch.empty(0, dtype=value.dtype, device=value.device)
        symbolic.set_(
            base.untyped_storage(),
            int(value.storage_offset()),
            tuple(int(extent) for extent in value.shape),
            tuple(int(stride) for stride in value.stride()),
        )
        result.append(symbolic)
    return tuple(result)


def _canonical_input_positions(example_inputs: tuple[object, ...]) -> tuple[int, ...]:
    representative: dict[int, int] = {}
    result: list[int] = []
    for position, value in enumerate(example_inputs):
        if not isinstance(value, torch.Tensor):
            result.append(position)
            continue
        storage_identity = live_storage_identity(value)
        result.append(representative.setdefault(storage_identity, position))
    return tuple(result)


def _storage_root(
    node: Node,
    placeholders: dict[Node, int],
    cache: dict[Node, _SemanticRoot],
) -> _SemanticRoot:
    existing = cache.get(node)
    if existing is not None:
        return existing
    position = placeholders.get(node)
    if position is not None:
        result: _SemanticRoot = _InputRoot(position)
        cache[node] = result
        return result
    source = _alias_source(node)
    result = (
        _storage_root(source, placeholders, cache)
        if source is not None
        else _FreshRoot(*_fresh_identity(node))
    )
    cache[node] = result
    return result


def _fresh_identity(node: Node) -> tuple[Node, int]:
    if node.op == "call_function" and node.target is operator.getitem:
        producer, index = node.args[:2]
        if isinstance(producer, Node) and isinstance(index, int):
            return producer, index
    return node, 0


def _alias_source(node: Node) -> Node | None:
    if node.op != "call_function":
        return None
    if node.target is operator.getitem:
        producer, index = node.args[:2]
        if isinstance(producer, (tuple, list)) and isinstance(index, int):
            selected = producer[index]
            return selected if isinstance(selected, Node) else None
        if not isinstance(producer, Node) or not isinstance(index, int):
            return None
        return _schema_alias_source(producer, index)
    return _schema_alias_source(node, 0)


def _schema_alias_source(node: Node, result_index: int) -> Node | None:
    schema = getattr(node.target, "_schema", None)
    if schema is None:
        return None
    contract = operator_alias_contract(schema)
    schema_result_index = 0 if len(contract.returns) == 1 else result_index
    if schema_result_index >= len(contract.returns):
        raise CaptureError(
            f"node {node.name!r} result {result_index} exceeds operator schema"
        )
    labels = contract.returns[schema_result_index].labels
    if not labels:
        return None
    matches: list[Node] = []
    for index, argument in enumerate(contract.arguments):
        if not labels.intersection(argument.labels):
            continue
        value = _schema_argument_value(node, index, argument.name)
        nodes, _ = tree_flatten(value)
        matches.extend(item for item in nodes if isinstance(item, Node))
    unique = tuple(dict.fromkeys(matches))
    if len(unique) != 1:
        raise CaptureError(
            "alias-producing operator must identify exactly one tensor source: "
            f"node={node.name}, target={node.target}, result={result_index}, "
            f"sources={[item.name for item in unique]}"
        )
    return unique[0]


def _capture_mutations(
    graph_module: GraphModule,
    placeholders: dict[Node, int],
    root_cache: dict[Node, _SemanticRoot],
) -> tuple[MutationBinding, ...]:
    result: list[MutationBinding] = []
    seen: set[tuple[int, str, str]] = set()
    for node in graph_module.graph.nodes:
        if node.op != "call_function":
            continue
        schema = getattr(node.target, "_schema", None)
        if schema is None:
            continue
        contract = operator_alias_contract(schema)
        for index, argument in enumerate(contract.arguments):
            if not argument.is_write:
                continue
            value = _schema_argument_value(node, index, argument.name)
            candidates, _ = tree_flatten(value)
            for candidate in candidates:
                if not isinstance(candidate, Node):
                    continue
                root = _storage_root(candidate, placeholders, root_cache)
                if not isinstance(root, _InputRoot):
                    continue
                key = (root.position, node.name, argument.name)
                if key in seen:
                    continue
                seen.add(key)
                result.append(
                    MutationBinding(
                        root.position,
                        None,
                        node.name,
                        str(node.target),
                        argument.name,
                    )
                )
    return tuple(result)


def _schema_argument_value(node: Node, index: int, name: str) -> Any:
    if index < len(node.args):
        return node.args[index]
    if name in node.kwargs:
        return node.kwargs[name]
    return None


def _reject_tensor_getattrs(graph_module: GraphModule) -> None:
    for node in graph_module.graph.nodes:
        if node.op != "get_attr":
            continue
        value: object = graph_module
        for component in str(node.target).split("."):
            value = getattr(value, component)
        if isinstance(value, torch.Tensor):
            raise CaptureError(
                "tensor get_attr must be lifted to an explicit task input: "
                f"node={node.name}, target={node.target}"
            )


def _raw_offset_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.storage_offset()) * tensor.element_size()


def _view_span_bytes(tensor: torch.Tensor) -> int:
    if tensor.layout is not torch.strided:
        raise CaptureError("task output contract currently requires strided tensors")
    if tensor.numel() == 0:
        return 0
    if any(stride < 0 for stride in tensor.stride()):
        raise CaptureError("task output contract does not support negative strides")
    last_element = sum(
        (extent - 1) * stride
        for extent, stride in zip(tensor.shape, tensor.stride(), strict=True)
    )
    return int((last_element + 1) * tensor.element_size())


__all__ = [
    "MutationBinding",
    "OutputView",
    "StorageRoot",
    "StorageRootKind",
    "TaskStorageContract",
    "capture_task_storage_contract",
]
