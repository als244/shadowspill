"""The contract built by replaying a task under a fake-tensor mode."""

from __future__ import annotations

import hashlib
import json
import operator
from collections.abc import Mapping
from typing import Any

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx import GraphModule, Node
from torch.utils._pytree import tree_flatten

from shadowspill.errors import CaptureError
from shadowspill.pytorch.capture.live_storage import (
    live_storage_bytes,
    live_storage_identity,
)
from shadowspill.pytorch.capture.schema import operator_alias_contract

from .records import MutationBinding, OutputView, StorageRoot, TaskStorageContract
from .roots import _FreshRoot, _InputRoot, _SemanticRoot


def make_storage_contract(
    roots: tuple[StorageRoot, ...],
    output_views: tuple[OutputView, ...],
    mutations: tuple[MutationBinding, ...],
) -> TaskStorageContract:
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
    placeholders: Mapping[Node, int],
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
