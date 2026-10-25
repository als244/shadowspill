"""Stable graph and representation identities for task compatibility."""

from __future__ import annotations

import hashlib
import json
import pickle
from typing import Literal

import torch
from torch.fx import GraphModule
from torch.fx.node import Node
from torch.utils._pytree import tree_flatten

from shadowspill.errors import CaptureError
from shadowspill.pytorch.accelerator import provider_version
from shadowspill.pytorch.capture.live_storage import live_storage_identity
from shadowspill.pytorch.capture.storage import (
    TaskStorageContract,
)
from shadowspill.pytorch.representations import (
    is_wrapper,
)

from .geometry import TensorGeometry


def _representation_identity(
    tensor: torch.Tensor, aliases: dict[int, int]
) -> dict[str, object]:
    if not is_wrapper(tensor):
        return {
            "alias": aliases.setdefault(live_storage_identity(tensor), len(aliases))
        }
    names, metadata = tensor.__tensor_flatten__()
    try:
        metadata_bytes = pickle.dumps(metadata, protocol=5)
    except Exception as error:
        raise CaptureError(
            "tensor representation metadata must be serializable"
        ) from error
    return {
        "type": f"{type(tensor).__module__}.{type(tensor).__qualname__}",
        "metadata": hashlib.sha256(metadata_bytes).hexdigest(),
        "components": [
            {
                "name": name,
                "geometry": TensorGeometry.from_tensor(value).identity(),
                "alias": (
                    None
                    if is_wrapper(value)
                    else aliases.setdefault(live_storage_identity(value), len(aliases))
                ),
                "representation": _representation_identity(value, aliases),
            }
            for name in names
            for value in (getattr(tensor, name),)
        ],
    }


def _operator_targets(graph_module: GraphModule) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                str(node.target)
                for node in graph_module.graph.nodes
                if node.op in {"call_function", "call_method", "call_module"}
            }
        )
    )


def _output_count(graph_module: GraphModule) -> int:
    output_node = next(node for node in graph_module.graph.nodes if node.op == "output")
    outputs, _ = tree_flatten(output_node.args[0])
    return len(outputs)


def _artifact_digest(
    kind: Literal["forward", "backward", "inference", "optimizer"],
    graph_module: GraphModule,
    tensor_inputs: tuple[TensorGeometry, ...],
    argument_count: int,
    tensor_positions: tuple[int, ...],
    tensor_alias_groups: tuple[int, ...],
    output_count: int,
    operators: tuple[str, ...],
    storage_contract: TaskStorageContract,
) -> str:
    identity = {
        "kind": kind,
        "graph": _canonical_graph(graph_module),
        "inputs": [geometry.identity() for geometry in tensor_inputs],
        "argument_count": argument_count,
        "tensor_argument_positions": tensor_positions,
        "tensor_argument_alias_groups": tensor_alias_groups,
        "output_count": output_count,
        "operators": operators,
        "storage_contract": storage_contract.identity(),
        "storage_contract_digest": storage_contract.compatibility_digest,
        "torch": torch.__version__,
        "provider": provider_version(),
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def _canonical_graph(graph_module: GraphModule) -> list[dict[str, object]]:
    """Describe FX semantics without task-, layer-, or node-ordinal names."""

    identities: dict[Node, int] = {}
    result: list[dict[str, object]] = []
    for node in graph_module.graph.nodes:
        identities[node] = len(identities)
        item: dict[str, object] = {"op": node.op}
        if node.op not in {"placeholder", "output"}:
            item["target"] = str(node.target)
        if node.op != "placeholder":
            item["args"] = _canonical_argument(node.args, identities)
            item["kwargs"] = _canonical_argument(node.kwargs, identities)
        result.append(item)
    return result


def _canonical_argument(value: object, identities: dict[Node, int]) -> object:
    if isinstance(value, Node):
        return {"node": identities[value]}
    if isinstance(value, tuple):
        return {"tuple": [_canonical_argument(item, identities) for item in value]}
    if isinstance(value, list):
        return {"list": [_canonical_argument(item, identities) for item in value]}
    if isinstance(value, dict):
        return {
            "dict": [
                (str(key), _canonical_argument(item, identities))
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            ]
        }
    if value is None or isinstance(value, bool | int | float | str):
        return value
    return {"type": type(value).__qualname__, "value": str(value)}
