"""Framework-owned capture records that lower into canonical ShadowSpill IR."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
from torch.fx import GraphModule
from torch.fx.node import Node, map_arg
from torch.utils._pytree import TreeSpec, tree_flatten, tree_unflatten

from .contracts import ObjectiveError, ObjectiveResult


@dataclass(frozen=True, slots=True)
class TensorGeometry:
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    storage_offset: int
    dtype: torch.dtype
    device_type: str
    requires_grad: bool

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> TensorGeometry:
        return cls(
            shape=tuple(tensor.shape),
            stride=tuple(tensor.stride()),
            storage_offset=int(tensor.storage_offset()),
            dtype=tensor.dtype,
            device_type=tensor.device.type,
            requires_grad=bool(tensor.requires_grad),
        )

    def identity(self) -> dict[str, object]:
        return {
            "shape": self.shape,
            "stride": self.stride,
            "storage_offset": self.storage_offset,
            "dtype": str(self.dtype),
            "device_type": self.device_type,
            "requires_grad": self.requires_grad,
        }


@dataclass(frozen=True, slots=True)
class GraphArtifact:
    """One explicit tensor graph and its structural ABI identity."""

    kind: Literal["forward", "backward", "inference", "optimizer"]
    graph_module: GraphModule
    tensor_inputs: tuple[TensorGeometry, ...]
    argument_count: int
    output_count: int
    operator_targets: tuple[str, ...]
    tensor_argument_positions: tuple[int, ...]
    tensor_argument_alias_groups: tuple[int, ...]
    compatibility_digest: str
    example_arguments: tuple[object, ...] = field(repr=False, compare=False)

    @classmethod
    def capture(
        cls,
        *,
        kind: Literal["forward", "backward", "inference", "optimizer"],
        graph_module: GraphModule,
        example_inputs: tuple[object, ...],
    ) -> GraphArtifact:
        graph_module, example_inputs, tensor_positions = _specialize_static_inputs(
            graph_module, example_inputs
        )
        tensor_arguments = tuple(
            value for value in example_inputs if isinstance(value, torch.Tensor)
        )
        tensor_inputs = tuple(
            TensorGeometry.from_tensor(value) for value in tensor_arguments
        )
        alias_group_by_storage: dict[int, int] = {}
        tensor_alias_groups = tuple(
            alias_group_by_storage.setdefault(
                int(value.untyped_storage()._cdata), len(alias_group_by_storage)
            )
            for value in tensor_arguments
        )
        operators = tuple(
            sorted(
                {
                    str(node.target)
                    for node in graph_module.graph.nodes
                    if node.op in {"call_function", "call_method", "call_module"}
                }
            )
        )
        output_node = next(
            node for node in graph_module.graph.nodes if node.op == "output"
        )
        outputs, _ = tree_flatten(output_node.args[0])
        identity = {
            "kind": kind,
            "graph": _canonical_graph(graph_module),
            "inputs": [geometry.identity() for geometry in tensor_inputs],
            "argument_count": len(example_inputs),
            "tensor_argument_positions": tensor_positions,
            "tensor_argument_alias_groups": tensor_alias_groups,
            "output_count": len(outputs),
            "operators": operators,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        }
        encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        return cls(
            kind=kind,
            graph_module=graph_module,
            tensor_inputs=tensor_inputs,
            argument_count=len(example_inputs),
            output_count=len(outputs),
            operator_targets=operators,
            tensor_argument_positions=tensor_positions,
            tensor_argument_alias_groups=tensor_alias_groups,
            compatibility_digest=hashlib.sha256(encoded.encode()).hexdigest(),
            example_arguments=example_inputs,
        )


def _specialize_static_inputs(
    graph_module: GraphModule, example_inputs: tuple[object, ...]
) -> tuple[GraphModule, tuple[object, ...], tuple[int, ...]]:
    """Replace guarded non-tensor placeholders with their captured constants."""

    placeholders = tuple(
        node for node in graph_module.graph.nodes if node.op == "placeholder"
    )
    if len(placeholders) != len(example_inputs):
        raise ValueError("graph placeholder count differs from example arguments")
    tensor_positions = tuple(
        index
        for index, value in enumerate(example_inputs)
        if isinstance(value, torch.Tensor)
    )
    if len(tensor_positions) == len(example_inputs):
        return graph_module, example_inputs, tensor_positions
    specialized = copy.deepcopy(graph_module)
    specialized_placeholders = tuple(
        node for node in specialized.graph.nodes if node.op == "placeholder"
    )
    for placeholder, value in zip(
        specialized_placeholders, example_inputs, strict=True
    ):
        if isinstance(value, torch.Tensor):
            continue

        def replace(
            node: Node,
            target: Node = placeholder,
            replacement: object = value,
        ) -> Any:
            return replacement if node is target else node

        for user in tuple(placeholder.users):
            user.args = map_arg(user.args, replace)
            user.kwargs = map_arg(user.kwargs, replace)
        specialized.graph.erase_node(placeholder)
    specialized.graph.lint()
    specialized.recompile()
    return (
        specialized,
        tuple(value for value in example_inputs if isinstance(value, torch.Tensor)),
        tensor_positions,
    )


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
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return {"type": type(value).__qualname__, "value": str(value)}


@dataclass(frozen=True, slots=True)
class AotGraphPair:
    """Explicit forward/backward pair for one recomputation choice."""

    forward: GraphArtifact
    backward: GraphArtifact
    recomputation: bool
    saved_value_count: int
    specialized_unit_tangent_count: int = 0


@dataclass(frozen=True, slots=True)
class ObjectiveSchema:
    """Loss/metrics output reconstruction without retaining framework closures."""

    metric_tree_spec: TreeSpec
    metric_leaf_count: int
    tensor_metric_positions: tuple[int, ...]
    static_metric_leaves: tuple[tuple[int, Any], ...]

    def rebuild_metrics(self, tensors: tuple[torch.Tensor, ...]) -> Any:
        if len(tensors) != len(self.tensor_metric_positions):
            raise ObjectiveError("objective metric tensor count changed")
        leaves: list[Any] = [None] * self.metric_leaf_count
        for position, value in self.static_metric_leaves:
            leaves[position] = copy.deepcopy(value)
        for position, value in zip(self.tensor_metric_positions, tensors, strict=True):
            leaves[position] = value
        return tree_unflatten(leaves, self.metric_tree_spec)


def normalize_objective_result(
    value: torch.Tensor | ObjectiveResult, *, require_grad: bool
) -> tuple[torch.Tensor, Any]:
    """Validate the scalar differentiable objective contract."""

    if isinstance(value, ObjectiveResult):
        loss, metrics = value.loss, value.metrics
    else:
        loss, metrics = value, None
    if not isinstance(loss, torch.Tensor):
        raise ObjectiveError("objective must return a tensor or ObjectiveResult")
    if loss.numel() != 1:
        raise ObjectiveError(
            f"objective loss must be scalar, got shape {tuple(loss.shape)}"
        )
    if not (loss.is_floating_point() or loss.is_complex()):
        raise ObjectiveError("objective loss must be floating point or complex")
    if require_grad and not loss.requires_grad:
        raise ObjectiveError("objective loss must require gradients")
    return loss, metrics


def capture_objective_schema(metrics: Any) -> ObjectiveSchema:
    """Freeze tensor/static metric positions after a storage-free probe."""

    leaves, tree_spec = tree_flatten(metrics)
    tensor_positions = tuple(
        index for index, value in enumerate(leaves) if isinstance(value, torch.Tensor)
    )
    static: list[tuple[int, Any]] = []
    try:
        for index, value in enumerate(leaves):
            if not isinstance(value, torch.Tensor):
                static.append((index, copy.deepcopy(value)))
    except BaseException as exc:
        raise ObjectiveError("static objective metrics must be copyable") from exc
    return ObjectiveSchema(
        metric_tree_spec=tree_spec,
        metric_leaf_count=len(leaves),
        tensor_metric_positions=tensor_positions,
        static_metric_leaves=tuple(static),
    )
