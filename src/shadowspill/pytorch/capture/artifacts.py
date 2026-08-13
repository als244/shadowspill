"""Framework-owned capture records that lower into canonical ShadowSpill IR."""

from __future__ import annotations

import copy
import hashlib
import json
import time
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal

import torch
from torch.fx import GraphModule
from torch.fx.node import Node, map_arg
from torch.utils._pytree import TreeSpec, tree_flatten, tree_unflatten

from shadowspill.pytorch.capture.live_storage import live_storage_identity
from shadowspill.pytorch.capture.storage import (
    ExplicitMutation,
    TaskStorageContract,
    capture_task_storage_contract,
)
from shadowspill.pytorch.contracts import CaptureError, ObjectiveError, ObjectiveResult

if TYPE_CHECKING:
    from shadowspill.pytorch.partition import PartitionedExport


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


class TaskInputRole(StrEnum):
    """Semantic source of one explicit compiled-task tensor argument."""

    PARAMETER = "parameter"
    BUFFER = "buffer"
    CONSTANT = "constant"
    USER_INPUT = "user_input"
    ACTIVATION = "activation"
    RESIDUAL = "residual"
    TANGENT = "tangent"
    GRADIENT = "gradient"
    OPTIMIZER_STATE = "optimizer_state"
    OPTIMIZER_HYPERPARAMETER = "optimizer_hyperparameter"
    ANONYMOUS = "anonymous"


@dataclass(frozen=True, slots=True)
class TaskInputProvenance:
    """Offline provenance and optional authentic value for one task input.

    ``representative_value`` is deliberately excluded from equality and ABI
    identity. It is an occurrence-local view of initialized/user-owned CPU
    storage and never becomes semantic identity evidence.
    """

    role: TaskInputRole
    source: str | None = None
    consumer_targets: tuple[str, ...] = ()
    representative_value: torch.Tensor | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.role, TaskInputRole):
            raise TypeError("task input role has an invalid type")
        if self.source is not None and not self.source:
            raise ValueError("task input provenance source must be non-empty")
        if any(not item for item in self.consumer_targets):
            raise ValueError("task input consumer targets must be non-empty")
        value = self.representative_value
        if value is not None and not isinstance(value, torch.Tensor):
            raise TypeError("representative task input must be a Tensor")

    def structural_identity(self) -> dict[str, object]:
        """Return only fields that can change representative-value policy."""

        return {"role": self.role.value}


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
    input_provenance: tuple[TaskInputProvenance, ...]
    storage_contract: TaskStorageContract
    storage_contract_capture_ns: int = field(compare=False)
    compatibility_digest: str
    example_arguments: tuple[object, ...] = field(repr=False, compare=False)

    @classmethod
    def input_compatibility_digest(
        cls,
        *,
        graph_module: GraphModule,
        example_inputs: tuple[object, ...],
        explicit_mutations: tuple[ExplicitMutation, ...] = (),
        input_provenance: tuple[TaskInputProvenance, ...] | None = None,
    ) -> str:
        """Identify a graph/input ABI without evaluating the graph outputs."""

        original_inputs = example_inputs
        graph_module, example_inputs, tensor_positions = _specialize_static_inputs(
            graph_module, example_inputs
        )
        _normalize_input_provenance(
            graph_module,
            original_inputs,
            tensor_positions,
            input_provenance,
        )
        tensor_arguments = tuple(
            value for value in example_inputs if isinstance(value, torch.Tensor)
        )
        alias_group_by_storage: dict[int, int] = {}
        identity = {
            "graph": _canonical_graph(graph_module),
            "inputs": [
                TensorGeometry.from_tensor(value).identity()
                for value in tensor_arguments
            ],
            "tensor_argument_positions": tensor_positions,
            "tensor_argument_alias_groups": [
                alias_group_by_storage.setdefault(
                    live_storage_identity(value), len(alias_group_by_storage)
                )
                for value in tensor_arguments
            ],
            "mutations": [
                {
                    "input_position": item.input_position,
                    "output_leaf_index": item.output_leaf_index,
                    "target": item.target,
                }
                for item in explicit_mutations
            ],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        }
        encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()

    @classmethod
    def capture(
        cls,
        *,
        kind: Literal["forward", "backward", "inference", "optimizer"],
        graph_module: GraphModule,
        example_inputs: tuple[object, ...],
        explicit_mutations: tuple[ExplicitMutation, ...] = (),
        input_provenance: tuple[TaskInputProvenance, ...] | None = None,
    ) -> GraphArtifact:
        original_inputs = example_inputs
        graph_module, example_inputs, tensor_positions = _specialize_static_inputs(
            graph_module, example_inputs
        )
        provenance = _normalize_input_provenance(
            graph_module,
            original_inputs,
            tensor_positions,
            input_provenance,
        )
        tensor_arguments = _tensor_arguments(example_inputs)
        tensor_inputs = tuple(
            TensorGeometry.from_tensor(value) for value in tensor_arguments
        )
        tensor_alias_groups = _tensor_alias_groups(tensor_arguments)
        operators = _operator_targets(graph_module)
        output_count = _output_count(graph_module)
        storage_contract, contract_capture_ns = _capture_artifact_storage(
            graph_module,
            example_inputs,
            _compact_mutations(explicit_mutations, tensor_positions),
        )
        digest = _artifact_digest(
            kind,
            graph_module,
            tensor_inputs,
            len(example_inputs),
            tensor_positions,
            tensor_alias_groups,
            output_count,
            operators,
            storage_contract,
        )
        return cls(
            kind=kind,
            graph_module=graph_module,
            tensor_inputs=tensor_inputs,
            argument_count=len(example_inputs),
            output_count=output_count,
            operator_targets=operators,
            tensor_argument_positions=tensor_positions,
            tensor_argument_alias_groups=tensor_alias_groups,
            input_provenance=provenance,
            storage_contract=storage_contract,
            storage_contract_capture_ns=contract_capture_ns,
            compatibility_digest=digest,
            example_arguments=example_inputs,
        )

    def rebind_examples(
        self,
        example_arguments: tuple[object, ...],
        *,
        input_provenance: tuple[TaskInputProvenance, ...] | None = None,
    ) -> GraphArtifact:
        """Bind equivalent live tensors without re-extracting graph semantics.

        Structural AOT cache hits reuse the exact same FX graph and storage
        contract.  Only their occurrence-specific tensor storages differ.  The
        complete tensor geometry and input-alias ABI are sufficient to prove
        that the cached contract still applies; re-running the graph merely to
        rediscover that immutable contract is both redundant and expensive.
        """

        if len(example_arguments) != self.argument_count or any(
            not isinstance(value, torch.Tensor) for value in example_arguments
        ):
            raise CaptureError("rebound graph arguments differ from the tensor ABI")
        tensors = tuple(
            value for value in example_arguments if isinstance(value, torch.Tensor)
        )
        geometry = tuple(TensorGeometry.from_tensor(value) for value in tensors)
        if geometry != self.tensor_inputs:
            raise CaptureError("rebound graph tensor geometry differs from its ABI")
        alias_group_by_storage: dict[int, int] = {}
        alias_groups = tuple(
            alias_group_by_storage.setdefault(
                live_storage_identity(value), len(alias_group_by_storage)
            )
            for value in tensors
        )
        if alias_groups != self.tensor_argument_alias_groups:
            raise CaptureError("rebound graph input aliases differ from its ABI")
        provenance = self.input_provenance
        if input_provenance is not None:
            if len(input_provenance) != len(provenance):
                raise CaptureError("rebound graph input provenance arity changed")
            if tuple(item.role for item in input_provenance) != tuple(
                item.role for item in provenance
            ):
                raise CaptureError("rebound graph input provenance roles changed")
            provenance = input_provenance
        return replace(
            self,
            example_arguments=example_arguments,
            input_provenance=provenance,
        )


def _tensor_arguments(values: tuple[object, ...]) -> tuple[torch.Tensor, ...]:
    return tuple(value for value in values if isinstance(value, torch.Tensor))


def _tensor_alias_groups(values: tuple[torch.Tensor, ...]) -> tuple[int, ...]:
    group_by_storage: dict[int, int] = {}
    return tuple(
        group_by_storage.setdefault(
            live_storage_identity(value),
            len(group_by_storage),
        )
        for value in values
    )


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


def _compact_mutations(
    mutations: tuple[ExplicitMutation, ...],
    tensor_positions: tuple[int, ...],
) -> tuple[ExplicitMutation, ...]:
    compact_position = {
        original: compact for compact, original in enumerate(tensor_positions)
    }
    compact: list[ExplicitMutation] = []
    for mutation in mutations:
        try:
            input_position = compact_position[mutation.input_position]
        except KeyError as exc:
            raise ValueError(
                "functional mutation target was specialized as static"
            ) from exc
        compact.append(
            ExplicitMutation(
                input_position,
                mutation.output_leaf_index,
                mutation.target,
            )
        )
    return tuple(compact)


def _capture_artifact_storage(
    graph_module: GraphModule,
    example_inputs: tuple[object, ...],
    mutations: tuple[ExplicitMutation, ...],
) -> tuple[TaskStorageContract, int]:
    started = time.perf_counter_ns()
    contract = capture_task_storage_contract(
        graph_module,
        example_inputs,
        explicit_mutations=mutations,
    )
    return contract, time.perf_counter_ns() - started


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
        "cuda": torch.version.cuda,
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def _normalize_input_provenance(
    graph_module: GraphModule,
    original_inputs: tuple[object, ...],
    tensor_positions: tuple[int, ...],
    supplied: tuple[TaskInputProvenance, ...] | None,
) -> tuple[TaskInputProvenance, ...]:
    """Attach compact tensor roles and static FX consumer descriptions."""

    if supplied is None:
        original = tuple(
            TaskInputProvenance(TaskInputRole.ANONYMOUS) for _value in original_inputs
        )
    else:
        if len(supplied) != len(original_inputs):
            raise ValueError("task input provenance must match argument arity")
        original = supplied
    compact = tuple(original[position] for position in tensor_positions)
    placeholders = tuple(
        node for node in graph_module.graph.nodes if node.op == "placeholder"
    )
    if len(placeholders) != len(compact):
        raise ValueError("specialized task placeholders differ from tensor inputs")
    result: list[TaskInputProvenance] = []
    for item, placeholder in zip(compact, placeholders, strict=True):
        consumers = tuple(
            sorted(
                {
                    str(user.target)
                    for user in placeholder.users
                    if user.op in {"call_function", "call_method", "call_module"}
                }
            )
        )
        result.append(replace(item, consumer_targets=consumers))
    return tuple(result)


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


def capture_forward_stage_artifacts(
    partitioned: PartitionedExport,
) -> tuple[GraphArtifact, ...]:
    """Capture one structural inference ABI for each stage occurrence.

    Partitioning itself stops at :class:`Stage`. This conversion belongs to
    graph capture because it adds the geometry-dependent task ABI consumed by
    compilation and profiling.
    """

    return tuple(
        GraphArtifact.capture(
            kind="inference",
            graph_module=example.stage.graph_module,
            example_inputs=example.inputs,
            explicit_mutations=example.stage.mutations,
            input_provenance=example.stage.input_provenance,
        )
        for example in partitioned.stages
    )


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
