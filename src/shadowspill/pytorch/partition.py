"""Automatic graph stages derived from repeated PyTorch module structure."""

from __future__ import annotations

import hashlib
import json
import operator
import os
import pickle
import tempfile
from collections.abc import Collection
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn
from torch.export.graph_signature import InputKind
from torch.fx import GraphModule, Interpreter, Node
from torch.fx.passes.fake_tensor_prop import FakeTensorProp
from torch.fx.passes.split_module import split_module
from torch.utils._pytree import tree_flatten

from .aot import (
    ExportCapture,
    TrainingObjectiveCapture,
    capture_graph_pair,
    rebind_backward_input_provenance,
)
from .capture import (
    AotGraphPair,
    GraphArtifact,
    TaskInputProvenance,
    TaskInputRole,
    TensorGeometry,
)
from .contracts import CaptureError
from .fx_graph_cache import SerializedFxGraph
from .output_contract import ExplicitMutation, TaskStorageContract
from .profiling import PlanningArtifactRecorder


@dataclass(frozen=True, slots=True)
class StageValueSource:
    """Root-graph provenance for one positional stage input."""

    root_input_index: int | None = None
    producer_stage_index: int | None = None
    producer_output_index: int | None = None

    def __post_init__(self) -> None:
        root = self.root_input_index is not None
        produced = self.producer_stage_index is not None
        if root == produced:
            raise ValueError("stage source must be either a root input or stage output")
        if root:
            if self.root_input_index is None or self.root_input_index < 0:
                raise ValueError("stage root-input source is invalid")
            if self.producer_output_index is not None:
                raise ValueError("root-input source cannot name a producer output")
        else:
            if (
                self.producer_stage_index is None
                or self.producer_stage_index < 0
                or self.producer_output_index is None
                or self.producer_output_index < 0
            ):
                raise ValueError("stage-output source is invalid")


@dataclass(frozen=True, slots=True)
class StageExample:
    """One split graph plus exact values observed at its functional ABI."""

    stage_id: str
    module_target: str
    graph_module: GraphModule
    inputs: tuple[object, ...]
    input_sources: tuple[StageValueSource | None, ...]
    input_provenance: tuple[TaskInputProvenance, ...]
    mutations: tuple[ExplicitMutation, ...]
    user_output_indices: tuple[int, ...]
    output: object


_StageRecord = tuple[
    str,
    GraphModule,
    tuple[object, ...],
    tuple[StageValueSource | None, ...],
    object,
]


@dataclass(frozen=True, slots=True)
class TrainingStage:
    """One automatic stage and its two legal differentiation variants."""

    example: StageExample
    differentiable_output_indices: tuple[int, ...]
    save_pair: AotGraphPair
    recompute_pair: AotGraphPair


@dataclass(frozen=True, slots=True)
class PartitionedTrainingCapture:
    """One objective capture decomposed into executable training stages."""

    training: TrainingObjectiveCapture
    partitioned: PartitionedExport
    stages: tuple[TrainingStage, ...]


@dataclass(frozen=True, slots=True)
class PartitionedExport:
    """Executable split root and topologically ordered stage examples."""

    root: GraphModule
    root_inputs: tuple[object, ...]
    root_input_provenance: tuple[TaskInputProvenance, ...]
    stages: tuple[StageExample, ...]
    repeated_groups: tuple[str, ...]
    user_output_indices: tuple[int, ...]


def training_parameter_stage_owners(
    captures: tuple[PartitionedTrainingCapture, ...],
    parameter_names: Collection[str],
) -> dict[str, tuple[int, ...]]:
    """Return the training stages whose backward passes contribute each parameter.

    Export makes parameters explicit root inputs.  Stage partitioning preserves
    that provenance in :class:`StageValueSource`, so optimizer grouping can use
    the same semantic stage boundaries without inspecting module-name patterns
    or runtime allocation behavior.
    """

    known = frozenset(parameter_names)
    owners: dict[str, set[int]] = {}
    expected_stage_count: int | None = None
    for capture in captures:
        if expected_stage_count is None:
            expected_stage_count = len(capture.stages)
        elif len(capture.stages) != expected_stage_count:
            raise CaptureError(
                "microbatch positions produced different training-stage counts"
            )
        input_specs = tuple(
            capture.training.exported.exported_program.graph_signature.input_specs
        )
        for stage_index, stage in enumerate(capture.stages):
            for source in stage.example.input_sources:
                if source is None or source.root_input_index is None:
                    continue
                try:
                    spec = input_specs[source.root_input_index]
                except IndexError as exc:
                    raise CaptureError(
                        "stage parameter provenance refers outside the Export ABI"
                    ) from exc
                if spec.kind is not InputKind.PARAMETER:
                    continue
                target = spec.target
                if not isinstance(target, str) or not target.startswith("model."):
                    raise CaptureError(
                        "objective Export parameter target is not rooted at model: "
                        f"{target!r}"
                    )
                name = target.removeprefix("model.")
                if name not in known:
                    raise CaptureError(
                        f"stage parameter {name!r} is absent from the optimizer model"
                    )
                owners.setdefault(name, set()).add(stage_index)
    return {name: tuple(sorted(indices)) for name, indices in owners.items()}


class _StageRecorder(Interpreter):
    def __init__(self, module: GraphModule) -> None:
        super().__init__(module)
        self.calls: list[tuple[str, tuple[object, ...], object]] = []

    def call_module(
        self, target: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Any:
        if kwargs:
            raise CaptureError("automatic stage calls require a positional ABI")
        output = super().call_module(target, args, kwargs)
        self.calls.append((str(target), args, output))
        return output


_GRAPH_PAIR_CACHE_SCHEMA = "shadowspill.aot_graph_pair/v3"


@dataclass(frozen=True, slots=True)
class _CachedGraphArtifact:
    """A GraphArtifact without live values or GraphModule retracing semantics."""

    kind: Literal["forward", "backward", "inference", "optimizer"]
    graph: SerializedFxGraph
    tensor_inputs: tuple[TensorGeometry, ...]
    argument_count: int
    output_count: int
    operator_targets: tuple[str, ...]
    tensor_argument_positions: tuple[int, ...]
    tensor_argument_alias_groups: tuple[int, ...]
    input_provenance: tuple[TaskInputProvenance, ...]
    storage_contract: TaskStorageContract
    storage_contract_capture_ns: int
    compatibility_digest: str

    @classmethod
    def capture(cls, artifact: GraphArtifact) -> _CachedGraphArtifact:
        provenance = tuple(
            TaskInputProvenance(item.role, item.source, item.consumer_targets)
            for item in artifact.input_provenance
        )
        return cls(
            artifact.kind,
            SerializedFxGraph.capture(artifact.graph_module),
            artifact.tensor_inputs,
            artifact.argument_count,
            artifact.output_count,
            artifact.operator_targets,
            artifact.tensor_argument_positions,
            artifact.tensor_argument_alias_groups,
            provenance,
            artifact.storage_contract,
            artifact.storage_contract_capture_ns,
            artifact.compatibility_digest,
        )

    def restore(self) -> GraphArtifact:
        graph_module = self.graph.restore()
        arguments = _synthetic_fake_arguments_from_geometry(
            self.tensor_inputs,
            self.tensor_argument_alias_groups,
            self.argument_count,
        )
        FakeTensorProp(graph_module).propagate(*arguments)
        return GraphArtifact(
            kind=self.kind,
            graph_module=graph_module,
            tensor_inputs=self.tensor_inputs,
            argument_count=self.argument_count,
            output_count=self.output_count,
            operator_targets=self.operator_targets,
            tensor_argument_positions=self.tensor_argument_positions,
            tensor_argument_alias_groups=self.tensor_argument_alias_groups,
            input_provenance=self.input_provenance,
            storage_contract=self.storage_contract,
            storage_contract_capture_ns=self.storage_contract_capture_ns,
            compatibility_digest=self.compatibility_digest,
            example_arguments=arguments,
        )


@dataclass(frozen=True, slots=True)
class _CachedAotGraphPair:
    """Persistent, value-free form of one save or recomputation pair."""

    forward: _CachedGraphArtifact
    backward: _CachedGraphArtifact
    recomputation: bool
    saved_value_count: int
    specialized_unit_tangent_count: int

    @classmethod
    def capture(cls, pair: AotGraphPair) -> _CachedAotGraphPair:
        return cls(
            _CachedGraphArtifact.capture(pair.forward),
            _CachedGraphArtifact.capture(pair.backward),
            pair.recomputation,
            pair.saved_value_count,
            pair.specialized_unit_tangent_count,
        )

    def restore(self) -> AotGraphPair:
        return AotGraphPair(
            forward=self.forward.restore(),
            backward=self.backward.restore(),
            recomputation=self.recomputation,
            saved_value_count=self.saved_value_count,
            specialized_unit_tangent_count=self.specialized_unit_tangent_count,
        )


class _TrainingGraphPairCache:
    """Reuse AOT graph code while preserving occurrence-specific storages."""

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        read_enabled: bool = True,
        write_enabled: bool = True,
        overwrite: bool = False,
        artifact_recorder: PlanningArtifactRecorder | None = None,
    ) -> None:
        self._pairs: dict[
            tuple[str, tuple[int, ...], bool], tuple[AotGraphPair, AotGraphPair]
        ] = {}
        self._root = None if root is None else Path(root).expanduser()
        self._read_enabled = read_enabled
        self._write_enabled = write_enabled
        self._overwrite = overwrite
        self._artifact_recorder = artifact_recorder
        self._keys_seen: set[tuple[str, tuple[int, ...], bool]] = set()
        self.hits = 0
        self.misses = 0

    @property
    def unique_keys(self) -> int:
        return len(self._keys_seen)

    def resolve(
        self,
        example: StageExample,
        roots: tuple[int, ...],
        *,
        specialize_unit_tangents: bool,
    ) -> tuple[AotGraphPair, AotGraphPair]:
        stage_abi = GraphArtifact.input_compatibility_digest(
            graph_module=example.graph_module,
            example_inputs=example.inputs,
            explicit_mutations=example.mutations,
            input_provenance=example.input_provenance,
        )
        key = (stage_abi, roots, specialize_unit_tangents)
        self._keys_seen.add(key)
        existing = self._pairs.get(key)
        if existing is None:
            existing = self._read(key)
            if existing is not None:
                self._pairs[key] = existing
                self.hits += 1
                return (
                    _rebind_graph_pair(existing[0], example, roots),
                    _rebind_graph_pair(existing[1], example, roots),
                )
            existing = self._capture(example, roots, specialize_unit_tangents)
            self._pairs[key] = existing
            self._write(key, existing)
            self.misses += 1
            return existing
        self.hits += 1
        save_pair, recompute_pair = existing
        return (
            _rebind_graph_pair(save_pair, example, roots),
            _rebind_graph_pair(recompute_pair, example, roots),
        )

    @staticmethod
    def _capture(
        example: StageExample,
        roots: tuple[int, ...],
        specialize_unit_tangents: bool,
    ) -> tuple[AotGraphPair, AotGraphPair]:
        return (
            capture_graph_pair(
                example.graph_module,
                example.inputs,
                recomputation=False,
                original_output=example.output,
                root_output_positions=roots,
                specialize_unit_tangents=specialize_unit_tangents,
                explicit_mutations=example.mutations,
                input_provenance=example.input_provenance,
            ),
            capture_graph_pair(
                example.graph_module,
                example.inputs,
                recomputation=True,
                original_output=example.output,
                root_output_positions=roots,
                specialize_unit_tangents=specialize_unit_tangents,
                explicit_mutations=example.mutations,
                input_provenance=example.input_provenance,
            ),
        )

    def _path(self, key: tuple[str, tuple[int, ...], bool]) -> Path | None:
        if self._root is None:
            return None
        payload = {
            "schema": _GRAPH_PAIR_CACHE_SCHEMA,
            "stage_abi": key[0],
            "roots": key[1],
            "specialize_unit_tangents": key[2],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        selection = hashlib.sha256(encoded.encode()).hexdigest()
        return self._root / "v3" / key[0][:2] / key[0] / selection / "graph_pairs.pt"

    def _manifest_path(self, key: tuple[str, tuple[int, ...], bool]) -> Path | None:
        path = self._path(key)
        return None if path is None else path.with_name("manifest.json")

    def _read(
        self, key: tuple[str, tuple[int, ...], bool]
    ) -> tuple[AotGraphPair, AotGraphPair] | None:
        path = self._path(key)
        if path is None or not self._read_enabled:
            return None
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except FileNotFoundError:
            return None
        except (
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            pickle.UnpicklingError,
        ) as exc:
            raise CaptureError(f"AOT graph-pair cache entry {path} is invalid") from exc
        if (
            not isinstance(payload, dict)
            or payload.get("schema") != _GRAPH_PAIR_CACHE_SCHEMA
            or payload.get("key") != key
        ):
            raise CaptureError(f"AOT graph-pair cache entry {path} has the wrong key")
        pairs = payload.get("pairs")
        if (
            not isinstance(pairs, tuple)
            or len(pairs) != 2
            or any(not isinstance(pair, _CachedAotGraphPair) for pair in pairs)
        ):
            raise CaptureError(f"AOT graph-pair cache entry {path} has invalid data")
        hydrated = tuple(pair.restore() for pair in pairs)
        result = (hydrated[0], hydrated[1])
        self._record(key, path, "read", result)
        manifest_path = self._manifest_path(key)
        if manifest_path is not None and manifest_path.exists():
            self._record(key, manifest_path, "read", result, kind="graph_pair_manifest")
        return result

    def _write(
        self,
        key: tuple[str, tuple[int, ...], bool],
        pairs: tuple[AotGraphPair, AotGraphPair],
    ) -> None:
        path = self._path(key)
        if path is None or not self._write_enabled:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        cached = tuple(_CachedAotGraphPair.capture(pair) for pair in pairs)
        if path.exists() and not self._overwrite:
            self._record(key, path, "matched", pairs)
            return
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent
        )
        try:
            with os.fdopen(descriptor, "wb") as output:
                torch.save(
                    {
                        "schema": _GRAPH_PAIR_CACHE_SCHEMA,
                        "key": key,
                        "pairs": cached,
                    },
                    output,
                )
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
        finally:
            with suppress(FileNotFoundError):
                os.unlink(temporary)
        manifest_path = self._manifest_path(key)
        assert manifest_path is not None
        manifest = {
            "schema": _GRAPH_PAIR_CACHE_SCHEMA,
            "stage_aot_input_abi": key[0],
            "differentiable_root_positions": list(key[1]),
            "specialize_unit_tangents": key[2],
            "save": {
                "forward": pairs[0].forward.compatibility_digest,
                "backward": pairs[0].backward.compatibility_digest,
            },
            "recompute": {
                "forward": pairs[1].forward.compatibility_digest,
                "backward": pairs[1].backward.compatibility_digest,
            },
        }
        _atomic_json(manifest_path, manifest)
        self._record(key, path, "write", pairs)
        self._record(
            key,
            manifest_path,
            "write",
            pairs,
            kind="graph_pair_manifest",
        )

    def _record(
        self,
        key: tuple[str, tuple[int, ...], bool],
        path: Path,
        access: str,
        pairs: tuple[AotGraphPair, AotGraphPair],
        *,
        kind: str = "aot_graph_pairs",
    ) -> None:
        if self._artifact_recorder is None:
            return
        digest = path.parent.name
        dependencies = (
            key[0],
            pairs[0].forward.compatibility_digest,
            pairs[0].backward.compatibility_digest,
            pairs[1].forward.compatibility_digest,
            pairs[1].backward.compatibility_digest,
        )
        self._artifact_recorder(
            category="graphpairs",
            kind=kind,
            digest=digest,
            path=path,
            access=access,
            schema=_GRAPH_PAIR_CACHE_SCHEMA,
            dependencies=dependencies,
        )


def _synthetic_fake_arguments_from_geometry(
    tensor_inputs: tuple[TensorGeometry, ...],
    alias_groups: tuple[int, ...],
    argument_count: int,
) -> tuple[torch.Tensor, ...]:
    if argument_count != len(tensor_inputs):
        raise CaptureError("cached AOT graph contains an unexpected static argument")
    members: dict[int, list[TensorGeometry]] = {}
    for group, geometry in zip(
        alias_groups,
        tensor_inputs,
        strict=True,
    ):
        members.setdefault(group, []).append(geometry)
    owners: dict[int, torch.Tensor] = {}
    for group, geometries in members.items():
        devices = {item.device_type for item in geometries}
        if len(devices) != 1:
            raise CaptureError("cached AOT alias group spans multiple devices")
        storage_bytes = max(_geometry_storage_bytes(item) for item in geometries)
        owners[group] = torch.empty(
            storage_bytes,
            dtype=torch.uint8,
            device=next(iter(devices)),
        )
    arguments: list[torch.Tensor] = []
    for group, geometry in zip(
        alias_groups,
        tensor_inputs,
        strict=True,
    ):
        value = torch.empty(0, dtype=geometry.dtype, device=geometry.device_type).set_(
            owners[group].untyped_storage(),
            geometry.storage_offset,
            geometry.shape,
            geometry.stride,
        )
        value.requires_grad_(geometry.requires_grad)
        arguments.append(value)
    return tuple(arguments)


def _geometry_storage_bytes(geometry: TensorGeometry) -> int:
    if any(size == 0 for size in geometry.shape):
        elements = geometry.storage_offset
    elif not geometry.shape:
        elements = geometry.storage_offset + 1
    else:
        if any(stride < 0 for stride in geometry.stride):
            raise CaptureError("cached AOT graph uses a negative-stride tensor")
        elements = (
            geometry.storage_offset
            + 1
            + sum(
                (size - 1) * stride
                for size, stride in zip(geometry.shape, geometry.stride, strict=True)
            )
        )
    return max(0, elements * torch.empty((), dtype=geometry.dtype).element_size())


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w") as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        with suppress(FileNotFoundError):
            os.unlink(temporary)


def partition_export(
    capture: ExportCapture,
    module: nn.Module,
    *,
    partition: str = "auto",
    representative_root_inputs: tuple[object, ...] | None = None,
) -> PartitionedExport:
    """Split at outer repeated-module boundaries or retain one whole graph."""

    if partition not in {"auto", "whole"}:
        raise CaptureError("partition must be 'auto' or 'whole'")
    repeated = _outer_repeated_groups(module) if partition == "auto" else ()
    root_provenance = _root_input_provenance(
        capture,
        representative_root_inputs=representative_root_inputs,
    )
    graph_module = capture.exported_program.graph_module
    assignments = _partition_assignments(graph_module, repeated)
    try:
        root = split_module(
            graph_module,
            graph_module,
            lambda node: assignments[node],
            keep_original_order=True,
        )
        recorder = _StageRecorder(root)
        recorder.run(*capture.flat_inputs)
    except BaseException as exc:
        if isinstance(exc, CaptureError):
            raise
        raise CaptureError(f"automatic stage partition failed: {exc}") from exc
    stage_records: list[_StageRecord] = []
    call_nodes = tuple(node for node in root.graph.nodes if node.op == "call_module")
    if len(call_nodes) != len(recorder.calls):
        raise CaptureError("split root task topology differs from recorded calls")
    placeholders = tuple(node for node in root.graph.nodes if node.op == "placeholder")
    placeholder_index = {node: index for index, node in enumerate(placeholders)}
    stage_index_by_node = {node: index for index, node in enumerate(call_nodes)}
    for (target, inputs, output), call_node in zip(
        recorder.calls, call_nodes, strict=True
    ):
        child = root.get_submodule(target)
        if not isinstance(child, GraphModule):
            raise CaptureError(f"partition {target!r} is not an FX GraphModule")
        argument_leaves, _ = tree_flatten(call_node.args)
        input_leaves, _ = tree_flatten(inputs)
        if len(argument_leaves) != len(input_leaves):
            raise CaptureError("stage input structure differs from split-root topology")
        sources = tuple(
            _stage_value_source(
                leaf,
                placeholder_index=placeholder_index,
                stage_index_by_node=stage_index_by_node,
            )
            if isinstance(value, torch.Tensor)
            else None
            for leaf, value in zip(argument_leaves, input_leaves, strict=True)
        )
        stage_records.append((target, child, inputs, sources, output))
    if not stage_records:
        raise CaptureError("partitioning produced no executable stage")
    mutations_by_stage = _partition_mutations(
        capture,
        root,
        call_nodes=call_nodes,
        placeholder_index=placeholder_index,
        stage_index_by_node=stage_index_by_node,
        stage_records=tuple(stage_records),
    )
    user_outputs_by_stage = _partition_user_outputs(
        capture,
        root,
        placeholder_index=placeholder_index,
        stage_index_by_node=stage_index_by_node,
    )
    stages = tuple(
        StageExample(
            stage_id=f"stage_{index:04d}",
            module_target=target,
            graph_module=child,
            inputs=inputs,
            input_sources=sources,
            input_provenance=tuple(
                _stage_input_provenance(source, root_provenance)
                if source is not None
                else TaskInputProvenance(TaskInputRole.USER_INPUT)
                for source in sources
            ),
            mutations=mutations_by_stage.get(index, ()),
            user_output_indices=user_outputs_by_stage.get(index, ()),
            output=output,
        )
        for index, (target, child, inputs, sources, output) in enumerate(stage_records)
    )
    return PartitionedExport(
        root=root,
        root_inputs=capture.flat_inputs,
        root_input_provenance=root_provenance,
        stages=stages,
        repeated_groups=repeated,
        user_output_indices=capture.user_output_indices,
    )


def _partition_user_outputs(
    capture: ExportCapture,
    root: GraphModule,
    *,
    placeholder_index: dict[Node, int],
    stage_index_by_node: dict[Node, int],
) -> dict[int, tuple[int, ...]]:
    """Project root user outputs onto their stage-local output positions."""

    output_node = next(node for node in root.graph.nodes if node.op == "output")
    output_leaves, _ = tree_flatten(output_node.args[0])
    result: dict[int, list[int]] = {}
    for output_index in capture.user_output_indices:
        try:
            root_output = output_leaves[output_index]
        except IndexError as exc:
            raise CaptureError("Export user output is absent from split root") from exc
        source = _stage_value_source(
            root_output,
            placeholder_index=placeholder_index,
            stage_index_by_node=stage_index_by_node,
        )
        if source.producer_stage_index is None or source.producer_output_index is None:
            raise CaptureError("Export user output is not stage-produced")
        result.setdefault(source.producer_stage_index, []).append(
            source.producer_output_index
        )
    return {index: tuple(values) for index, values in result.items()}


def _stage_value_source(
    node: object,
    *,
    placeholder_index: dict[Node, int],
    stage_index_by_node: dict[Node, int],
) -> StageValueSource:
    if not isinstance(node, Node):
        raise CaptureError("tensor stage input has no split-root FX provenance")
    root_index = placeholder_index.get(node)
    if root_index is not None:
        return StageValueSource(root_input_index=root_index)
    stage_index = stage_index_by_node.get(node)
    if stage_index is not None:
        return StageValueSource(
            producer_stage_index=stage_index,
            producer_output_index=0,
        )
    if node.op == "call_function" and node.target is operator.getitem:
        producer, output_index = node.args[:2]
        if isinstance(producer, Node) and isinstance(output_index, int):
            stage_index = stage_index_by_node.get(producer)
            if stage_index is not None:
                return StageValueSource(
                    producer_stage_index=stage_index,
                    producer_output_index=output_index,
                )
    raise CaptureError(
        "tensor stage input has unsupported split-root provenance: "
        f"node={node.name}, op={node.op}, target={node.target}"
    )


def _partition_mutations(
    capture: ExportCapture,
    root: GraphModule,
    *,
    call_nodes: tuple[Node, ...],
    placeholder_index: dict[Node, int],
    stage_index_by_node: dict[Node, int],
    stage_records: tuple[_StageRecord, ...],
) -> dict[int, tuple[ExplicitMutation, ...]]:
    """Project root Export mutations onto the stage that creates the value."""

    del call_nodes
    output_node = next(node for node in root.graph.nodes if node.op == "output")
    output_leaves, _ = tree_flatten(output_node.args[0])
    result: dict[int, list[ExplicitMutation]] = {}
    for mutation in capture.mutations:
        try:
            root_output = output_leaves[mutation.output_index]
        except IndexError as exc:
            raise CaptureError(
                "Export mutation output is absent from split root"
            ) from exc
        source = _stage_value_source(
            root_output,
            placeholder_index=placeholder_index,
            stage_index_by_node=stage_index_by_node,
        )
        if source.producer_stage_index is None or source.producer_output_index is None:
            raise CaptureError("Export mutation replacement is not stage-produced")
        stage_index = source.producer_stage_index
        sources = stage_records[stage_index][3]
        candidates = tuple(
            position
            for position, input_source in enumerate(sources)
            if input_source is not None
            and input_source.root_input_index == mutation.input_index
        )
        if len(candidates) != 1:
            raise CaptureError(
                "Export mutation target does not resolve to one producer-stage input: "
                f"stage={stage_index}, target={mutation.target!r}, "
                f"root_input={mutation.input_index}, candidates={candidates}"
            )
        result.setdefault(stage_index, []).append(
            ExplicitMutation(
                candidates[0],
                source.producer_output_index,
                mutation.target,
            )
        )
    return {index: tuple(values) for index, values in result.items()}


def capture_training_stages(
    partitioned: PartitionedExport,
    *,
    graph_pair_cache: _TrainingGraphPairCache | None = None,
) -> tuple[TrainingStage, ...]:
    """Differentiate every stage independently for save/recompute planning."""

    cache = graph_pair_cache or _TrainingGraphPairCache()
    stages: list[TrainingStage] = []
    for index, example in enumerate(partitioned.stages):
        leaves, _ = tree_flatten(example.output)
        if not leaves or any(not isinstance(value, torch.Tensor) for value in leaves):
            raise CaptureError("training stage outputs must be tensors")
        differentiable = tuple(
            position
            for position, value in enumerate(leaves)
            if value.requires_grad and (value.is_floating_point() or value.is_complex())
        )
        if not differentiable:
            raise CaptureError(f"training {example.stage_id} has no gradient output")
        roots = (
            (partitioned.user_output_indices[0],)
            if index == len(partitioned.stages) - 1
            else differentiable
        )
        if any(position not in differentiable for position in roots):
            raise CaptureError("terminal objective loss is not differentiable")
        save_pair, recompute_pair = cache.resolve(
            example,
            roots,
            specialize_unit_tangents=index == len(partitioned.stages) - 1,
        )
        stages.append(
            TrainingStage(
                example=example,
                differentiable_output_indices=roots,
                save_pair=save_pair,
                recompute_pair=recompute_pair,
            )
        )
    return tuple(stages)


def partition_training_capture(
    capture: TrainingObjectiveCapture,
    *,
    partition: str = "auto",
    graph_pair_cache: _TrainingGraphPairCache | None = None,
    representative_root_inputs: tuple[object, ...] | None = None,
) -> PartitionedTrainingCapture:
    """Partition and differentiate one captured objective template."""

    partitioned = partition_export(
        capture.exported,
        capture.capture_module,
        partition=partition,
        representative_root_inputs=representative_root_inputs,
    )
    return PartitionedTrainingCapture(
        training=capture,
        partitioned=partitioned,
        stages=capture_training_stages(partitioned, graph_pair_cache=graph_pair_cache),
    )


def _rebind_graph_pair(
    pair: AotGraphPair,
    example: StageExample,
    roots: tuple[int, ...],
) -> AotGraphPair:
    """Bind shared AOT forward inputs to one equivalent stage occurrence.

    The cached backward artifact describes a structural ABI, not a concrete
    layer's storage.  Its residual and tangent slots are rebound to canonical
    Program objects later by ``TaskBindingResolver``.  Re-executing the forward
    graph here solely to manufacture occurrence-specific FakeTensor residuals
    would duplicate work without adding semantic validation.
    """

    forward_arguments: list[torch.Tensor] = []
    for position in pair.forward.tensor_argument_positions:
        try:
            value = example.inputs[position]
        except IndexError as exc:
            raise CaptureError(
                "reused stage forward argument positions changed"
            ) from exc
        if not isinstance(value, torch.Tensor):
            raise CaptureError("reused stage tensor argument became static")
        forward_arguments.append(value.detach())
    if len(forward_arguments) != pair.forward.argument_count:
        raise CaptureError("reused stage forward tensor argument count changed")
    forward_provenance = tuple(
        example.input_provenance[position]
        for position in pair.forward.tensor_argument_positions
    )
    forward = pair.forward.rebind_examples(
        tuple(forward_arguments),
        input_provenance=forward_provenance,
    )
    backward = pair.backward.rebind_examples(
        pair.backward.example_arguments,
        input_provenance=rebind_backward_input_provenance(pair, forward),
    )
    if len(roots) < pair.specialized_unit_tangent_count:
        raise CaptureError("specialized tangent count exceeds stage roots")
    return AotGraphPair(
        forward=forward,
        backward=backward,
        recomputation=pair.recomputation,
        saved_value_count=pair.saved_value_count,
        specialized_unit_tangent_count=pair.specialized_unit_tangent_count,
    )


def capture_forward_stages(
    partitioned: PartitionedExport,
) -> tuple[GraphArtifact, ...]:
    """Return one structural inference ABI for each automatic stage."""

    return tuple(
        GraphArtifact.capture(
            kind="inference",
            graph_module=stage.graph_module,
            example_inputs=stage.inputs,
            explicit_mutations=stage.mutations,
            input_provenance=stage.input_provenance,
        )
        for stage in partitioned.stages
    )


def _root_input_provenance(
    capture: ExportCapture,
    *,
    representative_root_inputs: tuple[object, ...] | None,
) -> tuple[TaskInputProvenance, ...]:
    """Translate Export input kinds into task-local value policy roles."""

    inputs = capture.flat_inputs
    if representative_root_inputs is not None and len(
        representative_root_inputs
    ) != len(inputs):
        raise CaptureError("representative root inputs differ from the Export ABI")
    specs = tuple(capture.exported_program.graph_signature.input_specs)
    if len(specs) != len(inputs):
        raise CaptureError("Export input signature differs from flattened inputs")
    role_by_kind = {
        InputKind.PARAMETER: TaskInputRole.PARAMETER,
        InputKind.BUFFER: TaskInputRole.BUFFER,
        InputKind.CONSTANT_TENSOR: TaskInputRole.CONSTANT,
        InputKind.USER_INPUT: TaskInputRole.USER_INPUT,
    }
    result: list[TaskInputProvenance] = []
    for index, (spec, value) in enumerate(zip(specs, inputs, strict=True)):
        role = role_by_kind.get(spec.kind, TaskInputRole.USER_INPUT)
        source = spec.target if isinstance(spec.target, str) else f"input_{index}"
        reference: torch.Tensor | None = None
        if representative_root_inputs is not None:
            supplied = representative_root_inputs[index]
            if isinstance(value, torch.Tensor):
                if not isinstance(supplied, torch.Tensor):
                    raise CaptureError(
                        "representative root tensor became a static value"
                    )
                expected = TensorGeometry.from_tensor(value)
                actual = TensorGeometry.from_tensor(supplied)
                if (
                    expected.shape,
                    expected.stride,
                    expected.storage_offset,
                    expected.dtype,
                ) != (
                    actual.shape,
                    actual.stride,
                    actual.storage_offset,
                    actual.dtype,
                ):
                    raise CaptureError(
                        "representative root tensor geometry differs from Export"
                    )
                reference = supplied
        result.append(TaskInputProvenance(role, source, representative_value=reference))
    return tuple(result)


def _stage_input_provenance(
    source: StageValueSource,
    roots: tuple[TaskInputProvenance, ...],
) -> TaskInputProvenance:
    if source.root_input_index is not None:
        try:
            return roots[source.root_input_index]
        except IndexError as exc:
            raise CaptureError(
                "stage input provenance is outside the root ABI"
            ) from exc
    assert source.producer_stage_index is not None
    assert source.producer_output_index is not None
    return TaskInputProvenance(
        TaskInputRole.ACTIVATION,
        (
            f"stage_{source.producer_stage_index:04d}."
            f"output_{source.producer_output_index:04d}"
        ),
    )


def _outer_repeated_groups(module: nn.Module) -> tuple[str, ...]:
    candidates: list[str] = []
    for path, parent in module.named_modules():
        children = tuple(parent.named_children())
        if len(children) < 2:
            continue
        type_counts: dict[type[nn.Module], int] = {}
        for _name, child in children:
            type_counts[type(child)] = type_counts.get(type(child), 0) + 1
        if max(type_counts.values(), default=0) < 2:
            continue
        if any(
            path == selected or path.startswith(f"{selected}.")
            for selected in candidates
        ):
            continue
        candidates.append(path)
    return tuple(candidates)


def _anchor(node: Node, repeated_groups: tuple[str, ...]) -> str | None:
    stack = node.meta.get("nn_module_stack")
    if not isinstance(stack, dict):
        return None
    paths = tuple(
        value[0]
        for value in stack.values()
        if isinstance(value, tuple) and value and isinstance(value[0], str)
    )
    matches: list[tuple[int, str]] = []
    for group in repeated_groups:
        prefix = f"{group}." if group else ""
        for path in paths:
            if path == group or not path.startswith(prefix):
                continue
            child = path[len(prefix) :].split(".", 1)[0]
            matches.append((len(group), f"{prefix}{child}"))
    return max(matches)[1] if matches else None


def _partition_assignments(
    graph_module: GraphModule, repeated_groups: tuple[str, ...]
) -> dict[Node, int]:
    assignments: dict[Node, int] = {}
    partition_id = 0
    previous_anchor: str | None = None
    for node in graph_module.graph.nodes:
        if node.op in {"placeholder", "output", "get_attr"}:
            continue
        current_anchor = _anchor(node, repeated_groups)
        if current_anchor is not None and current_anchor != previous_anchor:
            if previous_anchor is not None:
                partition_id += 1
            previous_anchor = current_anchor
        assignments[node] = partition_id
    if not assignments:
        raise CaptureError("export graph has no executable operations")
    return assignments
