"""Value-free serialization for structural AOT graph pairs."""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeGuard

import torch
from torch.fx.passes.fake_tensor_prop import FakeTensorProp

from shadowspill.errors import CaptureError
from shadowspill.pytorch.capture.artifacts import (
    AotGraphPair,
    GraphArtifact,
    TaskInputProvenance,
    TensorGeometry,
)
from shadowspill.pytorch.capture.storage import TaskStorageContract
from shadowspill.pytorch.compilation.fx_graph import SerializedFxGraph

from .artifacts import GraphPairVariant


@dataclass(frozen=True, slots=True)
class CachedGraphArtifact:
    """A graph artifact without occurrence-local live values."""

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
    def capture(cls, artifact: GraphArtifact) -> CachedGraphArtifact:
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
        arguments = _synthetic_fake_arguments(
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
class CachedAotGraphPair:
    """Persistent, value-free form of one AOT forward/backward pair."""

    forward: CachedGraphArtifact
    backward: CachedGraphArtifact
    recomputation: bool
    saved_value_count: int
    specialized_unit_tangent_count: int

    @classmethod
    def capture(cls, pair: AotGraphPair) -> CachedAotGraphPair:
        return cls(
            CachedGraphArtifact.capture(pair.forward),
            CachedGraphArtifact.capture(pair.backward),
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


CachedGraphPairVariant = tuple[str, float | None, CachedAotGraphPair]


def valid_cached_variant(value: object) -> TypeGuard[CachedGraphPairVariant]:
    return (
        isinstance(value, tuple)
        and len(value) == 3
        and isinstance(value[0], str)
        and (value[1] is None or isinstance(value[1], float))
        and isinstance(value[2], CachedAotGraphPair)
    )


def restore_cached_variant(value: object) -> GraphPairVariant:
    if not valid_cached_variant(value):
        raise CaptureError("AOT graph-pair cache contains an invalid variant")
    option_id, memory_budget, cached = value
    return GraphPairVariant(option_id, memory_budget, cached.restore())


def atomic_json(path: Path, payload: dict[str, object]) -> None:
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


def _synthetic_fake_arguments(
    tensor_inputs: tuple[TensorGeometry, ...],
    alias_groups: tuple[int, ...],
    argument_count: int,
) -> tuple[torch.Tensor, ...]:
    if argument_count != len(tensor_inputs):
        raise CaptureError("cached AOT graph contains an unexpected static argument")
    members: dict[int, list[TensorGeometry]] = {}
    for group, geometry in zip(alias_groups, tensor_inputs, strict=True):
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
    for group, geometry in zip(alias_groups, tensor_inputs, strict=True):
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


__all__ = [
    "CachedAotGraphPair",
    "atomic_json",
    "restore_cached_variant",
    "valid_cached_variant",
]
