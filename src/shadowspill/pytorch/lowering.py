"""Lower framework-owned task artifacts into canonical ShadowSpill IR."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils._pytree import TreeSpec, tree_flatten

from shadowspill.ir import (
    AliasGroupSpec,
    DeviceSpec,
    MemoryLocation,
    ObjectRole,
    ObjectSpec,
    Persistence,
    Program,
    ResidencySpec,
    ResourceKind,
    ResourceSpec,
    TaskProfile,
    TaskSpec,
)

from .capture import GraphArtifact
from .contracts import CaptureError
from .partition import PartitionedExport
from .profiling import TaskMeasurement


@dataclass(frozen=True, slots=True)
class TensorSlot:
    """Position of one tensor leaf in a framework task ABI."""

    leaf_index: int
    object_id: str


@dataclass(frozen=True, slots=True)
class RegistrationBinding:
    """Original registered name associated with one logical tensor view."""

    name: str
    object_id: str
    parameter: bool


@dataclass(frozen=True, slots=True)
class TaskEntrypoint:
    """Framework-only executable binding for one canonical task."""

    task_id: str
    module_target: str
    artifact: GraphArtifact
    input_slots: tuple[TensorSlot, ...]
    output_slots: tuple[TensorSlot, ...]


@dataclass(frozen=True, slots=True)
class LoweredForwardProgram:
    """Canonical forward program plus non-serialized PyTorch bindings."""

    program: Program
    initial_residency: tuple[ResidencySpec, ...]
    final_residency: tuple[ResidencySpec, ...]
    entrypoints: tuple[TaskEntrypoint, ...]
    registrations: tuple[RegistrationBinding, ...]
    root_input_slots: tuple[TensorSlot, ...]
    output_tree_spec: TreeSpec
    output_leaf_count: int


@dataclass(frozen=True, slots=True)
class _TensorKey:
    storage_identity: int
    storage_offset: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype


@dataclass(slots=True)
class _ObjectRecord:
    object_id: str
    alias_group_id: str
    offset_bytes: int
    size_bytes: int
    role: ObjectRole
    persistence: Persistence


_ROLE_PRIORITY = {
    ObjectRole.OTHER: 0,
    ObjectRole.INPUT: 1,
    ObjectRole.ACTIVATION: 2,
    ObjectRole.OUTPUT: 3,
    ObjectRole.GRADIENT: 3,
    ObjectRole.OPTIMIZER_STATE: 4,
    ObjectRole.BUFFER: 5,
    ObjectRole.PARAMETER: 6,
}


class _TensorInventory:
    def __init__(self, *, device_id: str) -> None:
        self._device_id = device_id
        self._alias_by_storage: dict[int, str] = {}
        self._alias_sizes: dict[str, int] = {}
        self._retain_host: set[str] = set()
        self._object_by_key: dict[_TensorKey, str] = {}
        self._objects: list[_ObjectRecord] = []

    @staticmethod
    def key(tensor: torch.Tensor) -> _TensorKey:
        if tensor.layout is not torch.strided:
            raise CaptureError("program lowering currently requires strided tensors")
        return _TensorKey(
            storage_identity=tensor.untyped_storage()._cdata,
            storage_offset=int(tensor.storage_offset()),
            shape=tuple(tensor.shape),
            stride=tuple(tensor.stride()),
            dtype=tensor.dtype,
        )

    def add(
        self,
        tensor: torch.Tensor,
        *,
        role: ObjectRole,
        persistence: Persistence,
        retain_host_backing: bool = False,
    ) -> str:
        key = self.key(tensor)
        alias_id = self._alias_by_storage.get(key.storage_identity)
        storage_bytes = tensor.untyped_storage().nbytes()
        if alias_id is None:
            alias_id = f"alias_{len(self._alias_by_storage):06d}"
            self._alias_by_storage[key.storage_identity] = alias_id
            self._alias_sizes[alias_id] = storage_bytes
        elif self._alias_sizes[alias_id] != storage_bytes:
            raise CaptureError("one capture storage reported inconsistent byte extents")
        if retain_host_backing:
            self._retain_host.add(alias_id)
        object_id = self._object_by_key.get(key)
        if object_id is None:
            object_id = f"object_{len(self._objects):06d}"
            self._object_by_key[key] = object_id
            self._objects.append(
                _ObjectRecord(
                    object_id=object_id,
                    alias_group_id=alias_id,
                    offset_bytes=int(tensor.storage_offset()) * tensor.element_size(),
                    size_bytes=int(tensor.numel()) * tensor.element_size(),
                    role=role,
                    persistence=persistence,
                )
            )
        else:
            record = self._record(object_id)
            if persistence is Persistence.CHECKPOINT:
                record.persistence = persistence
            if _ROLE_PRIORITY[role] > _ROLE_PRIORITY[record.role]:
                record.role = role
        return object_id

    def alias_id(self, object_id: str) -> str:
        return self._record(object_id).alias_group_id

    def mark_output(self, object_id: str) -> None:
        record = self._record(object_id)
        if record.role in {ObjectRole.OTHER, ObjectRole.ACTIVATION}:
            record.role = ObjectRole.OUTPUT

    def _record(self, object_id: str) -> _ObjectRecord:
        index = int(object_id.removeprefix("object_"))
        return self._objects[index]

    def alias_groups(self) -> tuple[AliasGroupSpec, ...]:
        return tuple(
            AliasGroupSpec(
                alias_group_id,
                self._device_id,
                size_bytes,
                retain_host_backing=alias_group_id in self._retain_host,
            )
            for alias_group_id, size_bytes in self._alias_sizes.items()
        )

    def objects(self) -> tuple[ObjectSpec, ...]:
        return tuple(
            ObjectSpec(
                item.object_id,
                item.alias_group_id,
                item.offset_bytes,
                item.size_bytes,
                item.role,
                item.persistence,
            )
            for item in self._objects
        )


def lower_forward_program(
    model: nn.Module,
    partitioned: PartitionedExport,
    artifacts: tuple[GraphArtifact, ...],
    measurements: tuple[TaskMeasurement, ...],
    *,
    device_ordinal: int = 0,
) -> LoweredForwardProgram:
    """Create one deterministic canonical program from forward task positions."""

    stage_count = len(partitioned.stages)
    if len(artifacts) != stage_count or len(measurements) != stage_count:
        raise CaptureError("stage, artifact, and measurement counts must match")
    device_id = f"cuda_{device_ordinal}"
    inventory = _TensorInventory(device_id=device_id)
    registrations: list[RegistrationBinding] = []
    for name, parameter in model.named_parameters(remove_duplicate=False):
        object_id = inventory.add(
            parameter,
            role=ObjectRole.PARAMETER,
            persistence=Persistence.CHECKPOINT,
            retain_host_backing=True,
        )
        registrations.append(RegistrationBinding(name, object_id, True))
    parameter_names = {
        name for name, _ in model.named_parameters(remove_duplicate=False)
    }
    for name, buffer in model.named_buffers(remove_duplicate=False):
        if name in parameter_names:
            continue
        object_id = inventory.add(
            buffer,
            role=ObjectRole.BUFFER,
            persistence=Persistence.CHECKPOINT,
            retain_host_backing=True,
        )
        registrations.append(RegistrationBinding(name, object_id, False))

    root_input_slots: list[TensorSlot] = []
    for position, value in enumerate(partitioned.root_inputs):
        if not isinstance(value, torch.Tensor):
            continue
        object_id = inventory.add(
            value,
            role=ObjectRole.INPUT,
            persistence=Persistence.STEP,
            retain_host_backing=True,
        )
        root_input_slots.append(TensorSlot(position, object_id))

    profiles: list[TaskProfile] = []
    profile_by_key: dict[tuple[object, ...], str] = {}
    profile_ids: list[str] = []
    for artifact, measurement in zip(artifacts, measurements, strict=True):
        key = (
            artifact.compatibility_digest,
            measurement.runtime_ns,
            measurement.workspace_charged_bytes,
        )
        profile_id = profile_by_key.get(key)
        if profile_id is None:
            profile_id = f"profile_{len(profiles):06d}"
            profile_by_key[key] = profile_id
            profiles.append(
                TaskProfile(
                    profile_id,
                    measurement.runtime_ns,
                    measurement.workspace_charged_bytes,
                    artifact.compatibility_digest,
                )
            )
        profile_ids.append(profile_id)

    tasks: list[TaskSpec] = []
    entrypoints: list[TaskEntrypoint] = []
    produced_aliases: set[str] = set()
    last_output_objects: list[str] = []
    for index, (stage, artifact, profile_id) in enumerate(
        zip(partitioned.stages, artifacts, profile_ids, strict=True)
    ):
        input_leaves, _ = tree_flatten(stage.inputs)
        input_slots: list[TensorSlot] = []
        input_objects: list[str] = []
        for position, leaf in enumerate(input_leaves):
            if not isinstance(leaf, torch.Tensor):
                continue
            object_id = inventory.add(
                leaf, role=ObjectRole.INPUT, persistence=Persistence.STEP
            )
            input_slots.append(TensorSlot(position, object_id))
            if object_id not in input_objects:
                input_objects.append(object_id)

        input_aliases = {inventory.alias_id(value) for value in input_objects}
        output_leaves, _ = tree_flatten(stage.output)
        output_slots: list[TensorSlot] = []
        output_objects: list[str] = []
        for position, leaf in enumerate(output_leaves):
            if not isinstance(leaf, torch.Tensor):
                continue
            object_id = inventory.add(
                leaf, role=ObjectRole.ACTIVATION, persistence=Persistence.STEP
            )
            output_slots.append(TensorSlot(position, object_id))
            if object_id not in input_objects and object_id not in output_objects:
                output_objects.append(object_id)
                output_alias = inventory.alias_id(object_id)
                if output_alias not in input_aliases:
                    produced_aliases.add(output_alias)
            if index + 1 == stage_count:
                inventory.mark_output(object_id)
                last_output_objects.append(object_id)
        task_id = f"task_{index:06d}"
        dependencies = () if index == 0 else (f"task_{index - 1:06d}",)
        tasks.append(
            TaskSpec(
                task_id,
                ResourceSpec(device_id, ResourceKind.COMPUTE),
                profile_id,
                dependencies=dependencies,
                inputs=tuple(input_objects),
                outputs=tuple(output_objects),
                phase="forward",
            )
        )
        entrypoints.append(
            TaskEntrypoint(
                task_id,
                stage.module_target,
                artifact,
                tuple(input_slots),
                tuple(output_slots),
            )
        )

    alias_groups = inventory.alias_groups()
    objects = inventory.objects()
    initial_aliases = {
        item.alias_group_id
        for item in objects
        if item.role in {ObjectRole.PARAMETER, ObjectRole.BUFFER, ObjectRole.INPUT}
        and item.alias_group_id not in produced_aliases
    }
    final_host = {
        item.alias_group_id
        for item in objects
        if item.persistence is Persistence.CHECKPOINT or item.role is ObjectRole.INPUT
    }
    final_device = {inventory.alias_id(object_id) for object_id in last_output_objects}
    final_host -= final_device
    initial_residency = tuple(
        ResidencySpec(group.alias_group_id, MemoryLocation.HOST)
        for group in alias_groups
        if group.alias_group_id in initial_aliases
    )
    final_residency = tuple(
        ResidencySpec(
            group.alias_group_id,
            MemoryLocation.DEVICE
            if group.alias_group_id in final_device
            else MemoryLocation.HOST,
        )
        for group in alias_groups
        if group.alias_group_id in final_host | final_device
    )
    program = Program(
        devices=(DeviceSpec(device_id, "process_0", "cuda", device_ordinal),),
        alias_groups=alias_groups,
        objects=objects,
        profiles=tuple(profiles),
        tasks=tuple(tasks),
    )
    _last_leaves, last_tree_spec = tree_flatten(partitioned.stages[-1].output)
    return LoweredForwardProgram(
        program,
        initial_residency,
        final_residency,
        tuple(entrypoints),
        tuple(registrations),
        tuple(root_input_slots),
        last_tree_spec,
        len(_last_leaves),
    )


__all__ = [
    "LoweredForwardProgram",
    "RegistrationBinding",
    "TaskEntrypoint",
    "TensorSlot",
    "lower_forward_program",
]
