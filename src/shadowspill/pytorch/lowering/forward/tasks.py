"""Bind partitioned forward stages and emit their ordered task graph."""

from __future__ import annotations

import torch
from torch.utils._pytree import tree_flatten

from shadowspill.ir import (
    MutationSpec,
    ObjectRole,
    Persistence,
    ResourceKind,
    ResourceSpec,
    TaskSpec,
)
from shadowspill.pytorch.capture.artifacts import GraphArtifact
from shadowspill.pytorch.capture.storage import TaskStorageContract
from shadowspill.pytorch.compilation.layout import CompiledTaskLayout

from ...partition import PartitionedExport, StageExample
from ..catalog import ObjectCatalog, TensorSlot, tensor_value_role
from ..task_binding import TaskBindingResolver, resolve_stage_input_slots
from .artifacts import (
    ForwardObjects,
    ForwardPhysicalLayout,
    ForwardTaskGraph,
    TaskEntrypoint,
)


def emit_forward_tasks(
    partitioned: PartitionedExport,
    artifacts: tuple[GraphArtifact, ...],
    objects: ForwardObjects,
    physical: ForwardPhysicalLayout,
    *,
    device_id: str,
) -> ForwardTaskGraph:
    return _ForwardTaskEmitter(
        partitioned,
        artifacts,
        objects,
        physical,
        device_id=device_id,
    ).build()


class _ForwardTaskEmitter:
    def __init__(
        self,
        partitioned: PartitionedExport,
        artifacts: tuple[GraphArtifact, ...],
        objects: ForwardObjects,
        physical: ForwardPhysicalLayout,
        *,
        device_id: str,
    ) -> None:
        self.partitioned = partitioned
        self.artifacts = artifacts
        self.objects = objects
        self.physical = physical
        self.device_id = device_id
        self.tasks: list[TaskSpec] = []
        self.entrypoints: list[TaskEntrypoint] = []
        self.produced_aliases: set[str] = set()
        self.public_outputs: list[str] = []
        self.stage_outputs: list[dict[int, str]] = []

    def build(self) -> ForwardTaskGraph:
        for index, values in enumerate(self._stages()):
            self._emit_stage(index, *values)
        return ForwardTaskGraph(
            tuple(self.tasks),
            tuple(self.entrypoints),
            frozenset(self.produced_aliases),
            tuple(self.public_outputs),
        )

    def _stages(
        self,
    ) -> zip[
        tuple[
            StageExample,
            GraphArtifact,
            TaskStorageContract,
            CompiledTaskLayout,
            str,
        ]
    ]:
        return zip(
            self.partitioned.stages,
            self.artifacts,
            self.physical.contracts,
            self.physical.layouts,
            self.physical.profile_ids,
            strict=True,
        )

    def _emit_stage(
        self,
        index: int,
        stage: StageExample,
        artifact: GraphArtifact,
        contract: TaskStorageContract,
        layout: CompiledTaskLayout,
        profile_id: str,
    ) -> None:
        input_slots = self._stage_inputs(stage, artifact)
        input_objects = tuple(dict.fromkeys(slot.object_id for slot in input_slots))
        resolver = TaskBindingResolver(
            self.objects.catalog,
            artifact,
            input_slots,
            layout,
            storage_contract=contract,
        )
        output_slots, outputs = _bind_forward_outputs(
            stage,
            resolver,
            self.objects.catalog,
            input_objects,
            self.produced_aliases,
            self.public_outputs,
        )
        self.stage_outputs.append(
            {slot.leaf_index: slot.object_id for slot in output_slots}
        )
        task = self._task(index, profile_id, input_objects, outputs, resolver)
        self.tasks.append(task)
        self.entrypoints.append(
            TaskEntrypoint(
                task.task_id,
                stage.stage.module_target,
                artifact,
                input_slots,
                output_slots,
                resolver.replacement_output_leaves,
                resolver.storage_handoffs,
            )
        )

    def _stage_inputs(
        self,
        stage: StageExample,
        artifact: GraphArtifact,
    ) -> tuple[TensorSlot, ...]:
        return resolve_stage_input_slots(
            stage,
            artifact,
            root_objects=self.objects.root_objects,
            stage_outputs=tuple(self.stage_outputs),
            compact_leaf_indices=False,
        )

    def _task(
        self,
        index: int,
        profile_id: str,
        inputs: tuple[str, ...],
        outputs: tuple[str, ...],
        resolver: TaskBindingResolver,
    ) -> TaskSpec:
        task_id = f"task_{index:06d}"
        return TaskSpec(
            task_id,
            ResourceSpec(self.device_id, ResourceKind.COMPUTE),
            profile_id,
            dependencies=() if index == 0 else (f"task_{index - 1:06d}",),
            inputs=inputs,
            outputs=outputs,
            mutations=tuple(
                MutationSpec(object_id) for object_id in resolver.mutation_object_ids
            ),
            phase="forward",
        )


def _bind_forward_outputs(
    stage: StageExample,
    resolver: TaskBindingResolver,
    catalog: ObjectCatalog,
    input_objects: tuple[str, ...],
    produced_aliases: set[str],
    public_outputs: list[str],
) -> tuple[tuple[TensorSlot, ...], tuple[str, ...]]:
    input_aliases = {catalog.alias_id(value) for value in input_objects}
    output_slots: list[TensorSlot] = []
    outputs: list[str] = []
    leaves, _ = tree_flatten(stage.output)
    for position, leaf in enumerate(leaves):
        if not isinstance(leaf, torch.Tensor):
            continue
        object_id = resolver.bind(
            position,
            leaf,
            role=tensor_value_role(
                leaf,
                continuous_role=ObjectRole.ACTIVATION,
            ),
            persistence=Persistence.STEP,
        )
        output_slots.append(TensorSlot(position, object_id))
        alias_id = catalog.alias_id(object_id)
        if (
            object_id not in input_objects
            and alias_id not in input_aliases
            and object_id not in outputs
        ):
            outputs.append(object_id)
            produced_aliases.add(alias_id)
        if position in stage.stage.user_output_indices:
            catalog.mark_output(object_id)
            public_outputs.append(object_id)
    return tuple(output_slots), tuple(outputs)


__all__ = ["emit_forward_tasks"]
