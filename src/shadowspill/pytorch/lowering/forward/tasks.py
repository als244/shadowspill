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

from ...capture import GraphArtifact
from ...partition import PartitionedExport, StageExample
from ..catalog import ObjectCatalog, TensorSlot
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
    tasks: list[TaskSpec] = []
    entrypoints: list[TaskEntrypoint] = []
    produced_aliases: set[str] = set()
    public_outputs: list[str] = []
    stage_outputs: list[dict[int, str]] = []
    for index, (stage, artifact, contract, layout, profile_id) in enumerate(
        zip(
            partitioned.stages,
            artifacts,
            physical.contracts,
            physical.layouts,
            physical.profile_ids,
            strict=True,
        )
    ):
        input_slots = resolve_stage_input_slots(
            stage,
            artifact,
            root_objects=objects.root_objects,
            stage_outputs=tuple(stage_outputs),
            compact_leaf_indices=False,
        )
        input_objects = tuple(dict.fromkeys(slot.object_id for slot in input_slots))
        resolver = TaskBindingResolver(
            objects.catalog,
            artifact,
            input_slots,
            layout,
            storage_contract=contract,
        )
        output_slots, outputs = _bind_forward_outputs(
            stage,
            resolver,
            objects.catalog,
            input_objects,
            produced_aliases,
            public_outputs,
        )
        stage_outputs.append({slot.leaf_index: slot.object_id for slot in output_slots})
        task_id = f"task_{index:06d}"
        tasks.append(
            TaskSpec(
                task_id,
                ResourceSpec(device_id, ResourceKind.COMPUTE),
                profile_id,
                dependencies=() if index == 0 else (f"task_{index - 1:06d}",),
                inputs=input_objects,
                outputs=outputs,
                mutations=tuple(
                    MutationSpec(object_id)
                    for object_id in resolver.mutation_object_ids
                ),
                phase="forward",
            )
        )
        entrypoints.append(
            TaskEntrypoint(
                task_id,
                stage.module_target,
                artifact,
                input_slots,
                output_slots,
                resolver.replacement_output_leaves,
                resolver.storage_handoffs,
            )
        )
    return ForwardTaskGraph(
        tuple(tasks),
        tuple(entrypoints),
        frozenset(produced_aliases),
        tuple(public_outputs),
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
            role=ObjectRole.ACTIVATION,
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
        if position in stage.user_output_indices:
            catalog.mark_output(object_id)
            public_outputs.append(object_id)
    return tuple(output_slots), tuple(outputs)


__all__ = ["emit_forward_tasks"]
