"""Bind training boundaries to canonical storage objects."""

from __future__ import annotations

import torch
from torch.utils._pytree import tree_flatten

from shadowspill.build.lowering.geometry import tensor_value_role
from shadowspill.errors import CaptureError
from shadowspill.ir import ObjectRole, Persistence
from shadowspill.pytorch.graph_pairs import (
    DifferentiatedStage,
    PartitionedTrainingCapture,
)
from shadowspill.pytorch.representations import RootInputKey, tensor_components

from ..catalog import CompiledTasks
from ..objects import ObjectCatalog
from ..task_binding import (
    TaskBindingResolver,
    resolve_stage_input_slots,
)
from .backward import prior_gradient_positions
from .records import (
    FixedTensorBinding,
    TrainingBoundaries,
    TrainingObjects,
)


def bind_training_boundaries(
    captures: tuple[PartitionedTrainingCapture, ...],
    objects: TrainingObjects,
    compiled: CompiledTasks,
) -> TrainingBoundaries:
    root_objects = tuple(
        _root_objects(capture.partitioned.root_inputs, objects.catalog)
        for capture in captures
    )
    boundary_ids: list[tuple[tuple[str, ...], ...]] = []
    cotangents: dict[tuple[int, str], str] = {}
    fixed_tensors: dict[str, FixedTensorBinding] = {}
    public_outputs: dict[int, tuple[str, ...]] = {}
    for position, capture in enumerate(captures):
        position_boundaries: list[tuple[str, ...]] = []
        for stage_index, stage in enumerate(capture.stages):
            ids = _bind_canonical_stage_boundary(
                position,
                stage_index,
                capture,
                stage,
                position_boundaries,
                root_objects[position],
                objects.catalog,
                compiled,
            )
            position_boundaries.append(ids)
            terminal = stage_index == len(capture.stages) - 1
            if terminal:
                public_outputs[position] = tuple(
                    ids[index] for index in stage.example.stage.user_output_indices
                )
            _register_stage_cotangents(
                position,
                stage,
                ids,
                objects.catalog,
                cotangents,
                fixed_tensors,
                terminal=terminal,
            )
        boundary_ids.append(tuple(position_boundaries))
    return TrainingBoundaries(
        tuple(boundary_ids),
        root_objects,
        cotangents,
        fixed_tensors,
        public_outputs,
    )


def _bind_canonical_stage_boundary(
    position: int,
    stage_index: int,
    capture: PartitionedTrainingCapture,
    stage: DifferentiatedStage,
    prior_boundaries: list[tuple[str, ...]],
    root_objects: dict[RootInputKey, str],
    catalog: ObjectCatalog,
    compiled: CompiledTasks,
) -> tuple[str, ...]:
    leaves, _ = tree_flatten(stage.example.output)
    terminal = stage_index == len(capture.stages) - 1
    public_leaves = stage.example.stage.user_output_indices
    expected_public_count = (
        1 + len(capture.training.objective_schema.tensor_metric_positions)
        if terminal
        else 0
    )
    if len(public_leaves) != expected_public_count:
        raise CaptureError("stage user outputs differ from objective schema")
    pair = stage.graph_pairs.reference
    if pair.forward.output_count - pair.saved_value_count != len(leaves):
        raise CaptureError("stage boundary output count changed during AOT")
    # The earlier stages' boundaries as they are: each a tuple of object
    # IDs in leaf order. Rebuilt as mappings here, every stage paid for all
    # the stages before it.
    inputs = resolve_stage_input_slots(
        stage.example,
        pair.forward,
        root_objects=root_objects,
        stage_outputs=prior_boundaries,
        compact_leaf_indices=True,
    )
    resolver = TaskBindingResolver(
        catalog,
        pair.forward,
        inputs,
        compiled.geometry(pair.forward),
        storage_contract=compiled.contract(pair.forward),
    )
    ids: list[str] = []
    for index, value in enumerate(leaves):
        if not isinstance(value, torch.Tensor):
            raise CaptureError("training stage output became non-tensor")
        ids.append(
            resolver.bind(
                index,
                role=(
                    ObjectRole.OUTPUT
                    if index in public_leaves
                    else tensor_value_role(
                        value,
                        continuous_role=ObjectRole.ACTIVATION,
                    )
                ),
                persistence=Persistence.STEP,
            )
        )
    return tuple(ids)


def _register_stage_cotangents(
    position: int,
    stage: DifferentiatedStage,
    boundary_ids: tuple[str, ...],
    catalog: ObjectCatalog,
    cotangents: dict[tuple[int, str], str],
    fixed_tensors: dict[str, FixedTensorBinding],
    *,
    terminal: bool,
) -> None:
    pair = stage.graph_pairs.reference
    tangent_values = tuple(
        value
        for index, value in enumerate(pair.backward.example_arguments)
        if index >= pair.saved_value_count
        and index not in prior_gradient_positions(pair.backward)
    )
    explicit_indices = stage.differentiable_output_indices[
        : len(stage.differentiable_output_indices) - pair.specialized_unit_tangent_count
    ]
    if len(tangent_values) != len(explicit_indices):
        raise CaptureError("stage tangent contract differs from selected roots")
    for output_index, tangent in zip(
        explicit_indices,
        tangent_values,
        strict=True,
    ):
        if not isinstance(tangent, torch.Tensor):
            raise CaptureError("stage tangent became non-tensor")
        if terminal:
            tangent_id = catalog.add(
                tangent,
                role=ObjectRole.INPUT,
                persistence=Persistence.STEP,
                retain_spill_copy=True,
            )
            fixed_tensors.setdefault(
                tangent_id,
                FixedTensorBinding(tangent_id, tangent),
            )
        else:
            activation_id = boundary_ids[output_index]
            tangent_id = catalog.add(
                torch.empty_like(tangent),
                role=ObjectRole.GRADIENT,
                persistence=Persistence.STEP,
            )
            cotangents[(position, activation_id)] = tangent_id


def _root_objects(
    values: tuple[object, ...], inventory: ObjectCatalog
) -> dict[RootInputKey, str]:
    result: dict[RootInputKey, str] = {}
    for index, value in enumerate(values):
        if not isinstance(value, torch.Tensor):
            continue
        for path, component in tensor_components(value):
            key: RootInputKey = (index, path) if path else index
            result[key] = inventory.add(
                component,
                role=tensor_value_role(component, continuous_role=ObjectRole.INPUT),
                persistence=Persistence.STEP,
            )
    return result
