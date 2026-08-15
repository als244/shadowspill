"""Bind stage boundaries and graph-pair variants to canonical objects."""

from __future__ import annotations

import torch
from torch.utils._pytree import tree_flatten

from shadowspill.ir import ObjectRole, Persistence
from shadowspill.pytorch.capture.artifacts import AotGraphPair
from shadowspill.pytorch.capture.storage import TaskStorageContract
from shadowspill.pytorch.compilation.layout import CompiledTaskLayout

from ...contracts import CaptureError
from ...graph_pairs import (
    DifferentiatedStage,
    GraphPairVariant,
    PartitionedTrainingCapture,
    saved_value_footprint,
)
from ..catalog import (
    ObjectCatalog,
    TensorSlot,
    serialized_dtype_role,
    tensor_value_role,
)
from ..profiles import TaskProfileCatalog
from ..task_binding import (
    TaskBindingResolver,
    TaskStorageHandoff,
    resolve_stage_input_slots,
)
from .artifacts import (
    FixedTensorBinding,
    PreparedStageVariant,
    TrainingBoundaries,
    TrainingObjects,
)


def bind_training_boundaries(
    captures: tuple[PartitionedTrainingCapture, ...],
    objects: TrainingObjects,
    profiles: TaskProfileCatalog,
    metadata: tuple[str | None, ...],
) -> TrainingBoundaries:
    root_objects = tuple(
        {
            slot.leaf_index: slot.object_id
            for slot in _tensor_slots(
                capture.partitioned.root_inputs,
                objects.catalog,
            )
        }
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
                profiles,
                metadata[position],
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
    root_objects: dict[int, str],
    catalog: ObjectCatalog,
    profiles: TaskProfileCatalog,
    metadata_digest: str | None,
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
    inputs = resolve_stage_input_slots(
        stage.example,
        pair.forward,
        root_objects=root_objects,
        stage_outputs=tuple(
            {leaf_index: object_id for leaf_index, object_id in enumerate(ids)}
            for ids in prior_boundaries
        ),
        compact_leaf_indices=True,
    )
    resolver = TaskBindingResolver(
        catalog,
        pair.forward,
        inputs,
        profiles.layout(pair.forward, metadata_digest),
        storage_contract=profiles.contract(pair.forward),
    )
    ids: list[str] = []
    for index, value in enumerate(leaves):
        if not isinstance(value, torch.Tensor):
            raise CaptureError("training stage output became non-tensor")
        ids.append(
            resolver.bind_contract(
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
    tangent_values = pair.backward.example_arguments[pair.saved_value_count :]
    explicit_indices = stage.differentiable_output_indices[
        : len(stage.differentiable_output_indices) - pair.specialized_unit_tangent_count
    ]
    if len(tangent_values) != len(explicit_indices):
        raise CaptureError("stage tangent ABI differs from selected roots")
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


def prepare_training_variants(
    captures: tuple[PartitionedTrainingCapture, ...],
    objects: TrainingObjects,
    boundaries: TrainingBoundaries,
    profiles: TaskProfileCatalog,
    metadata: tuple[str | None, ...],
) -> tuple[tuple[dict[str, PreparedStageVariant], ...], ...]:
    prepared: list[tuple[dict[str, PreparedStageVariant], ...]] = []
    parameter_ids = set(objects.parameter_objects.values())
    for position, capture in enumerate(captures):
        stage_variants = tuple(
            _prepare_stage_variants(
                position,
                stage_index,
                stage,
                objects,
                boundaries,
                profiles,
                metadata[position],
                parameter_ids,
            )
            for stage_index, stage in enumerate(capture.stages)
        )
        prepared.append(stage_variants)
    return tuple(prepared)


def _prepare_stage_variants(
    position: int,
    stage_index: int,
    stage: DifferentiatedStage,
    objects: TrainingObjects,
    boundaries: TrainingBoundaries,
    profiles: TaskProfileCatalog,
    metadata_digest: str | None,
    parameter_ids: set[str],
) -> dict[str, PreparedStageVariant]:
    canonical_outputs = boundaries.object_ids[position][stage_index]
    terminal = stage_index == len(boundaries.object_ids[position]) - 1
    return {
        option.option_id: _prepare_stage_variant(
            position,
            stage_index,
            stage,
            option,
            objects,
            boundaries,
            profiles,
            metadata_digest,
            parameter_ids,
            canonical_outputs,
            terminal=terminal,
        )
        for option in stage.graph_pairs.variants
    }


def _prepare_stage_variant(
    position: int,
    stage_index: int,
    stage: DifferentiatedStage,
    option: GraphPairVariant,
    objects: TrainingObjects,
    boundaries: TrainingBoundaries,
    profiles: TaskProfileCatalog,
    metadata_digest: str | None,
    parameter_ids: set[str],
    canonical_outputs: tuple[str, ...],
    *,
    terminal: bool,
) -> PreparedStageVariant:
    pair = option.pair
    forward_inputs = _variant_forward_inputs(
        position,
        stage_index,
        stage,
        pair,
        boundaries,
    )
    forward = _stage_forward_outputs(
        pair,
        forward_inputs,
        canonical_outputs,
        objects.catalog,
        profiles.layout(pair.forward, metadata_digest),
        profiles.contract(pair.forward),
        context=(
            f"microbatch={position}, stage={stage_index}, "
            f"variant={option.option_id}, "
            f"artifact={pair.forward.compatibility_digest}"
        ),
    )
    forward_outputs, residuals, mutations, replacement_leaves, handoffs = forward
    backward_inputs = _stage_backward_inputs(
        position,
        stage,
        pair,
        forward_outputs,
        canonical_outputs,
        boundaries.cotangents,
        boundaries.fixed_tensors,
        objects.catalog,
        terminal=terminal,
    )
    contributions, backward_handoffs = _variant_backward_contributions(
        position,
        stage_index,
        option,
        forward_inputs,
        backward_inputs,
        objects,
        boundaries,
        profiles,
        metadata_digest,
        parameter_ids,
    )
    return PreparedStageVariant(
        stage,
        pair,
        forward_inputs,
        forward_outputs,
        backward_inputs,
        contributions,
        residuals,
        stage.example.stage.user_output_indices,
        mutations,
        replacement_leaves,
        handoffs,
        backward_handoffs,
    )


def _variant_forward_inputs(
    position: int,
    stage_index: int,
    stage: DifferentiatedStage,
    pair: AotGraphPair,
    boundaries: TrainingBoundaries,
) -> tuple[TensorSlot, ...]:
    stage_outputs = tuple(
        {leaf_index: object_id for leaf_index, object_id in enumerate(ids)}
        for ids in boundaries.object_ids[position][:stage_index]
    )
    return resolve_stage_input_slots(
        stage.example,
        pair.forward,
        root_objects=boundaries.root_objects[position],
        stage_outputs=stage_outputs,
        compact_leaf_indices=True,
    )


def _variant_backward_contributions(
    position: int,
    stage_index: int,
    option: GraphPairVariant,
    forward_inputs: tuple[TensorSlot, ...],
    backward_inputs: tuple[TensorSlot, ...],
    objects: TrainingObjects,
    boundaries: TrainingBoundaries,
    profiles: TaskProfileCatalog,
    metadata_digest: str | None,
    parameter_ids: set[str],
) -> tuple[tuple[TensorSlot, ...], tuple[TaskStorageHandoff, ...]]:
    pair = option.pair
    try:
        return _stage_backward_contributions(
            position,
            pair,
            forward_inputs,
            backward_inputs,
            parameter_ids,
            objects.gradient_by_parameter,
            boundaries.cotangents,
            objects.catalog,
            profiles.layout(pair.backward, metadata_digest),
            profiles.contract(pair.backward),
        )
    except CaptureError as exc:
        raise CaptureError(
            "backward layout reconciliation failed for "
            f"microbatch={position}, stage={stage_index}, "
            f"variant={option.option_id}, artifact="
            f"{pair.backward.compatibility_digest[:12]}: {exc}"
        ) from exc


def _stage_forward_outputs(
    pair: AotGraphPair,
    forward_inputs: tuple[TensorSlot, ...],
    canonical_outputs: tuple[str, ...],
    inventory: ObjectCatalog,
    compiled_layout: CompiledTaskLayout,
    storage_contract: TaskStorageContract,
    *,
    context: str,
) -> tuple[
    tuple[TensorSlot, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[int, ...],
    tuple[TaskStorageHandoff, ...],
]:
    public_count = pair.forward.output_count - pair.saved_value_count
    if public_count != len(canonical_outputs):
        raise CaptureError("stage boundary output count changed across AOT capture")
    slots: list[TensorSlot] = []
    saved_internal_object_ids: list[str] = []
    if compiled_layout.contract_digest != storage_contract.compatibility_digest:
        raise CaptureError(f"{context}: compiled layout belongs to another contract")
    resolver = TaskBindingResolver(
        inventory,
        pair.forward,
        forward_inputs,
        compiled_layout,
        storage_contract=storage_contract,
    )
    internal_root_ids = frozenset(
        saved_value_footprint(pair, storage_contract).internal_root_ids
    )
    root_by_leaf = {
        item.leaf_index: item.root_id for item in storage_contract.output_views
    }
    view_by_leaf = {
        item.leaf_index: item for item in storage_contract.output_views
    }
    for index in range(pair.forward.output_count):
        try:
            role = serialized_dtype_role(
                view_by_leaf[index].dtype,
                continuous_role=ObjectRole.ACTIVATION,
            )
        except KeyError as error:
            raise CaptureError(
                f"{context}: forward output {index} has no storage view"
            ) from error
        if index < public_count:
            object_id = resolver.bind_contract(
                index,
                role=role,
                persistence=Persistence.STEP,
                canonical_object_id=canonical_outputs[index],
            )
        else:
            object_id = resolver.bind_contract(
                index,
                role=role,
                persistence=Persistence.STEP,
            )
        slots.append(TensorSlot(index, object_id))
        if index >= public_count and root_by_leaf.get(index) in internal_root_ids:
            saved_internal_object_ids.append(object_id)
    return (
        tuple(slots),
        tuple(dict.fromkeys(saved_internal_object_ids)),
        resolver.mutation_object_ids,
        resolver.replacement_output_leaves,
        resolver.storage_handoffs,
    )


def _stage_backward_inputs(
    position: int,
    stage: DifferentiatedStage,
    pair: AotGraphPair,
    forward_outputs: tuple[TensorSlot, ...],
    canonical_outputs: tuple[str, ...],
    cotangent_by_activation: dict[tuple[int, str], str],
    fixed_tensors: dict[str, FixedTensorBinding],
    inventory: ObjectCatalog,
    *,
    terminal: bool,
) -> tuple[TensorSlot, ...]:
    public_count = pair.forward.output_count - pair.saved_value_count
    residuals = tuple(
        TensorSlot(index, slot.object_id)
        for index, slot in enumerate(forward_outputs[public_count:])
    )
    tangent_values = pair.backward.example_arguments[pair.saved_value_count :]
    explicit_indices = stage.differentiable_output_indices[
        : len(stage.differentiable_output_indices) - pair.specialized_unit_tangent_count
    ]
    if len(tangent_values) != len(explicit_indices):
        raise CaptureError("stage backward tangent arity changed")
    tangents: list[TensorSlot] = []
    for ordinal, (output_index, value) in enumerate(
        zip(explicit_indices, tangent_values, strict=True)
    ):
        if not isinstance(value, torch.Tensor):
            raise CaptureError("stage backward tangent became non-tensor")
        if terminal:
            matching = next(
                (
                    item.object_id
                    for item in fixed_tensors.values()
                    if tuple(item.value.shape) == tuple(value.shape)
                    and item.value.dtype == value.dtype
                ),
                None,
            )
            if matching is None:
                matching = inventory.add(
                    value,
                    role=ObjectRole.INPUT,
                    persistence=Persistence.STEP,
                    retain_spill_copy=True,
                )
                fixed_tensors[matching] = FixedTensorBinding(matching, value)
            tangent_id = matching
        else:
            tangent_id = cotangent_by_activation[
                (position, canonical_outputs[output_index])
            ]
        tangents.append(TensorSlot(len(residuals) + ordinal, tangent_id))
    return (*residuals, *tangents)


def _stage_backward_contributions(
    position: int,
    pair: AotGraphPair,
    forward_inputs: tuple[TensorSlot, ...],
    backward_inputs: tuple[TensorSlot, ...],
    parameter_ids: set[str],
    gradient_by_parameter: dict[str, str],
    cotangent_by_activation: dict[tuple[int, str], str],
    inventory: ObjectCatalog,
    compiled_layout: CompiledTaskLayout,
    storage_contract: TaskStorageContract,
) -> tuple[tuple[TensorSlot, ...], tuple[TaskStorageHandoff, ...]]:
    input_by_position = {slot.leaf_index: slot.object_id for slot in forward_inputs}
    resolver = TaskBindingResolver(
        inventory,
        pair.backward,
        backward_inputs,
        compiled_layout,
        storage_contract=storage_contract,
    )
    output_leaves = {item.leaf_index for item in storage_contract.output_views}
    results: list[TensorSlot] = []
    for output_index in pair.forward.tensor_argument_positions:
        object_id = input_by_position.get(output_index)
        if object_id is None or output_index not in output_leaves:
            continue
        destination: str | None
        if object_id in parameter_ids:
            destination = gradient_by_parameter[object_id]
        else:
            destination = cotangent_by_activation.get((position, object_id))
            if destination is None:
                continue
        bound = resolver.bind_contract(
            output_index,
            role=ObjectRole.GRADIENT,
            persistence=Persistence.STEP,
            canonical_object_id=destination,
        )
        results.append(TensorSlot(output_index, bound))
    return tuple(results), resolver.storage_handoffs


def _tensor_slots(
    values: tuple[object, ...], inventory: ObjectCatalog
) -> tuple[TensorSlot, ...]:
    return tuple(
        TensorSlot(
            index,
            inventory.add(
                value,
                role=tensor_value_role(value, continuous_role=ObjectRole.INPUT),
                persistence=Persistence.STEP,
            ),
        )
        for index, value in enumerate(values)
        if isinstance(value, torch.Tensor)
    )


__all__ = ["bind_training_boundaries", "prepare_training_variants"]
