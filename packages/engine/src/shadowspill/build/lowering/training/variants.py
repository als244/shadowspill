"""Bind training variants to canonical storage objects."""

from __future__ import annotations

from shadowspill.errors import CaptureError
from shadowspill.pytorch.capture.artifacts import AotGraphPair
from shadowspill.pytorch.graph_pairs import (
    DifferentiatedStage,
    GraphPairVariant,
)
from shadowspill.task.slots import ObjectSlot

from ..catalog import CompiledTasks
from ..task_binding import (
    TaskStorageHandoff,
    resolve_stage_input_slots,
)
from .backward import (
    bind_backward_contributions,
    bind_backward_inputs,
    bind_prior_gradients,
    prior_gradient_positions,
)
from .outputs import bind_forward_outputs
from .records import (
    PreparedStageVariant,
    TrainingBoundaries,
    TrainingObjects,
)


def prepare_stage_variant(
    position: int,
    stage_index: int,
    stage: DifferentiatedStage,
    option: GraphPairVariant,
    objects: TrainingObjects,
    boundaries: TrainingBoundaries,
    compiled: CompiledTasks,
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
    forward = bind_forward_outputs(
        pair,
        forward_inputs,
        canonical_outputs,
        objects.catalog,
        compiled.geometry(pair.forward),
        compiled.contract(pair.forward),
        problem=(
            f"microbatch={position}, stage={stage_index}, "
            f"variant={option.option_id}, "
            f"artifact={pair.forward.compatibility_digest}"
        ),
    )
    forward_outputs, residuals, mutations, replacement_leaves, handoffs = forward
    backward_inputs = bind_backward_inputs(
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
    if prior_gradient_positions(pair.backward):
        backward_inputs = bind_prior_gradients(
            pair,
            backward_inputs,
            forward_inputs,
            objects.gradient_by_parameter,
            boundaries.cotangents,
            position,
        )
    contributions, backward_handoffs = _variant_backward_contributions(
        position,
        stage_index,
        option,
        forward_inputs,
        backward_inputs,
        objects,
        boundaries,
        compiled,
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
) -> tuple[ObjectSlot, ...]:
    return resolve_stage_input_slots(
        stage.example,
        pair.forward,
        root_objects=boundaries.root_objects[position],
        stage_outputs=boundaries.object_ids[position][:stage_index],
        compact_leaf_indices=True,
    )


def _variant_backward_contributions(
    position: int,
    stage_index: int,
    option: GraphPairVariant,
    forward_inputs: tuple[ObjectSlot, ...],
    backward_inputs: tuple[ObjectSlot, ...],
    objects: TrainingObjects,
    boundaries: TrainingBoundaries,
    compiled: CompiledTasks,
    parameter_ids: set[str],
) -> tuple[tuple[ObjectSlot, ...], tuple[TaskStorageHandoff, ...]]:
    pair = option.pair
    try:
        return bind_backward_contributions(
            position,
            pair,
            forward_inputs,
            backward_inputs,
            parameter_ids,
            objects.gradient_by_parameter,
            boundaries.cotangents,
            objects.catalog,
            compiled.geometry(pair.backward),
            compiled.contract(pair.backward),
        )
    except CaptureError as exc:
        raise CaptureError(
            "backward layout reconciliation failed for "
            f"microbatch={position}, stage={stage_index}, "
            f"variant={option.option_id}, artifact="
            f"{pair.backward.compatibility_digest[:12]}: {exc}"
        ) from exc
