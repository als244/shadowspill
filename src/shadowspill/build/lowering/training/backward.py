"""Bind training backward to canonical storage objects."""

from __future__ import annotations

import torch

from shadowspill.errors import CaptureError
from shadowspill.ir import ObjectRole, Persistence
from shadowspill.pytorch.capture.artifacts import AotGraphPair, GraphArtifact
from shadowspill.pytorch.capture.storage import TaskStorageContract
from shadowspill.pytorch.graph_pairs import (
    DifferentiatedStage,
)
from shadowspill.task.geometry import TaskStorageGeometry
from shadowspill.task.inputs import TaskInputRole
from shadowspill.task.slots import ObjectSlot

from ..objects import ObjectCatalog
from ..task_binding import (
    TaskBindingResolver,
    TaskStorageHandoff,
)
from .records import (
    FixedTensorBinding,
)


def prior_gradient_positions(backward: GraphArtifact) -> tuple[int, ...]:
    """Report the arguments a backward adds its gradients into.

    These are appended after the tangents, so anything that reasons about the
    tangent arity has to discount them.
    """

    return tuple(
        index
        for index, item in enumerate(backward.input_provenance)
        if item.role is TaskInputRole.GRADIENT
    )


def bind_prior_gradients(
    pair: AotGraphPair,
    backward_inputs: tuple[ObjectSlot, ...],
    forward_inputs: tuple[ObjectSlot, ...],
    gradient_by_parameter: dict[str, str],
    cotangent_by_activation: dict[tuple[int, str], str],
    example_position: int,
) -> tuple[ObjectSlot, ...]:
    """Bind the gradients an accumulating backward adds onto.

    The arguments were appended in the order of the outputs they accumulate
    into, so pairing the two in that order says which gradient each one is.
    Each of those outputs is the gradient of the forward argument at the same
    position, so that argument names the parameter gradient or activation
    cotangent the sum belongs in, just as for the contribution destinations.
    """

    sources = {slot.leaf_index: slot.object_id for slot in forward_inputs}
    logical_positions = {
        logical: physical
        for physical, (logical, path) in enumerate(pair.forward.input_components)
        if not path
    }
    priors: list[ObjectSlot] = []
    leaf_by_input = {
        mutation.input_position: mutation.replacement_output_leaf
        for mutation in pair.backward.storage_contract.mutations
        if mutation.replacement_output_leaf is not None
    }
    for position in prior_gradient_positions(pair.backward):
        leaf = leaf_by_input[position]
        parameter = (
            pair.gradient_provenance[leaf].source
            if pair.gradient_provenance
            and pair.gradient_provenance[leaf].role is TaskInputRole.PARAMETER
            else sources.get(leaf)
        )
        destination = (
            gradient_by_parameter.get(parameter) if parameter is not None else None
        )
        if destination is None:
            source = sources.get(logical_positions.get(leaf, leaf))
            if source is not None:
                destination = cotangent_by_activation.get((example_position, source))
        if destination is None:
            raise CaptureError(
                f"accumulating backward has no gradient destination at leaf {leaf}"
            )
        priors.append(ObjectSlot(position, destination))
    return (*backward_inputs, *priors)


def bind_backward_inputs(
    position: int,
    stage: DifferentiatedStage,
    pair: AotGraphPair,
    forward_outputs: tuple[ObjectSlot, ...],
    canonical_outputs: tuple[str, ...],
    cotangent_by_activation: dict[tuple[int, str], str],
    fixed_tensors: dict[str, FixedTensorBinding],
    inventory: ObjectCatalog,
    *,
    terminal: bool,
) -> tuple[ObjectSlot, ...]:
    public_count = pair.forward.output_count - pair.saved_value_count
    residuals = tuple(
        ObjectSlot(index, slot.object_id)
        for index, slot in enumerate(forward_outputs[public_count:])
    )
    priors = len(prior_gradient_positions(pair.backward))
    arguments = pair.backward.example_arguments
    tangent_values = arguments[pair.saved_value_count : len(arguments) - priors]
    explicit_indices = stage.differentiable_output_indices[
        : len(stage.differentiable_output_indices) - pair.specialized_unit_tangent_count
    ]
    if len(tangent_values) != len(explicit_indices):
        raise CaptureError("stage backward tangent arity changed")
    tangents: list[ObjectSlot] = []
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
        tangents.append(ObjectSlot(len(residuals) + ordinal, tangent_id))
    return (*residuals, *tangents)


def bind_backward_contributions(
    position: int,
    pair: AotGraphPair,
    forward_inputs: tuple[ObjectSlot, ...],
    backward_inputs: tuple[ObjectSlot, ...],
    parameter_ids: set[str],
    gradient_by_parameter: dict[str, str],
    cotangent_by_activation: dict[tuple[int, str], str],
    inventory: ObjectCatalog,
    compiled_layout: TaskStorageGeometry,
    storage_contract: TaskStorageContract,
) -> tuple[tuple[ObjectSlot, ...], tuple[TaskStorageHandoff, ...]]:
    input_by_position = {slot.leaf_index: slot.object_id for slot in forward_inputs}
    resolver = TaskBindingResolver(
        inventory,
        pair.backward,
        backward_inputs,
        compiled_layout,
        storage_contract=storage_contract,
    )
    output_leaves = {item.leaf_index for item in storage_contract.output_views}
    results: list[ObjectSlot] = []
    indices = (
        range(len(pair.gradient_provenance))
        if pair.gradient_provenance
        else pair.forward.tensor_argument_positions
    )
    logical_positions = {
        logical: physical
        for physical, (logical, path) in enumerate(pair.forward.input_components)
        if not path
    }
    for output_index in indices:
        object_id = input_by_position.get(
            logical_positions.get(output_index, output_index)
        )
        if output_index not in output_leaves:
            continue
        destination: str | None
        provenance = (
            pair.gradient_provenance[output_index] if pair.gradient_provenance else None
        )
        if provenance is not None and provenance.role is TaskInputRole.PARAMETER:
            assert provenance.source is not None
            destination = gradient_by_parameter[provenance.source]
        elif object_id in parameter_ids:
            destination = gradient_by_parameter[object_id]
        else:
            if object_id is None:
                continue
            destination = cotangent_by_activation.get((position, object_id))
            if destination is None:
                continue
        bound = resolver.bind(
            output_index,
            role=ObjectRole.GRADIENT,
            persistence=Persistence.STEP,
            canonical_object_id=destination,
        )
        results.append(ObjectSlot(output_index, bound))
    return tuple(results), resolver.storage_handoffs
