"""Register training model, inputs, gradients, and optimizer state."""

from __future__ import annotations

from collections.abc import Collection

import torch
import torch.nn as nn
from torch.export.graph_signature import InputKind

from shadowspill.errors import CaptureError
from shadowspill.ir import ObjectRole, Persistence
from shadowspill.pytorch.capture.aot import TrainingObjectiveCapture
from shadowspill.pytorch.capture.live_storage import live_view_key
from shadowspill.pytorch.optimizer import (
    OptimizerCapture,
    OptimizerTensorRole,
    training_parameters_with_gradients,
)
from shadowspill.task.slots import ObjectSlot

from ...graph_pairs import PartitionedTrainingCapture
from ..catalog import (
    ObjectCatalog,
    register_model_state,
    tensor_value_role,
)
from ..program import execution_device_id, publish_storage_program
from .artifacts import (
    GradientBinding,
    OptimizerObjectBinding,
    TrainingObjects,
    TrainingStorageLayout,
)


def lower_training_storage_layout(
    model: nn.Module,
    captures: tuple[TrainingObjectiveCapture, ...],
    *,
    device_ordinal: int = 0,
) -> TrainingStorageLayout:
    """Assign stable model/input IDs before the optimizer is built."""

    if not captures:
        raise CaptureError("training storage layout requires a microbatch")
    device_id = execution_device_id(device_ordinal)
    inventory = ObjectCatalog(device_id=device_id)
    registrations, _parameter_objects = register_model_state(model, inventory)
    root_slots, _initial_inputs = _register_supplied_inputs(captures, inventory)
    return TrainingStorageLayout(
        publish_storage_program(inventory, device_ordinal=device_ordinal),
        registrations,
        root_slots,
    )


def register_training_objects(
    model: nn.Module,
    captures: tuple[PartitionedTrainingCapture, ...],
    optimizer: OptimizerCapture,
    *,
    device_id: str,
) -> TrainingObjects:
    catalog = ObjectCatalog(device_id=device_id)
    registrations, parameter_objects = register_model_state(model, catalog)
    base_captures = tuple(item.training for item in captures)
    root_slots, _initial_inputs = _register_supplied_inputs(
        base_captures,
        catalog,
    )
    gradients = _register_gradients(
        model,
        catalog,
        parameter_objects,
        receives_gradient=training_parameters_with_gradients(
            captures, dict(model.named_parameters())
        ),
    )
    gradient_by_parameter = {
        item.parameter_object_id: item.gradient_object_id for item in gradients
    }
    optimizer_objects = _register_optimizer_objects(
        optimizer,
        catalog,
        gradients,
    )
    return TrainingObjects(
        catalog,
        registrations,
        root_slots,
        parameter_objects,
        gradients,
        gradient_by_parameter,
        optimizer_objects,
    )


#: The root inputs something outside the step has to supply, as Export
#: classifies them. Parameters and buffers are model state and are registered
#: from the module. Everything else a task consumes and no task produces has
#: to arrive from outside, and a lifted constant is one of those: Export lifts
#: a tensor a module built inline into the signature, where it belongs to
#: neither `named_parameters` nor `named_buffers`. Forward lowering registers
#: root inputs by excluding model state, which covers the same set.
_SUPPLIED_INPUT_KINDS = frozenset({InputKind.USER_INPUT, InputKind.CONSTANT_TENSOR})


def _register_supplied_inputs(
    captures: tuple[TrainingObjectiveCapture, ...], inventory: ObjectCatalog
) -> tuple[tuple[tuple[ObjectSlot, ...], ...], set[str]]:
    positions: list[tuple[ObjectSlot, ...]] = []
    initial: set[str] = set()
    for capture in captures:
        slots: list[ObjectSlot] = []
        for index, (spec, value) in enumerate(
            zip(
                capture.exported.exported_program.graph_signature.input_specs,
                capture.exported.flat_inputs,
                strict=True,
            )
        ):
            if spec.kind not in _SUPPLIED_INPUT_KINDS or not isinstance(
                value, torch.Tensor
            ):
                continue
            object_id = inventory.add(
                value,
                role=tensor_value_role(value, continuous_role=ObjectRole.INPUT),
                persistence=Persistence.STEP,
                retain_spill_copy=True,
            )
            slots.append(ObjectSlot(index, object_id))
            initial.add(object_id)
        positions.append(tuple(slots))
    return tuple(positions), initial


def _register_gradients(
    model: nn.Module,
    inventory: ObjectCatalog,
    parameter_objects: dict[tuple[int, int], str],
    *,
    receives_gradient: Collection[str],
) -> tuple[GradientBinding, ...]:
    """Give a gradient object to every parameter a task will write one for.

    A parameter the objective never reaches gets no gradient however it is
    flagged, and reserving one for it puts an object in the plan that no
    task produces: the schedule then fetches it for the optimizer and the
    runtime refuses, because nothing ever wrote it.
    """

    results: list[GradientBinding] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or name not in receives_gradient:
            continue
        parameter_id = parameter_objects[live_view_key(parameter)]
        # A program is lowered from fake tensors, so this describes the
        # gradient's geometry and allocates nothing; the object it becomes
        # is created in a pool when the plan runs.
        gradient = torch.empty_like(parameter, memory_format=torch.preserve_format)
        gradient_id = inventory.add(
            gradient, role=ObjectRole.GRADIENT, persistence=Persistence.STEP
        )
        results.append(GradientBinding(name, parameter_id, gradient_id))
    return tuple(results)


def _register_optimizer_objects(
    optimizer: OptimizerCapture,
    inventory: ObjectCatalog,
    gradients: tuple[GradientBinding, ...],
) -> tuple[OptimizerObjectBinding, ...]:
    parameter_names = {item.parameter_name for item in gradients}
    gradient_names = {f"gradient.{item.parameter_name}" for item in gradients}
    results: list[OptimizerObjectBinding] = []
    for binding in optimizer.bindings:
        if binding.role is OptimizerTensorRole.PARAMETER:
            if binding.name not in parameter_names:
                raise CaptureError(
                    f"optimizer parameter {binding.name!r} has no model binding"
                )
            continue
        if binding.role is OptimizerTensorRole.GRADIENT:
            if binding.name not in gradient_names:
                raise CaptureError(
                    f"optimizer gradient {binding.name!r} has no planned gradient"
                )
            continue
        if not binding.spillable:
            continue
        object_id = inventory.add(
            binding.tensor,
            role=ObjectRole.OPTIMIZER_STATE,
            persistence=Persistence.CHECKPOINT,
            retain_spill_copy=True,
        )
        results.append(
            OptimizerObjectBinding(
                binding.name,
                object_id,
                binding.role,
                binding.mutable,
            )
        )
    return tuple(results)


__all__ = ["lower_training_storage_layout", "register_training_objects"]
