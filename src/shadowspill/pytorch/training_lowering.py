"""Lower explicit objective graph pairs into one accumulated training Program."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.export.graph_signature import InputKind
from torch.utils._pytree import tree_flatten

from shadowspill.ir import (
    DeviceSpec,
    MemoryLocation,
    MutationSpec,
    ObjectRole,
    Persistence,
    Program,
    RecomputationGroup,
    RecomputationOption,
    ResidencySpec,
    ResourceKind,
    ResourceSpec,
    TaskProfile,
    TaskSpec,
)

from .aot import TrainingCapture
from .capture import AotGraphPair, GraphArtifact
from .contracts import CaptureError
from .lowering import RegistrationBinding, TensorSlot, _TensorInventory
from .optimizer import (
    OptimizerCapture,
    OptimizerTensorRole,
)
from .profiling import TaskMeasurement


@dataclass(frozen=True, slots=True)
class GradientBinding:
    parameter_name: str
    parameter_object_id: str
    gradient_object_id: str


@dataclass(frozen=True, slots=True)
class FixedTensorBinding:
    """Frontend-owned constant tensor input required by a captured task."""

    object_id: str
    value: torch.Tensor


@dataclass(frozen=True, slots=True)
class TrainingTaskEntrypoint:
    task_id: str
    phase: str
    microbatch: int | None
    variant: str | None
    artifact: GraphArtifact | None
    input_slots: tuple[TensorSlot, ...]
    output_slots: tuple[TensorSlot, ...]
    gradient_output_slots: tuple[TensorSlot, ...] = ()
    public_output_count: int = 0


@dataclass(frozen=True, slots=True)
class LoweredTrainingProgram:
    program: Program
    initial_residency: tuple[ResidencySpec, ...]
    final_residency: tuple[ResidencySpec, ...]
    registrations: tuple[RegistrationBinding, ...]
    root_input_slots: tuple[tuple[TensorSlot, ...], ...]
    entrypoints: tuple[TrainingTaskEntrypoint, ...]
    gradients: tuple[GradientBinding, ...]
    fixed_tensors: tuple[FixedTensorBinding, ...]
    optimizer_task_id: str


@dataclass(frozen=True, slots=True)
class TrainingStorageLayout:
    """Deterministic model/input identities needed before optimizer capture."""

    program: Program
    registrations: tuple[RegistrationBinding, ...]
    root_input_slots: tuple[tuple[TensorSlot, ...], ...]


@dataclass(frozen=True, slots=True)
class _PairValues:
    forward_outputs: tuple[object, ...]
    backward_outputs: tuple[object, ...]


def lower_training_storage_layout(
    model: nn.Module,
    captures: tuple[TrainingCapture, ...],
    *,
    device_ordinal: int = 0,
) -> TrainingStorageLayout:
    """Assign stable model/input IDs before the optimizer factory is invoked."""

    if not captures:
        raise CaptureError("training storage layout requires a microbatch")
    device_id = f"cuda_{device_ordinal}"
    inventory = _TensorInventory(device_id=device_id)
    registrations, _parameter_objects = _register_model(model, inventory)
    root_slots, _initial_inputs = _register_microbatch_inputs(captures, inventory)
    return TrainingStorageLayout(
        Program(
            devices=(DeviceSpec(device_id, "process_0", "cuda", device_ordinal),),
            alias_groups=inventory.alias_groups(),
            objects=inventory.objects(),
            profiles=(),
            tasks=(),
        ),
        registrations,
        root_slots,
    )


def lower_training_program(
    model: nn.Module,
    captures: tuple[TrainingCapture, ...],
    measurements: dict[str, TaskMeasurement],
    optimizer: OptimizerCapture,
    *,
    device_ordinal: int = 0,
) -> LoweredTrainingProgram:
    """Compose microbatch graph-pair alternatives and one optimizer mutation."""

    if not captures:
        raise CaptureError("training lowering requires at least one microbatch")
    if optimizer.recurrent is None:
        raise CaptureError("initial training lowering requires a graphable optimizer")
    state_bindings = tuple(
        item
        for item in optimizer.bindings
        if item.role in {OptimizerTensorRole.STATE, OptimizerTensorRole.HYPERPARAMETER}
    )
    if state_bindings:
        raise CaptureError(
            "stateful optimizer lowering requires the initial/recurrent plan pair"
        )
    device_id = f"cuda_{device_ordinal}"
    inventory = _TensorInventory(device_id=device_id)
    registrations, parameter_objects = _register_model(model, inventory)
    root_slots, _initial_inputs = _register_microbatch_inputs(captures, inventory)
    gradients = _register_gradients(model, inventory, parameter_objects)
    gradient_by_parameter = {
        item.parameter_object_id: item.gradient_object_id for item in gradients
    }

    profile_by_digest: dict[str, str] = {}
    profiles: list[TaskProfile] = []

    def profile_id(artifact: GraphArtifact, extra_workspace: int = 0) -> str:
        measurement = measurements.get(artifact.compatibility_digest)
        if measurement is None:
            raise CaptureError(
                f"training profile scatter is missing {artifact.compatibility_digest}"
            )
        key = f"{artifact.compatibility_digest}:{extra_workspace}"
        existing = profile_by_digest.get(key)
        if existing is not None:
            return existing
        result = f"profile_{len(profiles):06d}"
        profile_by_digest[key] = result
        profiles.append(
            TaskProfile(
                result,
                measurement.runtime_ns,
                measurement.workspace_charged_bytes + extra_workspace,
                artifact.compatibility_digest,
            )
        )
        return result

    tasks: list[TaskSpec] = []
    entrypoints: list[TrainingTaskEntrypoint] = []
    groups: list[RecomputationGroup] = []
    public_objects_by_position: dict[int, tuple[str, ...]] = {}
    previous_backward_ids: tuple[str, ...] = ()
    all_backward_ids: list[str] = []
    fixed_tensors: dict[str, FixedTensorBinding] = {}
    gradient_bytes = sum(
        next(
            item.size_bytes
            for item in inventory.objects()
            if item.object_id == binding.gradient_object_id
        )
        for binding in gradients
    )

    for position, capture in enumerate(captures):
        option_task_ids: dict[str, tuple[str, str]] = {}
        option_residual_aliases: dict[str, tuple[str, ...]] = {}
        for variant, pair in (
            ("save", capture.save_pair),
            ("recompute", capture.recompute_pair),
        ):
            values = _execute_pair_examples(pair)
            forward_id = f"task_{len(tasks):06d}"
            backward_id = f"task_{len(tasks) + 1:06d}"
            forward_inputs = _tensor_slots(pair.forward.example_arguments, inventory)
            forward_output_slots, public_ids, residual_aliases = _forward_outputs(
                position,
                pair,
                capture,
                values.forward_outputs,
                inventory,
                public_objects_by_position,
            )
            backward_inputs, fixed = _backward_inputs(
                pair,
                values,
                inventory,
                forward_output_slots,
                capture,
            )
            fixed_tensors.setdefault(fixed.object_id, fixed)
            gradient_slots = _gradient_outputs(
                pair,
                values.backward_outputs,
                inventory,
                parameter_objects,
                gradient_by_parameter,
            )
            backward_inputs_objects = [slot.object_id for slot in backward_inputs]
            forward_input_aliases = {
                inventory.alias_id(slot.object_id) for slot in forward_inputs
            }
            outputs = (
                tuple(item.gradient_object_id for item in gradients)
                if position == 0
                else ()
            )
            mutations = (
                ()
                if position == 0
                else tuple(MutationSpec(item.gradient_object_id) for item in gradients)
            )
            if position != 0:
                backward_inputs_objects.extend(
                    item.gradient_object_id for item in gradients
                )
            tasks.extend(
                (
                    TaskSpec(
                        forward_id,
                        ResourceSpec(device_id, ResourceKind.COMPUTE),
                        profile_id(pair.forward),
                        dependencies=previous_backward_ids,
                        inputs=_unique(slot.object_id for slot in forward_inputs),
                        outputs=_unique(
                            slot.object_id
                            for slot in forward_output_slots
                            if inventory.alias_id(slot.object_id)
                            not in forward_input_aliases
                        ),
                        phase="forward",
                    ),
                    TaskSpec(
                        backward_id,
                        ResourceSpec(device_id, ResourceKind.COMPUTE),
                        profile_id(
                            pair.backward,
                            gradient_bytes if position != 0 else 0,
                        ),
                        dependencies=(forward_id, *previous_backward_ids),
                        inputs=_unique(backward_inputs_objects),
                        outputs=outputs,
                        mutations=mutations,
                        phase="backward",
                    ),
                )
            )
            entrypoints.extend(
                (
                    TrainingTaskEntrypoint(
                        forward_id,
                        "forward",
                        position,
                        variant,
                        pair.forward,
                        forward_inputs,
                        forward_output_slots,
                        public_output_count=len(public_ids),
                    ),
                    TrainingTaskEntrypoint(
                        backward_id,
                        "backward",
                        position,
                        variant,
                        pair.backward,
                        backward_inputs,
                        (),
                        gradient_output_slots=gradient_slots,
                    ),
                )
            )
            option_task_ids[variant] = (forward_id, backward_id)
            all_backward_ids.append(backward_id)
            option_residual_aliases[variant] = residual_aliases
        groups.append(
            RecomputationGroup(
                f"recompute_{position:04d}",
                tuple(
                    RecomputationOption(
                        variant,
                        option_task_ids[variant],
                        option_residual_aliases[variant],
                    )
                    for variant in ("save", "recompute")
                ),
            )
        )
        previous_backward_ids = tuple(
            task_id
            for variant in ("save", "recompute")
            for task_id in (option_task_ids[variant][1],)
        )

    optimizer_task_id = f"task_{len(tasks):06d}"
    optimizer_inputs = tuple(item.parameter_object_id for item in gradients) + tuple(
        item.gradient_object_id for item in gradients
    )
    tasks.append(
        TaskSpec(
            optimizer_task_id,
            ResourceSpec(device_id, ResourceKind.COMPUTE),
            profile_id(optimizer.recurrent),
            dependencies=tuple(all_backward_ids),
            inputs=optimizer_inputs,
            mutations=tuple(
                MutationSpec(item.parameter_object_id) for item in gradients
            ),
            phase="optimizer",
        )
    )
    entrypoints.append(
        TrainingTaskEntrypoint(
            optimizer_task_id,
            "optimizer",
            None,
            None,
            optimizer.recurrent,
            (),
            (),
        )
    )

    alias_groups = inventory.alias_groups()
    objects = inventory.objects()
    parameter_aliases = {
        next(
            item.alias_group_id
            for item in objects
            if item.object_id == binding.parameter_object_id
        )
        for binding in gradients
    }
    produced_objects = {object_id for task in tasks for object_id in task.outputs}
    initial_input_objects = {
        object_id
        for task in tasks
        for object_id in task.inputs
        if object_id not in produced_objects
    }
    input_aliases = {
        item.alias_group_id
        for item in objects
        if item.object_id in initial_input_objects
        and item.alias_group_id not in parameter_aliases
    }
    public_aliases = {
        next(item.alias_group_id for item in objects if item.object_id == object_id)
        for values in public_objects_by_position.values()
        for object_id in values
    }
    initial_residency = tuple(
        ResidencySpec(item.alias_group_id, MemoryLocation.HOST)
        for item in alias_groups
        if item.alias_group_id in parameter_aliases | input_aliases
    )
    final_residency = tuple(
        ResidencySpec(
            item.alias_group_id,
            MemoryLocation.DEVICE
            if item.alias_group_id in public_aliases
            else MemoryLocation.HOST,
        )
        for item in alias_groups
        if item.alias_group_id in parameter_aliases | input_aliases | public_aliases
    )
    program = Program(
        devices=(DeviceSpec(device_id, "process_0", "cuda", device_ordinal),),
        alias_groups=alias_groups,
        objects=objects,
        profiles=tuple(profiles),
        tasks=tuple(tasks),
        recomputation_groups=tuple(groups),
    )
    return LoweredTrainingProgram(
        program,
        initial_residency,
        final_residency,
        registrations,
        root_slots,
        tuple(entrypoints),
        gradients,
        tuple(fixed_tensors.values()),
        optimizer_task_id,
    )


def _register_model(
    model: nn.Module, inventory: _TensorInventory
) -> tuple[tuple[RegistrationBinding, ...], dict[tuple[int, int], str]]:
    registrations: list[RegistrationBinding] = []
    parameter_objects: dict[tuple[int, int], str] = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        object_id = inventory.add(
            parameter,
            role=ObjectRole.PARAMETER,
            persistence=Persistence.CHECKPOINT,
            retain_host_backing=True,
        )
        registrations.append(RegistrationBinding(name, object_id, True))
        parameter_objects[
            (parameter.untyped_storage()._cdata, int(parameter.storage_offset()))
        ] = object_id
    for name, buffer in model.named_buffers(remove_duplicate=False):
        object_id = inventory.add(
            buffer,
            role=ObjectRole.BUFFER,
            persistence=Persistence.CHECKPOINT,
            retain_host_backing=True,
        )
        registrations.append(RegistrationBinding(name, object_id, False))
    return tuple(registrations), parameter_objects


def _register_microbatch_inputs(
    captures: tuple[TrainingCapture, ...], inventory: _TensorInventory
) -> tuple[tuple[tuple[TensorSlot, ...], ...], set[str]]:
    positions: list[tuple[TensorSlot, ...]] = []
    initial: set[str] = set()
    for capture in captures:
        slots: list[TensorSlot] = []
        for index, (spec, value) in enumerate(
            zip(
                capture.exported.exported_program.graph_signature.input_specs,
                capture.exported.flat_inputs,
                strict=True,
            )
        ):
            if spec.kind is not InputKind.USER_INPUT or not isinstance(
                value, torch.Tensor
            ):
                continue
            object_id = inventory.add(
                value,
                role=ObjectRole.INPUT,
                persistence=Persistence.STEP,
                retain_host_backing=True,
            )
            slots.append(TensorSlot(index, object_id))
            initial.add(object_id)
        positions.append(tuple(slots))
    return tuple(positions), initial


def _register_gradients(
    model: nn.Module,
    inventory: _TensorInventory,
    parameter_objects: dict[tuple[int, int], str],
) -> tuple[GradientBinding, ...]:
    results: list[GradientBinding] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        parameter_id = parameter_objects[
            (parameter.untyped_storage()._cdata, int(parameter.storage_offset()))
        ]
        gradient = torch.empty_like(parameter, memory_format=torch.preserve_format)
        gradient_id = inventory.add(
            gradient, role=ObjectRole.GRADIENT, persistence=Persistence.STEP
        )
        results.append(GradientBinding(name, parameter_id, gradient_id))
    return tuple(results)


def _execute_pair_examples(pair: AotGraphPair) -> _PairValues:
    forward_raw = pair.forward.graph_module(*pair.forward.example_arguments)
    forward_values, _ = tree_flatten(forward_raw)
    public_count = pair.forward.output_count - pair.saved_value_count
    residuals = forward_values[public_count:]
    loss = forward_values[0]
    if not isinstance(loss, torch.Tensor):
        raise CaptureError("captured objective loss is not a tensor")
    backward_raw = pair.backward.graph_module(*residuals, torch.ones_like(loss))
    backward_values, _ = tree_flatten(backward_raw)
    return _PairValues(tuple(forward_values), tuple(backward_values))


def _tensor_slots(
    values: tuple[object, ...], inventory: _TensorInventory
) -> tuple[TensorSlot, ...]:
    return tuple(
        TensorSlot(
            index,
            inventory.add(value, role=ObjectRole.INPUT, persistence=Persistence.STEP),
        )
        for index, value in enumerate(values)
        if isinstance(value, torch.Tensor)
    )


def _forward_outputs(
    position: int,
    pair: AotGraphPair,
    capture: TrainingCapture,
    values: tuple[object, ...],
    inventory: _TensorInventory,
    public_by_position: dict[int, tuple[str, ...]],
) -> tuple[tuple[TensorSlot, ...], tuple[str, ...], tuple[str, ...]]:
    public_count = 1 + len(capture.objective_schema.tensor_metric_positions)
    canonical_public = public_by_position.get(position)
    slots: list[TensorSlot] = []
    public_ids: list[str] = []
    residual_aliases: list[str] = []
    for index, value in enumerate(values):
        if not isinstance(value, torch.Tensor):
            raise CaptureError("AOT forward returned a non-tensor leaf")
        if index < public_count and canonical_public is not None:
            object_id = canonical_public[index]
        else:
            object_id = inventory.add(
                value,
                role=ObjectRole.OUTPUT
                if index < public_count
                else ObjectRole.ACTIVATION,
                persistence=Persistence.STEP,
            )
        slots.append(TensorSlot(index, object_id))
        if index < public_count:
            public_ids.append(object_id)
        else:
            residual_aliases.append(inventory.alias_id(object_id))
    if pair.saved_value_count != len(values) - public_count:
        raise CaptureError("AOT residual count changed during training lowering")
    if canonical_public is None:
        public_by_position[position] = tuple(public_ids)
    return tuple(slots), tuple(public_ids), tuple(dict.fromkeys(residual_aliases))


def _backward_inputs(
    pair: AotGraphPair,
    values: _PairValues,
    inventory: _TensorInventory,
    forward_slots: tuple[TensorSlot, ...],
    capture: TrainingCapture,
) -> tuple[tuple[TensorSlot, ...], FixedTensorBinding]:
    public_count = 1 + len(capture.objective_schema.tensor_metric_positions)
    residual_slots = tuple(
        TensorSlot(index, slot.object_id)
        for index, slot in enumerate(forward_slots[public_count:])
    )
    tangent = pair.backward.example_arguments[-1]
    if not isinstance(tangent, torch.Tensor):
        raise CaptureError("AOT backward tangent is not a tensor")
    tangent_id = inventory.add(
        tangent,
        role=ObjectRole.INPUT,
        persistence=Persistence.STEP,
        retain_host_backing=True,
    )
    return (
        (*residual_slots, TensorSlot(len(residual_slots), tangent_id)),
        FixedTensorBinding(tangent_id, tangent),
    )


def _gradient_outputs(
    pair: AotGraphPair,
    values: tuple[object, ...],
    inventory: _TensorInventory,
    parameter_objects: dict[tuple[int, int], str],
    gradient_by_parameter: dict[str, str],
) -> tuple[TensorSlot, ...]:
    tensor_primals = tuple(
        value
        for value in pair.forward.example_arguments
        if isinstance(value, torch.Tensor)
    )
    positions = pair.forward.tensor_argument_positions
    if len(positions) != len(tensor_primals) or any(
        position >= len(values) for position in positions
    ):
        raise CaptureError("AOT backward gradient/primal positions differ")
    results: list[TensorSlot] = []
    for index, primal in zip(positions, tensor_primals, strict=True):
        gradient = values[index]
        parameter_id = parameter_objects.get(
            (primal.untyped_storage()._cdata, int(primal.storage_offset()))
        )
        if parameter_id is None:
            continue
        if not isinstance(gradient, torch.Tensor):
            raise CaptureError("trainable parameter produced no AOT gradient")
        gradient_id = gradient_by_parameter[parameter_id]
        expected = next(
            item for item in inventory.objects() if item.object_id == gradient_id
        )
        if gradient.untyped_storage().nbytes() < expected.size_bytes:
            raise CaptureError("parameter gradient has an invalid storage extent")
        results.append(TensorSlot(index, gradient_id))
    return tuple(results)


def _unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


__all__ = [
    "FixedTensorBinding",
    "GradientBinding",
    "LoweredTrainingProgram",
    "TrainingStorageLayout",
    "TrainingTaskEntrypoint",
    "lower_training_program",
    "lower_training_storage_layout",
]
