"""Lower explicit objective graph pairs into one accumulated training Program."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
from torch.export.graph_signature import InputKind
from torch.utils._pytree import tree_flatten

from shadowspill.ir import (
    DeviceSpec,
    MemoryLocation,
    MutationSpec,
    ObjectRole,
    ObjectSpec,
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
    OptimizerTask,
    OptimizerTaskArtifact,
    OptimizerTensorRole,
)
from .partition import PartitionedTrainingCapture, TrainingStage
from .profiling import TaskMeasurement


@dataclass(frozen=True, slots=True)
class GradientBinding:
    parameter_name: str
    parameter_object_id: str
    gradient_object_id: str


@dataclass(frozen=True, slots=True)
class OptimizerObjectBinding:
    name: str
    object_id: str
    role: OptimizerTensorRole
    mutable: bool
    created_on_first_step: bool


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
    artifact: GraphArtifact | OptimizerTaskArtifact | None
    input_slots: tuple[TensorSlot, ...]
    output_slots: tuple[TensorSlot, ...]
    gradient_output_slots: tuple[TensorSlot, ...] = ()
    public_output_count: int = 0
    optimizer_binding_names: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class LoweredTrainingProgram:
    program: Program
    initial_residency: tuple[ResidencySpec, ...]
    final_residency: tuple[ResidencySpec, ...]
    registrations: tuple[RegistrationBinding, ...]
    root_input_slots: tuple[tuple[TensorSlot, ...], ...]
    entrypoints: tuple[TrainingTaskEntrypoint, ...]
    gradients: tuple[GradientBinding, ...]
    optimizer_objects: tuple[OptimizerObjectBinding, ...]
    fixed_tensors: tuple[FixedTensorBinding, ...]
    optimizer_task_ids: tuple[str, ...]

    @property
    def optimizer_task_id(self) -> str:
        return self.optimizer_task_ids[-1]


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


@dataclass(frozen=True, slots=True)
class _PreparedStageVariant:
    stage: TrainingStage
    pair: AotGraphPair
    forward_inputs: tuple[TensorSlot, ...]
    forward_outputs: tuple[TensorSlot, ...]
    backward_inputs: tuple[TensorSlot, ...]
    contributions: tuple[TensorSlot, ...]
    residual_aliases: tuple[str, ...]
    public_output_count: int


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


def lower_partitioned_training_program(
    model: nn.Module,
    captures: tuple[PartitionedTrainingCapture, ...],
    measurements: dict[str, TaskMeasurement],
    optimizer: OptimizerCapture,
    *,
    device_ordinal: int = 0,
    optimizer_phase: Literal["initial", "recurrent"] = "recurrent",
) -> LoweredTrainingProgram:
    """Compose stage-local graph pairs into one accumulated training program."""

    if not captures:
        raise CaptureError("partitioned training lowering requires a microbatch")
    if optimizer.recurrent is None:
        raise CaptureError("partitioned training requires a bounded optimizer task")
    if optimizer_phase not in {"initial", "recurrent"}:
        raise CaptureError(f"unknown optimizer phase {optimizer_phase!r}")
    device_id = f"cuda_{device_ordinal}"
    inventory = _TensorInventory(device_id=device_id)
    registrations, parameter_objects = _register_model(model, inventory)
    base_captures = tuple(item.training for item in captures)
    root_slots, _initial_inputs = _register_microbatch_inputs(base_captures, inventory)
    gradients = _register_gradients(model, inventory, parameter_objects)
    gradient_by_parameter = {
        item.parameter_object_id: item.gradient_object_id for item in gradients
    }
    optimizer_objects = _register_optimizer_objects(optimizer, inventory, gradients)

    profile_by_digest: dict[str, str] = {}
    profiles: list[TaskProfile] = []

    def profile_id(
        artifact: GraphArtifact | OptimizerTaskArtifact, extra_workspace: int = 0
    ) -> str:
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

    parameter_ids = set(parameter_objects.values())
    boundary_ids: list[list[tuple[str, ...]]] = []
    cotangent_by_activation: dict[tuple[int, str], str] = {}
    fixed_tensors: dict[str, FixedTensorBinding] = {}
    public_objects_by_position: dict[int, tuple[str, ...]] = {}

    for position, capture in enumerate(captures):
        position_boundaries: list[tuple[str, ...]] = []
        for stage_index, stage in enumerate(capture.stages):
            leaves, _ = tree_flatten(stage.example.output)
            terminal = stage_index == len(capture.stages) - 1
            public_count = (
                1 + len(capture.training.objective_schema.tensor_metric_positions)
                if terminal
                else 0
            )
            if terminal and len(leaves) != public_count:
                raise CaptureError(
                    "terminal stage output differs from objective schema"
                )
            ids: list[str] = []
            for index, value in enumerate(leaves):
                if not isinstance(value, torch.Tensor):
                    raise CaptureError("training stage output became non-tensor")
                ids.append(
                    inventory.add(
                        value,
                        role=(
                            ObjectRole.OUTPUT
                            if terminal and index < public_count
                            else ObjectRole.ACTIVATION
                        ),
                        persistence=Persistence.STEP,
                    )
                )
            position_boundaries.append(tuple(ids))
            if terminal:
                public_objects_by_position[position] = tuple(ids[:public_count])

            save_pair = stage.save_pair
            tangent_values = save_pair.backward.example_arguments[
                save_pair.saved_value_count :
            ]
            if len(tangent_values) != len(stage.differentiable_output_indices):
                raise CaptureError("stage tangent ABI differs from selected roots")
            for output_index, tangent in zip(
                stage.differentiable_output_indices,
                tangent_values,
                strict=True,
            ):
                if not isinstance(tangent, torch.Tensor):
                    raise CaptureError("stage tangent became non-tensor")
                if terminal:
                    tangent_id = inventory.add(
                        tangent,
                        role=ObjectRole.INPUT,
                        persistence=Persistence.STEP,
                        retain_host_backing=True,
                    )
                    fixed_tensors.setdefault(
                        tangent_id, FixedTensorBinding(tangent_id, tangent)
                    )
                else:
                    activation_id = ids[output_index]
                    tangent_id = inventory.add(
                        torch.empty_like(tangent),
                        role=ObjectRole.GRADIENT,
                        persistence=Persistence.STEP,
                    )
                    cotangent_by_activation[(position, activation_id)] = tangent_id
        boundary_ids.append(position_boundaries)

    prepared: list[list[dict[str, _PreparedStageVariant]]] = []
    for position, capture in enumerate(captures):
        position_variants: list[dict[str, _PreparedStageVariant]] = []
        for stage_index, stage in enumerate(capture.stages):
            variants: dict[str, _PreparedStageVariant] = {}
            terminal = stage_index == len(capture.stages) - 1
            public_count = (
                1 + len(capture.training.objective_schema.tensor_metric_positions)
                if terminal
                else 0
            )
            canonical_outputs = boundary_ids[position][stage_index]
            for variant, pair in (
                ("save", stage.save_pair),
                ("recompute", stage.recompute_pair),
            ):
                values = _execute_stage_pair_examples(pair)
                forward_inputs = _tensor_slots(
                    pair.forward.example_arguments, inventory
                )
                forward_outputs, residual_aliases = _stage_forward_outputs(
                    pair,
                    values.forward_outputs,
                    canonical_outputs,
                    inventory,
                )
                backward_inputs = _stage_backward_inputs(
                    position,
                    stage,
                    pair,
                    forward_outputs,
                    canonical_outputs,
                    cotangent_by_activation,
                    fixed_tensors,
                    inventory,
                    terminal=terminal,
                )
                contributions = _stage_backward_contributions(
                    position,
                    pair,
                    values.backward_outputs,
                    forward_inputs,
                    backward_inputs,
                    parameter_ids,
                    gradient_by_parameter,
                    cotangent_by_activation,
                    inventory,
                )
                variants[variant] = _PreparedStageVariant(
                    stage,
                    pair,
                    forward_inputs,
                    forward_outputs,
                    backward_inputs,
                    contributions,
                    residual_aliases,
                    public_count,
                )
            position_variants.append(variants)
        prepared.append(position_variants)

    tasks: list[TaskSpec] = []
    entrypoints: list[TrainingTaskEntrypoint] = []
    forward_ids: dict[tuple[int, int, str], str] = {}
    backward_ids: dict[tuple[int, int, str], str] = {}
    completion_ids: tuple[str, ...] = ()
    all_backward_ids: list[str] = []
    initial_writers: dict[str, tuple[str, ...]] = {}
    latest_contributors: dict[str, tuple[str, ...]] = {}
    object_producers: dict[str, list[str]] = {}

    for position, position_variants in enumerate(prepared):
        previous_forward_ids = completion_ids
        for stage_index, variants in enumerate(position_variants):
            current_ids: list[str] = []
            for variant in ("save", "recompute"):
                item = variants[variant]
                task_id = f"task_{len(tasks):06d}"
                forward_ids[(position, stage_index, variant)] = task_id
                input_aliases = {
                    inventory.alias_id(slot.object_id) for slot in item.forward_inputs
                }
                dependencies = list(previous_forward_ids)
                for slot in item.forward_inputs:
                    dependencies.extend(object_producers.get(slot.object_id, ()))
                task = TaskSpec(
                    task_id,
                    ResourceSpec(device_id, ResourceKind.COMPUTE),
                    profile_id(item.pair.forward),
                    dependencies=_unique(dependencies),
                    inputs=_unique(slot.object_id for slot in item.forward_inputs),
                    outputs=_unique(
                        slot.object_id
                        for slot in item.forward_outputs
                        if inventory.alias_id(slot.object_id) not in input_aliases
                    ),
                    phase="forward",
                )
                tasks.append(task)
                for object_id in task.outputs:
                    object_producers.setdefault(object_id, []).append(task_id)
                entrypoints.append(
                    TrainingTaskEntrypoint(
                        task_id,
                        "forward",
                        position,
                        variant,
                        item.pair.forward,
                        item.forward_inputs,
                        item.forward_outputs,
                        public_output_count=item.public_output_count,
                    )
                )
                current_ids.append(task_id)
            previous_forward_ids = tuple(current_ids)

        downstream_ids: tuple[str, ...] = ()
        for stage_index in reversed(range(len(position_variants))):
            variants = position_variants[stage_index]
            stage_task_ids = tuple(
                f"task_{len(tasks) + offset:06d}" for offset in range(2)
            )
            first_destinations = {
                slot.object_id
                for item in variants.values()
                for slot in item.contributions
                if slot.object_id not in initial_writers
            }
            for variant, task_id in zip(
                ("save", "recompute"), stage_task_ids, strict=True
            ):
                item = variants[variant]
                backward_ids[(position, stage_index, variant)] = task_id
                destinations = _unique(slot.object_id for slot in item.contributions)
                mutated = tuple(
                    object_id
                    for object_id in destinations
                    if object_id not in first_destinations
                )
                dependencies = list(
                    dict.fromkeys(
                        (
                            forward_ids[(position, stage_index, variant)],
                            *downstream_ids,
                        )
                    )
                )
                for slot in item.backward_inputs:
                    object_id = slot.object_id
                    dependencies.extend(object_producers.get(object_id, ()))
                    dependencies.extend(initial_writers.get(object_id, ()))
                    dependencies.extend(latest_contributors.get(object_id, ()))
                for object_id in mutated:
                    dependencies.extend(initial_writers[object_id])
                    dependencies.extend(latest_contributors[object_id])
                mutation_bytes = sum(
                    next(
                        value.size_bytes
                        for value in inventory.objects()
                        if value.object_id == object_id
                    )
                    for object_id in mutated
                )
                inputs = [slot.object_id for slot in item.backward_inputs]
                inputs.extend(mutated)
                input_aliases = {
                    inventory.alias_id(object_id) for object_id in inputs
                }
                task = TaskSpec(
                    task_id,
                    ResourceSpec(device_id, ResourceKind.COMPUTE),
                    profile_id(item.pair.backward, mutation_bytes),
                    dependencies=_unique(dependencies),
                    inputs=_unique(inputs),
                    outputs=tuple(
                        object_id
                        for object_id in destinations
                        if object_id in first_destinations
                        and inventory.alias_id(object_id) not in input_aliases
                    ),
                    mutations=tuple(MutationSpec(object_id) for object_id in mutated),
                    phase="backward",
                )
                tasks.append(task)
                for object_id in task.outputs:
                    object_producers.setdefault(object_id, []).append(task_id)
                entrypoints.append(
                    TrainingTaskEntrypoint(
                        task_id,
                        "backward",
                        position,
                        variant,
                        item.pair.backward,
                        item.backward_inputs,
                        (),
                        gradient_output_slots=item.contributions,
                    )
                )
                all_backward_ids.append(task_id)
            contribution_destinations = _unique(
                slot.object_id
                for item in variants.values()
                for slot in item.contributions
            )
            for object_id in contribution_destinations:
                initial_writers.setdefault(object_id, stage_task_ids)
                latest_contributors[object_id] = stage_task_ids
            downstream_ids = stage_task_ids
        completion_ids = downstream_ids

    groups = tuple(
        RecomputationGroup(
            f"recompute_{position:04d}_{stage_index:04d}",
            tuple(
                RecomputationOption(
                    variant,
                    (
                        forward_ids[(position, stage_index, variant)],
                        backward_ids[(position, stage_index, variant)],
                    ),
                    prepared[position][stage_index][variant].residual_aliases,
                )
                for variant in ("save", "recompute")
            ),
        )
        for position, position_variants in enumerate(prepared)
        for stage_index in range(len(position_variants))
    )

    optimizer_task_ids = _append_optimizer_tasks(
        tasks,
        entrypoints,
        optimizer,
        optimizer_objects,
        gradients,
        optimizer_phase=optimizer_phase,
        dependencies=tuple(all_backward_ids),
        device_id=device_id,
        profile_id=profile_id,
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
    input_aliases = _external_input_aliases(objects, tuple(tasks), parameter_aliases)
    public_aliases = {
        next(item.alias_group_id for item in objects if item.object_id == object_id)
        for values in public_objects_by_position.values()
        for object_id in values
    }
    optimizer_aliases = {
        next(
            item.alias_group_id
            for item in objects
            if item.object_id == binding.object_id
        )
        for binding in optimizer_objects
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
        if item.alias_group_id
        in parameter_aliases | input_aliases | public_aliases | optimizer_aliases
    )
    program = Program(
        devices=(DeviceSpec(device_id, "process_0", "cuda", device_ordinal),),
        alias_groups=alias_groups,
        objects=objects,
        profiles=tuple(profiles),
        tasks=tuple(tasks),
        recomputation_groups=groups,
    )
    return LoweredTrainingProgram(
        program,
        initial_residency,
        final_residency,
        registrations,
        root_slots,
        tuple(entrypoints),
        gradients,
        optimizer_objects,
        tuple(fixed_tensors.values()),
        optimizer_task_ids,
    )


def lower_training_program(
    model: nn.Module,
    captures: tuple[TrainingCapture, ...],
    measurements: dict[str, TaskMeasurement],
    optimizer: OptimizerCapture,
    *,
    device_ordinal: int = 0,
    optimizer_phase: Literal["initial", "recurrent"] = "recurrent",
) -> LoweredTrainingProgram:
    """Compose microbatch graph-pair alternatives and one optimizer mutation."""

    if not captures:
        raise CaptureError("training lowering requires at least one microbatch")
    if optimizer.recurrent is None:
        raise CaptureError("training lowering requires a bounded optimizer task")
    if optimizer_phase not in {"initial", "recurrent"}:
        raise CaptureError(f"unknown optimizer phase {optimizer_phase!r}")
    device_id = f"cuda_{device_ordinal}"
    inventory = _TensorInventory(device_id=device_id)
    registrations, parameter_objects = _register_model(model, inventory)
    root_slots, _initial_inputs = _register_microbatch_inputs(captures, inventory)
    gradients = _register_gradients(model, inventory, parameter_objects)
    gradient_by_parameter = {
        item.parameter_object_id: item.gradient_object_id for item in gradients
    }
    optimizer_objects = _register_optimizer_objects(optimizer, inventory, gradients)

    profile_by_digest: dict[str, str] = {}
    profiles: list[TaskProfile] = []

    def profile_id(
        artifact: GraphArtifact | OptimizerTaskArtifact, extra_workspace: int = 0
    ) -> str:
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

    optimizer_task_ids = _append_optimizer_tasks(
        tasks,
        entrypoints,
        optimizer,
        optimizer_objects,
        gradients,
        optimizer_phase=optimizer_phase,
        dependencies=tuple(all_backward_ids),
        device_id=device_id,
        profile_id=profile_id,
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
    input_aliases = _external_input_aliases(objects, tuple(tasks), parameter_aliases)
    public_aliases = {
        next(item.alias_group_id for item in objects if item.object_id == object_id)
        for values in public_objects_by_position.values()
        for object_id in values
    }
    optimizer_aliases = {
        next(
            item.alias_group_id
            for item in objects
            if item.object_id == binding.object_id
        )
        for binding in optimizer_objects
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
        if item.alias_group_id
        in parameter_aliases | input_aliases | public_aliases | optimizer_aliases
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
        optimizer_objects,
        tuple(fixed_tensors.values()),
        optimizer_task_ids,
    )


def _append_optimizer_tasks(
    tasks: list[TaskSpec],
    entrypoints: list[TrainingTaskEntrypoint],
    optimizer: OptimizerCapture,
    optimizer_objects: tuple[OptimizerObjectBinding, ...],
    gradients: tuple[GradientBinding, ...],
    *,
    optimizer_phase: Literal["initial", "recurrent"],
    dependencies: tuple[str, ...],
    device_id: str,
    profile_id: Callable[..., str],
) -> tuple[str, ...]:
    """Append dependency-closed optimizer components in semantic order."""

    if optimizer.recurrent is None:
        raise CaptureError("optimizer has no recurrent artifact")
    object_by_name = {
        **{item.parameter_name: item.parameter_object_id for item in gradients},
        **{
            f"gradient.{item.parameter_name}": item.gradient_object_id
            for item in gradients
        },
        **{item.name: item.object_id for item in optimizer_objects},
    }
    object_binding = {item.name: item for item in optimizer_objects}
    has_lazy_outputs = any(item.created_on_first_step for item in optimizer_objects)
    components = (
        (
            OptimizerTask(
                optimizer.recurrent,
                tuple(binding.name for binding in optimizer.bindings),
                optimizer.mutation_names,
            ),
        )
        if optimizer_phase == "initial" and has_lazy_outputs
        else optimizer.recurrent_tasks
    )
    if not components:
        raise CaptureError("optimizer has no executable task components")

    result: list[str] = []
    preceding = dependencies
    for component in components:
        task_id = f"task_{len(tasks):06d}"
        component_objects = tuple(
            object_by_name[name]
            for name in component.binding_names
            if name in object_by_name
        )
        outputs = tuple(
            object_by_name[name]
            for name in component.mutation_names
            if name in object_binding
            and optimizer_phase == "initial"
            and object_binding[name].created_on_first_step
        )
        mutations = tuple(
            MutationSpec(object_by_name[name])
            for name in component.mutation_names
            if name in object_by_name and object_by_name[name] not in outputs
        )
        tasks.append(
            TaskSpec(
                task_id,
                ResourceSpec(device_id, ResourceKind.COMPUTE),
                profile_id(component.artifact),
                dependencies=preceding,
                inputs=tuple(
                    object_id
                    for object_id in component_objects
                    if object_id not in outputs
                ),
                outputs=outputs,
                mutations=mutations,
                phase="optimizer",
            )
        )
        entrypoints.append(
            TrainingTaskEntrypoint(
                task_id,
                "optimizer",
                None,
                None,
                component.artifact,
                (),
                (),
                optimizer_binding_names=component.binding_names,
            )
        )
        result.append(task_id)
        preceding = _unique((*dependencies, task_id))
    return tuple(result)


def _register_model(
    model: nn.Module, inventory: _TensorInventory
) -> tuple[tuple[RegistrationBinding, ...], dict[tuple[int, int], str]]:
    registrations: list[RegistrationBinding] = []
    parameter_objects: dict[tuple[int, int], str] = {}
    checkpoint_names = set(model.state_dict())
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
            persistence=(
                Persistence.CHECKPOINT if name in checkpoint_names else Persistence.RUN
            ),
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


def _register_optimizer_objects(
    optimizer: OptimizerCapture,
    inventory: _TensorInventory,
    gradients: tuple[GradientBinding, ...],
) -> tuple[OptimizerObjectBinding, ...]:
    parameter_names = {item.parameter_name for item in gradients}
    gradient_names = {f"gradient.{item.parameter_name}" for item in gradients}
    created = (
        set()
        if optimizer.initialized_state_dict is not None
        else set(optimizer.created_state_names)
    )
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
            retain_host_backing=True,
        )
        results.append(
            OptimizerObjectBinding(
                binding.name,
                object_id,
                binding.role,
                binding.mutable,
                binding.name in created,
            )
        )
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


def _execute_stage_pair_examples(pair: AotGraphPair) -> _PairValues:
    forward_raw = pair.forward.graph_module(*pair.forward.example_arguments)
    forward_values, _ = tree_flatten(forward_raw)
    residuals = forward_values[pair.forward.output_count - pair.saved_value_count :]
    tangents = pair.backward.example_arguments[pair.saved_value_count :]
    backward_raw = pair.backward.graph_module(*residuals, *tangents)
    backward_values, _ = tree_flatten(backward_raw)
    return _PairValues(tuple(forward_values), tuple(backward_values))


def _stage_forward_outputs(
    pair: AotGraphPair,
    values: tuple[object, ...],
    canonical_outputs: tuple[str, ...],
    inventory: _TensorInventory,
) -> tuple[tuple[TensorSlot, ...], tuple[str, ...]]:
    public_count = pair.forward.output_count - pair.saved_value_count
    if public_count != len(canonical_outputs):
        raise CaptureError("stage boundary output count changed across AOT capture")
    slots: list[TensorSlot] = []
    residual_aliases: list[str] = []
    for index, value in enumerate(values):
        if not isinstance(value, torch.Tensor):
            raise CaptureError("AOT stage forward returned a non-tensor leaf")
        if index < public_count:
            object_id = canonical_outputs[index]
            inventory.associate_storage(value, object_id)
        else:
            object_id = inventory.add(
                value,
                role=ObjectRole.ACTIVATION,
                persistence=Persistence.STEP,
            )
        slots.append(TensorSlot(index, object_id))
        if index >= public_count:
            residual_aliases.append(inventory.alias_id(object_id))
    return tuple(slots), tuple(dict.fromkeys(residual_aliases))


def _stage_backward_inputs(
    position: int,
    stage: TrainingStage,
    pair: AotGraphPair,
    forward_outputs: tuple[TensorSlot, ...],
    canonical_outputs: tuple[str, ...],
    cotangent_by_activation: dict[tuple[int, str], str],
    fixed_tensors: dict[str, FixedTensorBinding],
    inventory: _TensorInventory,
    *,
    terminal: bool,
) -> tuple[TensorSlot, ...]:
    public_count = pair.forward.output_count - pair.saved_value_count
    residuals = tuple(
        TensorSlot(index, slot.object_id)
        for index, slot in enumerate(forward_outputs[public_count:])
    )
    tangent_values = pair.backward.example_arguments[pair.saved_value_count :]
    if len(tangent_values) != len(stage.differentiable_output_indices):
        raise CaptureError("stage backward tangent arity changed")
    tangents: list[TensorSlot] = []
    for ordinal, (output_index, value) in enumerate(
        zip(stage.differentiable_output_indices, tangent_values, strict=True)
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
                    retain_host_backing=True,
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
    values: tuple[object, ...],
    forward_inputs: tuple[TensorSlot, ...],
    backward_inputs: tuple[TensorSlot, ...],
    parameter_ids: set[str],
    gradient_by_parameter: dict[str, str],
    cotangent_by_activation: dict[tuple[int, str], str],
    inventory: _TensorInventory,
) -> tuple[TensorSlot, ...]:
    input_by_position = {slot.leaf_index: slot.object_id for slot in forward_inputs}
    backward_object_by_key = {
        inventory.key(value): slot.object_id
        for slot in backward_inputs
        if slot.leaf_index < len(pair.backward.example_arguments)
        and isinstance(
            value := pair.backward.example_arguments[slot.leaf_index], torch.Tensor
        )
    }
    results: list[TensorSlot] = []
    for output_index in pair.forward.tensor_argument_positions:
        object_id = input_by_position.get(output_index)
        if object_id is None or output_index >= len(values):
            continue
        value = values[output_index]
        if not isinstance(value, torch.Tensor):
            continue
        destination: str | None
        if object_id in parameter_ids:
            destination = gradient_by_parameter[object_id]
        else:
            destination = cotangent_by_activation.get((position, object_id))
            if destination is None:
                continue
            source = backward_object_by_key.get(inventory.key(value))
            if source is not None:
                inventory.merge_object_aliases(destination, source)
        results.append(TensorSlot(output_index, destination))
    return tuple(results)


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
            inventory.associate_storage(value, object_id)
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


def _external_input_aliases(
    objects: tuple[ObjectSpec, ...],
    tasks: tuple[TaskSpec, ...],
    parameter_aliases: set[str],
) -> set[str]:
    """Return input storage bundles that no task in the program produces."""

    alias_by_object = {item.object_id: item.alias_group_id for item in objects}
    produced_aliases = {
        alias_by_object[object_id] for task in tasks for object_id in task.outputs
    }
    return {
        alias_by_object[object_id]
        for task in tasks
        for object_id in task.inputs
        if alias_by_object[object_id] not in produced_aliases
        and alias_by_object[object_id] not in parameter_aliases
    }


def _unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


__all__ = [
    "FixedTensorBinding",
    "GradientBinding",
    "LoweredTrainingProgram",
    "OptimizerObjectBinding",
    "TrainingStorageLayout",
    "TrainingTaskEntrypoint",
    "lower_partitioned_training_program",
    "lower_training_program",
    "lower_training_storage_layout",
]
