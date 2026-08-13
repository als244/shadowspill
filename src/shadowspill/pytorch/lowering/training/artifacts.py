"""Immutable bindings exchanged by training-lowering phases."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from shadowspill.ir import Program, RecomputationGroup, ResidencySpec, TaskSpec

from ...capture import AotGraphPair, GraphArtifact
from ...optimizer import OptimizerTaskArtifact, OptimizerTensorRole
from ...partition import TrainingStage
from ..catalog import ObjectCatalog, RegistrationBinding, TensorSlot
from ..task_binding import TaskStorageHandoff


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
    public_output_leaves: tuple[int, ...] = ()
    optimizer_binding_names: tuple[str, ...] = ()
    stage_index: int | None = None
    replacement_output_leaves: tuple[int, ...] = ()
    storage_handoffs: tuple[TaskStorageHandoff, ...] = ()


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
class PreparedStageVariant:
    stage: TrainingStage
    pair: AotGraphPair
    forward_inputs: tuple[TensorSlot, ...]
    forward_outputs: tuple[TensorSlot, ...]
    backward_inputs: tuple[TensorSlot, ...]
    contributions: tuple[TensorSlot, ...]
    residual_object_ids: tuple[str, ...]
    public_output_leaves: tuple[int, ...]
    mutation_object_ids: tuple[str, ...]
    replacement_output_leaves: tuple[int, ...]
    forward_storage_handoffs: tuple[TaskStorageHandoff, ...]
    backward_storage_handoffs: tuple[TaskStorageHandoff, ...]


@dataclass(frozen=True, slots=True)
class TrainingObjects:
    catalog: ObjectCatalog
    registrations: tuple[RegistrationBinding, ...]
    root_slots: tuple[tuple[TensorSlot, ...], ...]
    parameter_objects: dict[tuple[int, int], str]
    gradients: tuple[GradientBinding, ...]
    gradient_by_parameter: dict[str, str]
    optimizer_objects: tuple[OptimizerObjectBinding, ...]


@dataclass(frozen=True, slots=True)
class TrainingBoundaries:
    object_ids: tuple[tuple[tuple[str, ...], ...], ...]
    root_objects: tuple[dict[int, str], ...]
    cotangents: dict[tuple[int, str], str]
    fixed_tensors: dict[str, FixedTensorBinding]
    public_outputs: dict[int, tuple[str, ...]]


@dataclass(frozen=True, slots=True)
class TrainingTaskGraph:
    tasks: tuple[TaskSpec, ...]
    entrypoints: tuple[TrainingTaskEntrypoint, ...]
    recomputation_groups: tuple[RecomputationGroup, ...]
    optimizer_task_ids: tuple[str, ...]
