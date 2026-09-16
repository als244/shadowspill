"""Immutable bindings exchanged by training-lowering phases."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from shadowspill.ir import (
    ResidencySpec,
    ShadowSpillProgram,
    TaskAlternativeGroup,
    TaskSpec,
)
from shadowspill.pytorch.capture.artifacts import (
    AotGraphPair,
    GraphArtifact,
)
from shadowspill.pytorch.optimizer import OptimizerTaskArtifact, OptimizerTensorRole
from shadowspill.task.entrypoints import TaskEntrypoint
from shadowspill.task.slots import ObjectSlot, TaskStorageHandoff

from ...graph_pairs import DifferentiatedStage
from ..catalog import ObjectCatalog, RegistrationBinding


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
class LoweredTrainingProgram:
    program: ShadowSpillProgram
    initial_residency: tuple[ResidencySpec, ...]
    final_residency: tuple[ResidencySpec, ...]
    registrations: tuple[RegistrationBinding, ...]
    root_input_slots: tuple[tuple[ObjectSlot, ...], ...]
    entrypoints: tuple[TaskEntrypoint, ...]
    #: What to call for each task. The entrypoint is neutral; this is not.
    executables: Mapping[str, GraphArtifact | OptimizerTaskArtifact | None]
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

    program: ShadowSpillProgram
    registrations: tuple[RegistrationBinding, ...]
    root_input_slots: tuple[tuple[ObjectSlot, ...], ...]


@dataclass(frozen=True, slots=True)
class PreparedStageVariant:
    stage: DifferentiatedStage
    pair: AotGraphPair
    forward_inputs: tuple[ObjectSlot, ...]
    forward_outputs: tuple[ObjectSlot, ...]
    backward_inputs: tuple[ObjectSlot, ...]
    contributions: tuple[ObjectSlot, ...]
    saved_internal_object_ids: tuple[str, ...]
    public_output_leaves: tuple[int, ...]
    mutation_object_ids: tuple[str, ...]
    replacement_output_leaves: tuple[int, ...]
    forward_storage_handoffs: tuple[TaskStorageHandoff, ...]
    backward_storage_handoffs: tuple[TaskStorageHandoff, ...]


@dataclass(frozen=True, slots=True)
class TrainingObjects:
    catalog: ObjectCatalog
    registrations: tuple[RegistrationBinding, ...]
    root_slots: tuple[tuple[ObjectSlot, ...], ...]
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
    entrypoints: tuple[TaskEntrypoint, ...]
    #: What to call for each task. The entrypoint is neutral; this is not.
    executables: Mapping[str, GraphArtifact | OptimizerTaskArtifact | None]
    task_alternative_groups: tuple[TaskAlternativeGroup, ...]
    optimizer_task_ids: tuple[str, ...]
