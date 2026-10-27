"""Immutable bindings exchanged by training-lowering phases."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from shadowspill.build.lowering.registration import RegistrationBinding
from shadowspill.pytorch.capture.artifacts import (
    AotGraphPair,
)
from shadowspill.pytorch.graph_pairs import DifferentiatedStage
from shadowspill.pytorch.optimizer import OptimizerTensorRole
from shadowspill.pytorch.representations import RootInputKey
from shadowspill.task.slots import ObjectSlot, TaskStorageHandoff

from ..objects import ObjectCatalog


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


@dataclass(frozen=True, slots=True)
class FixedTensorBinding:
    """Frontend-owned constant tensor input required by a captured task."""

    object_id: str
    value: torch.Tensor


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
    parameter_objects: dict[str, str]
    gradients: tuple[GradientBinding, ...]
    gradient_by_parameter: dict[str, str]
    optimizer_objects: tuple[OptimizerObjectBinding, ...]


@dataclass(frozen=True, slots=True)
class TrainingBoundaries:
    object_ids: tuple[tuple[tuple[str, ...], ...], ...]
    root_objects: tuple[dict[RootInputKey, str], ...]
    cotangents: dict[tuple[int, str], str]
    fixed_tensors: dict[str, FixedTensorBinding]
    public_outputs: dict[int, tuple[str, ...]]
