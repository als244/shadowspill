"""Immutable objects passed between PyTorch planning boundaries."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils._pytree import TreeSpec

from shadowspill.ir import ExecutionPlan
from shadowspill.planner import AdmissionFacts, ProgramPlanResult
from shadowspill.pytorch.capture.aot import ExportCapture, TrainingObjectiveCapture
from shadowspill.pytorch.capture.artifacts import GraphArtifact
from shadowspill.pytorch.compilation.compiler import CompiledTaskSet
from shadowspill.pytorch.materialization.training import TrainingMaterializedState
from shadowspill.pytorch.optimizer import OptimizerCapture, OptimizerTaskArtifact
from shadowspill.pytorch.profiling import (
    ProfilingMetadata,
    ProfilingResult,
    ResolvedTaskManifests,
    TaskMeasurement,
)
from shadowspill.pytorch.profiling.profiler import TaskProfiler
from shadowspill.runtime.bootstrap import InstalledRuntime
from shadowspill.simulator import SimulationConfig

from ..graph_pairs import PartitionedTrainingCapture
from ..guards import InputSignature
from ..lowering.forward import LoweredForwardProgram
from ..lowering.profiles import ProfileMeasurementKey
from ..lowering.training import (
    LoweredTrainingProgram,
    TrainingStorageLayout,
)
from ..partition import PartitionedExport
from ..sharing import ResolvedSharedInput, ResolvedSharedOutput
from .admission import SelectedAdmission


@dataclass(frozen=True, slots=True)
class ForwardCaptureArtifacts:
    """Offline capture/partition output consumed by structural profiling."""

    signature: InputSignature
    cpu_inputs: tuple[object, ...]
    workload: ProfilingMetadata
    installed: InstalledRuntime
    device_ordinal: int
    fake_model: nn.Module
    capture: ExportCapture
    partitioned: PartitionedExport
    tasks: tuple[GraphArtifact, ...]
    output_tree_spec: TreeSpec
    shared_inputs: tuple[ResolvedSharedInput, ...]
    shared_outputs: tuple[ResolvedSharedOutput, ...]


@dataclass(frozen=True, slots=True)
class ForwardProfileArtifacts:
    """Compiled manifests, measurements, and executable structural tasks."""

    profiler: TaskProfiler
    manifests: ResolvedTaskManifests
    profiles: ProfilingResult
    compiled_tasks: CompiledTaskSet


@dataclass(frozen=True, slots=True)
class ForwardProgramArtifacts:
    """Canonical ShadowSpillProgram plus the exact inputs a search is given."""

    lowered: LoweredForwardProgram
    measurements: dict[str, TaskMeasurement]
    measurements_by_profile: dict[str, TaskMeasurement]
    workspace_reserve: int
    dynamic_scratch_reserve_bytes: int
    simulation_config: SimulationConfig
    admission: AdmissionFacts


@dataclass(frozen=True, slots=True)
class TrainingCaptureArtifacts:
    """Offline objective capture, stage graph pairs, and storage identities."""

    signatures: tuple[InputSignature, ...]
    cpu_inputs: tuple[tuple[object, ...], ...]
    workloads: tuple[ProfilingMetadata, ...]
    installed: InstalledRuntime
    device_ordinal: int
    fake_model: nn.Module
    captures: tuple[TrainingObjectiveCapture, ...]
    partitioned: tuple[PartitionedTrainingCapture, ...]
    layout: TrainingStorageLayout


@dataclass(frozen=True, slots=True)
class TrainingMaterializationArtifacts:
    """Allocator-owned model state and the optimizer captured over it.

    ``optimizer_parameters`` are the optimizer's parameters by the model's
    names: the model's own, or the master copy planning made of one.
    """

    state: TrainingMaterializedState
    optimizer: torch.optim.Optimizer
    optimizer_capture: OptimizerCapture
    optimizer_parameters: Mapping[str, nn.Parameter]


@dataclass(frozen=True, slots=True)
class TrainingProfileArtifacts:
    """Unique structural contract inventory, manifests, and task measurements."""

    partitioned: tuple[PartitionedTrainingCapture, ...]
    compile_tasks: tuple[OptimizerTaskArtifact, ...]
    profile_keys: tuple[tuple[str, str | None], ...]
    profile_tasks: tuple[OptimizerTaskArtifact, ...]
    profile_metadata_digests: tuple[str | None, ...]
    profiler: TaskProfiler
    manifests: ResolvedTaskManifests
    profiles: ProfilingResult


@dataclass(frozen=True, slots=True)
class TrainingProgramArtifacts:
    """The step's canonical ShadowSpillProgram plus the exact inputs a search is
    given."""

    lowered: LoweredTrainingProgram
    measurements: dict[ProfileMeasurementKey, TaskMeasurement]
    measurements_by_profile: dict[str, TaskMeasurement]
    workspace_reserve: int
    dynamic_scratch_reserve_bytes: int
    simulation_config: SimulationConfig
    admission: AdmissionFacts


@dataclass(frozen=True, slots=True)
class TrainingExecutableArtifacts:
    """Selected compiled callables and their executable storage contracts."""

    tasks: CompiledTaskSet


@dataclass(frozen=True, slots=True)
class TrainingAdmissionArtifacts:
    """The step's physically admitted execution plan."""

    plan: ExecutionPlan
    admission: SelectedAdmission
    result: ProgramPlanResult


__all__ = [
    "ForwardCaptureArtifacts",
    "ForwardProfileArtifacts",
    "ForwardProgramArtifacts",
    "TrainingAdmissionArtifacts",
    "TrainingCaptureArtifacts",
    "TrainingExecutableArtifacts",
    "TrainingMaterializationArtifacts",
    "TrainingProfileArtifacts",
    "TrainingProgramArtifacts",
]
