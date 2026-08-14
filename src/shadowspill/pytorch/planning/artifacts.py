"""Immutable objects passed between PyTorch planning boundaries."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils._pytree import TreeSpec

from shadowspill.ir import ExecutionPlan
from shadowspill.planner import PressureFitResult
from shadowspill.planner._cache import CachedPressureFitResult
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
from shadowspill.pytorch.profiling.profiler import CudaTaskProfiler
from shadowspill.pytorch.runtime_adapter.allocator import InstalledAllocator
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
from .admission import SelectedSpatialLayout


@dataclass(frozen=True, slots=True)
class ForwardCaptureArtifacts:
    """Offline capture/partition output consumed by structural profiling."""

    signature: InputSignature
    cpu_inputs: tuple[object, ...]
    workload: ProfilingMetadata
    installed: InstalledAllocator
    device_ordinal: int
    fake_model: nn.Module
    capture: ExportCapture
    partitioned: PartitionedExport
    tasks: tuple[GraphArtifact, ...]
    output_tree_spec: TreeSpec


@dataclass(frozen=True, slots=True)
class ForwardProfileArtifacts:
    """Compiled manifests, measurements, and executable structural tasks."""

    profiler: CudaTaskProfiler
    manifests: ResolvedTaskManifests
    profiles: ProfilingResult
    compiled_tasks: CompiledTaskSet


@dataclass(frozen=True, slots=True)
class ForwardProgramArtifacts:
    """Canonical Program plus exact PressureFit call inputs."""

    lowered: LoweredForwardProgram
    measurements: dict[str, TaskMeasurement]
    measurements_by_profile: dict[str, TaskMeasurement]
    workspace_reserve: int
    simulation_config: SimulationConfig


@dataclass(frozen=True, slots=True)
class TrainingCaptureArtifacts:
    """Offline objective capture, stage graph pairs, and storage identities."""

    signatures: tuple[InputSignature, ...]
    cpu_inputs: tuple[tuple[object, ...], ...]
    workloads: tuple[ProfilingMetadata, ...]
    installed: InstalledAllocator
    device_ordinal: int
    fake_model: nn.Module
    captures: tuple[TrainingObjectiveCapture, ...]
    partitioned: tuple[PartitionedTrainingCapture, ...]
    layout: TrainingStorageLayout


@dataclass(frozen=True, slots=True)
class TrainingMaterializationArtifacts:
    """Allocator-owned model state and one captured optimizer factory result."""

    state: TrainingMaterializedState
    optimizer: torch.optim.Optimizer
    optimizer_capture: OptimizerCapture


@dataclass(frozen=True, slots=True)
class TrainingProfileArtifacts:
    """Unique structural ABI inventory, manifests, and task measurements."""

    compile_tasks: tuple[OptimizerTaskArtifact, ...]
    profile_keys: tuple[tuple[str, str | None], ...]
    profile_tasks: tuple[OptimizerTaskArtifact, ...]
    profile_metadata_digests: tuple[str | None, ...]
    profiler: CudaTaskProfiler
    manifests: ResolvedTaskManifests
    profiles: ProfilingResult


@dataclass(frozen=True, slots=True)
class TrainingProgramArtifacts:
    """Initial/recurrent Programs and their exact PressureFit call inputs."""

    initial: LoweredTrainingProgram
    recurrent: LoweredTrainingProgram
    measurements: dict[ProfileMeasurementKey, TaskMeasurement]
    measurements_by_profile: dict[str, TaskMeasurement]
    workspace_reserve: int
    simulation_config: SimulationConfig


@dataclass(frozen=True, slots=True)
class TrainingSelections:
    """Cached or freshly selected recurrent and optional first-step plans."""

    recurrent: CachedPressureFitResult
    initial: CachedPressureFitResult | None

    @property
    def results(self) -> tuple[PressureFitResult, ...]:
        if self.initial is None:
            return (self.recurrent.result,)
        return (self.initial.result, self.recurrent.result)


@dataclass(frozen=True, slots=True)
class TrainingExecutableArtifacts:
    """Selected compiled callables and their executable storage contracts."""

    tasks: CompiledTaskSet


@dataclass(frozen=True, slots=True)
class TrainingAdmissionArtifacts:
    """Physically admitted initial and recurrent execution plans."""

    recurrent: ExecutionPlan
    initial: ExecutionPlan | None
    recurrent_layout: SelectedSpatialLayout
    initial_layout: SelectedSpatialLayout | None


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
    "TrainingSelections",
]
