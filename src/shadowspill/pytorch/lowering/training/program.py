"""High-level orchestrators for partitioned training lowering."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

import torch.nn as nn

from ...contracts import CaptureError
from ...inductor_adapter import ExecutableRootAllocation
from ...optimizer import OptimizerCapture
from ...output_contract import TaskStorageContract
from ...partition import PartitionedTrainingCapture
from ...profiling import TaskMeasurement
from ..profiles import CompiledLayoutCache, ProfileMeasurementKey, TaskProfileCatalog
from ..program import execution_device_id, publish_program
from .artifacts import LoweredTrainingProgram
from .bindings import bind_training_boundaries, prepare_training_variants
from .objects import register_training_objects
from .residency import derive_training_residency
from .tasks import emit_training_tasks


def lower_partitioned_training_program(
    model: nn.Module,
    captures: tuple[PartitionedTrainingCapture, ...],
    measurements: Mapping[ProfileMeasurementKey, TaskMeasurement],
    optimizer: OptimizerCapture,
    *,
    storage_contracts: Mapping[str, TaskStorageContract] | None = None,
    compiled_root_allocations: Mapping[str, tuple[ExecutableRootAllocation, ...]]
    | None = None,
    device_ordinal: int = 0,
    optimizer_phase: Literal["initial", "recurrent"] = "recurrent",
    optimizer_ordering: Literal["stage_interleaved", "tail"] = "stage_interleaved",
    layout_cache: CompiledLayoutCache | None = None,
    profiling_metadata_digests: tuple[str, ...] | None = None,
    profile_compatibility_digests: Mapping[tuple[str, str | None], str] | None = None,
) -> LoweredTrainingProgram:
    """Compose stage-local graph pairs into one accumulated training program."""

    metadata = _validate_training_lowering(
        captures,
        optimizer,
        optimizer_phase=optimizer_phase,
        optimizer_ordering=optimizer_ordering,
        profiling_metadata_digests=profiling_metadata_digests,
    )
    device_id = execution_device_id(device_ordinal)
    objects = register_training_objects(
        model,
        captures,
        optimizer,
        device_id=device_id,
    )
    profiles = TaskProfileCatalog(
        measurements,
        storage_contracts=storage_contracts,
        root_allocations=compiled_root_allocations,
        compatibility_digests=profile_compatibility_digests,
        metadata_enabled=profiling_metadata_digests is not None,
        layout_cache=layout_cache or CompiledLayoutCache(),
    )
    boundaries = bind_training_boundaries(
        captures,
        objects,
        profiles,
        metadata,
    )
    prepared = prepare_training_variants(
        captures,
        objects,
        boundaries,
        profiles,
        metadata,
    )
    graph = emit_training_tasks(
        prepared,
        metadata,
        objects,
        optimizer,
        profiles,
        optimizer_phase=optimizer_phase,
        optimizer_ordering=optimizer_ordering,
        device_id=device_id,
    )
    initial_residency, final_residency = derive_training_residency(
        objects,
        boundaries,
        graph.tasks,
    )
    return LoweredTrainingProgram(
        publish_program(
            objects.catalog,
            profiles,
            graph.tasks,
            device_ordinal=device_ordinal,
            recomputation_groups=graph.recomputation_groups,
        ),
        initial_residency,
        final_residency,
        objects.registrations,
        objects.root_slots,
        graph.entrypoints,
        objects.gradients,
        objects.optimizer_objects,
        tuple(boundaries.fixed_tensors.values()),
        graph.optimizer_task_ids,
    )


def _validate_training_lowering(
    captures: tuple[PartitionedTrainingCapture, ...],
    optimizer: OptimizerCapture,
    *,
    optimizer_phase: str,
    optimizer_ordering: str,
    profiling_metadata_digests: tuple[str, ...] | None,
) -> tuple[str | None, ...]:
    if not captures:
        raise CaptureError("partitioned training lowering requires a microbatch")
    if optimizer.recurrent is None:
        raise CaptureError("partitioned training requires a bounded optimizer task")
    if optimizer_phase not in {"initial", "recurrent"}:
        raise CaptureError(f"unknown optimizer phase {optimizer_phase!r}")
    if optimizer_ordering not in {"stage_interleaved", "tail"}:
        raise CaptureError(f"unknown optimizer ordering {optimizer_ordering!r}")
    if profiling_metadata_digests is None:
        return (None,) * len(captures)
    if len(profiling_metadata_digests) != len(captures):
        raise CaptureError(
            "profiling metadata must have one digest per training microbatch"
        )
    return profiling_metadata_digests


__all__ = ["lower_partitioned_training_program"]
