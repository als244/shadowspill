"""High-level orchestrators for partitioned training lowering."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Literal

import torch.nn as nn

from shadowspill.errors import CaptureError
from shadowspill.pytorch.capture.storage import TaskStorageContract
from shadowspill.pytorch.optimizer import OptimizerCapture
from shadowspill.step import StepDataOrdering
from shadowspill.task.manifest import ExecutableRootAllocation
from shadowspill.task.profiles import TaskMeasurement

from ...graph_pairs import PartitionedTrainingCapture
from ..profiles import CompiledLayoutIndex, ProfileMeasurementKey, TaskProfileCatalog
from ..program import execution_device_id, publish_program
from .artifacts import LoweredTrainingProgram, TrainingBoundaries, TrainingObjects
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
    optimizer_ordering: Literal["stage_interleaved", "tail"] = "stage_interleaved",
    data_ordering: StepDataOrdering | None = None,
    layout_cache: CompiledLayoutIndex | None = None,
    profiling_metadata_digests: tuple[str, ...] | None = None,
    profile_compatibility_digests: Mapping[tuple[str, str | None], str] | None = None,
) -> LoweredTrainingProgram:
    """Compose stage-local graph pairs into one accumulated training program.

    ``data_ordering`` is the walk the program's task order follows; left out,
    the step runs depth-first, one microbatch after another.
    """

    return TrainingLowering(
        model,
        captures,
        measurements,
        optimizer,
        storage_contracts=storage_contracts,
        compiled_root_allocations=compiled_root_allocations,
        device_ordinal=device_ordinal,
        optimizer_ordering=optimizer_ordering,
        layout_cache=layout_cache,
        profiling_metadata_digests=profiling_metadata_digests,
        profile_compatibility_digests=profile_compatibility_digests,
    ).lower(data_ordering)


class TrainingLowering:
    """One step's lowering, shared by the programs of its orderings.

    An ordering is the walk a step's microbatches take; it decides the order
    of the tasks and which microbatch's backward creates each gradient, and
    nothing before that. Validating the captures, registering the objects
    and binding the stage boundaries are the same for every ordering, so
    they are done once, on the first program asked for, and each program
    then starts from its own copy of what they produced: object and profile
    identifiers are numbered in the order they are first used, and a copy
    leaves each program numbering them exactly as it would alone.
    """

    def __init__(
        self,
        model: nn.Module,
        captures: tuple[PartitionedTrainingCapture, ...],
        measurements: Mapping[ProfileMeasurementKey, TaskMeasurement],
        optimizer: OptimizerCapture,
        *,
        storage_contracts: Mapping[str, TaskStorageContract] | None = None,
        compiled_root_allocations: Mapping[str, tuple[ExecutableRootAllocation, ...]]
        | None = None,
        device_ordinal: int = 0,
        optimizer_ordering: Literal["stage_interleaved", "tail"] = "stage_interleaved",
        layout_cache: CompiledLayoutIndex | None = None,
        profiling_metadata_digests: tuple[str, ...] | None = None,
        profile_compatibility_digests: Mapping[tuple[str, str | None], str]
        | None = None,
    ) -> None:
        self._model = model
        self._captures = captures
        self._measurements = measurements
        self._optimizer = optimizer
        self._storage_contracts = storage_contracts
        self._compiled_root_allocations = compiled_root_allocations
        self._device_ordinal = device_ordinal
        self._optimizer_ordering = optimizer_ordering
        self._layout_cache = layout_cache or CompiledLayoutIndex()
        self._profiling_metadata_digests = profiling_metadata_digests
        self._profile_compatibility_digests = profile_compatibility_digests
        self._validated: tuple[str | None, ...] | None = None
        self._shared: (
            tuple[TrainingObjects, TaskProfileCatalog, TrainingBoundaries] | None
        ) = None

    def _metadata(self) -> tuple[str | None, ...]:
        if self._validated is None:
            self._validated = _validate_training_lowering(
                self._captures,
                self._optimizer,
                optimizer_ordering=self._optimizer_ordering,
                profiling_metadata_digests=self._profiling_metadata_digests,
            )
        return self._validated

    def _prepare(
        self,
    ) -> tuple[TrainingObjects, TaskProfileCatalog, TrainingBoundaries]:
        if self._shared is None:
            objects = register_training_objects(
                self._model,
                self._captures,
                self._optimizer,
                device_id=execution_device_id(self._device_ordinal),
            )
            profiles = TaskProfileCatalog(
                self._measurements,
                storage_contracts=self._storage_contracts,
                root_allocations=self._compiled_root_allocations,
                compatibility_digests=self._profile_compatibility_digests,
                metadata_enabled=self._profiling_metadata_digests is not None,
                layout_cache=self._layout_cache,
            )
            boundaries = bind_training_boundaries(
                self._captures,
                objects,
                profiles,
                self._metadata(),
            )
            self._shared = (objects, profiles, boundaries)
        return self._shared

    def lower(self, ordering: StepDataOrdering | None = None) -> LoweredTrainingProgram:
        """The program whose tasks follow ``ordering``, depth-first when it is
        left out: one microbatch after another."""

        captures = self._captures
        metadata = self._metadata()
        if ordering is None:
            ordering = StepDataOrdering.depth_first(len(captures))
        if ordering.microbatches != len(captures):
            raise CaptureError(
                f"ordering {ordering.label} covers {ordering.microbatches} "
                f"microbatches, but the step has {len(captures)}"
            )
        shared_objects, shared_profiles, shared_boundaries = self._prepare()
        objects = replace(
            shared_objects,
            catalog=shared_objects.catalog.copy(),
            parameter_objects=dict(shared_objects.parameter_objects),
            gradient_by_parameter=dict(shared_objects.gradient_by_parameter),
        )
        profiles = shared_profiles.copy()
        boundaries = replace(
            shared_boundaries,
            cotangents=dict(shared_boundaries.cotangents),
            fixed_tensors=dict(shared_boundaries.fixed_tensors),
            public_outputs=dict(shared_boundaries.public_outputs),
        )
        device_id = execution_device_id(self._device_ordinal)
        prepared = prepare_training_variants(
            captures,
            objects,
            boundaries,
            profiles,
            metadata,
            ordering,
        )
        graph = emit_training_tasks(
            prepared,
            metadata,
            objects,
            self._optimizer,
            profiles,
            optimizer_ordering=self._optimizer_ordering,
            ordering=ordering,
            device_id=device_id,
        )
        initial_residency, final_residency = derive_training_residency(
            objects,
            boundaries,
            graph.tasks,
            extra_public_outputs=tuple(
                slot.object_id
                for entry in graph.entrypoints
                if entry.task_id in graph.parameter_metric_schemas
                for slot in entry.output_slots
            ),
        )
        return LoweredTrainingProgram(
            publish_program(
                objects.catalog,
                profiles,
                graph.tasks,
                device_ordinal=self._device_ordinal,
                task_alternative_groups=graph.task_alternative_groups,
            ),
            initial_residency,
            final_residency,
            objects.registrations,
            objects.root_slots,
            graph.entrypoints,
            graph.executables,
            objects.gradients,
            objects.optimizer_objects,
            tuple(boundaries.fixed_tensors.values()),
            graph.optimizer_task_ids,
            graph.parameter_metric_schemas,
        )


def _validate_training_lowering(
    captures: tuple[PartitionedTrainingCapture, ...],
    optimizer: OptimizerCapture,
    *,
    optimizer_ordering: str,
    profiling_metadata_digests: tuple[str, ...] | None,
) -> tuple[str | None, ...]:
    if not captures:
        raise CaptureError("partitioned training lowering requires a microbatch")
    if optimizer.update is None:
        raise CaptureError("partitioned training requires a bounded optimizer task")
    if optimizer_ordering not in {"stage_interleaved", "tail"}:
        raise CaptureError(f"unknown optimizer ordering {optimizer_ordering!r}")
    if profiling_metadata_digests is None:
        return (None,) * len(captures)
    if len(profiling_metadata_digests) != len(captures):
        raise CaptureError(
            "profiling metadata must have one digest per training microbatch"
        )
    return profiling_metadata_digests


__all__ = ["TrainingLowering", "lower_partitioned_training_program"]
