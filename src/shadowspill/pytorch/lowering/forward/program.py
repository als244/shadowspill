"""High-level orchestrator for partitioned forward lowering."""

from __future__ import annotations

from collections.abc import Mapping

import torch.nn as nn
from torch.utils._pytree import tree_flatten

from shadowspill.errors import CaptureError
from shadowspill.ir import MemoryLocation, SharedResidencyPolicy
from shadowspill.pytorch.capture.artifacts import GraphArtifact
from shadowspill.pytorch.capture.storage import TaskStorageContract
from shadowspill.pytorch.compilation.inductor import ExecutableRootAllocation
from shadowspill.pytorch.profiling import TaskMeasurement

from ...partition import PartitionedExport
from ..profiles import TaskProfileCatalog
from ..program import execution_device_id, publish_program
from .artifacts import ForwardPhysicalLayout, LoweredForwardProgram
from .objects import register_forward_objects
from .residency import derive_forward_residency, finalize_forward_shared_residency
from .tasks import emit_forward_tasks


def lower_partitioned_forward_program(
    model: nn.Module,
    partitioned: PartitionedExport,
    artifacts: tuple[GraphArtifact, ...],
    measurements: tuple[TaskMeasurement, ...],
    *,
    storage_contracts: Mapping[str, TaskStorageContract] | None = None,
    compiled_root_allocations: Mapping[str, tuple[ExecutableRootAllocation, ...]]
    | None = None,
    device_ordinal: int = 0,
    profile_compatibility_digests: tuple[str, ...] | None = None,
    public_output_locations: Mapping[int, MemoryLocation] | None = None,
    shared_residency_by_root: Mapping[
        int, tuple[SharedResidencyPolicy, bool]
    ] | None = None,
) -> LoweredForwardProgram:
    """Create one deterministic canonical program from forward task positions."""

    physical = resolve_forward_profiles(
        partitioned,
        artifacts,
        measurements,
        storage_contracts=storage_contracts,
        compiled_root_allocations=compiled_root_allocations,
        profile_compatibility_digests=profile_compatibility_digests,
    )
    device_id = execution_device_id(device_ordinal)
    objects = register_forward_objects(
        model,
        partitioned,
        device_id=device_id,
        shared_residency_by_root=shared_residency_by_root,
    )
    graph = emit_forward_tasks(
        partitioned,
        artifacts,
        objects,
        physical,
        device_id=device_id,
    )
    finalize_forward_shared_residency(objects, graph)
    initial_residency, final_residency = derive_forward_residency(
        objects,
        graph,
        public_output_locations=public_output_locations,
    )
    output_leaves, output_tree_spec = tree_flatten(partitioned.stages[-1].output)
    return LoweredForwardProgram(
        publish_program(
            objects.catalog,
            physical.profiles,
            graph.tasks,
            device_ordinal=device_ordinal,
        ),
        initial_residency,
        final_residency,
        graph.entrypoints,
        objects.registrations,
        objects.root_input_slots,
        graph.public_outputs,
        output_tree_spec,
        len(output_leaves),
    )


def resolve_forward_profiles(
    partitioned: PartitionedExport,
    artifacts: tuple[GraphArtifact, ...],
    measurements: tuple[TaskMeasurement, ...],
    *,
    storage_contracts: Mapping[str, TaskStorageContract] | None,
    compiled_root_allocations: Mapping[str, tuple[ExecutableRootAllocation, ...]]
    | None,
    profile_compatibility_digests: tuple[str, ...] | None,
) -> ForwardPhysicalLayout:
    occurrence_keys, profile_digests = _forward_profile_identities(
        partitioned,
        artifacts,
        measurements,
        profile_compatibility_digests,
    )
    profiles = _forward_profile_catalog(
        artifacts,
        measurements,
        occurrence_keys,
        profile_digests,
        storage_contracts=storage_contracts,
        compiled_root_allocations=compiled_root_allocations,
    )
    return ForwardPhysicalLayout(
        tuple(profiles.contract(artifact) for artifact in artifacts),
        tuple(
            profiles.layout(artifact, occurrence_key)
            for artifact, occurrence_key in zip(
                artifacts,
                occurrence_keys,
                strict=True,
            )
        ),
        profiles,
        _forward_profile_ids(profiles, artifacts, occurrence_keys),
    )


def _forward_profile_identities(
    partitioned: PartitionedExport,
    artifacts: tuple[GraphArtifact, ...],
    measurements: tuple[TaskMeasurement, ...],
    compatibility_digests: tuple[str, ...] | None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    stage_count = len(partitioned.stages)
    if len(artifacts) != stage_count or len(measurements) != stage_count:
        raise CaptureError("stage, artifact, and measurement counts must match")
    digests = (
        tuple(item.compatibility_digest for item in artifacts)
        if compatibility_digests is None
        else compatibility_digests
    )
    if len(digests) != stage_count:
        raise CaptureError("profile identities must align with forward stages")
    return (
        tuple(f"forward_position_{index:06d}" for index in range(stage_count)),
        digests,
    )


def _forward_profile_catalog(
    artifacts: tuple[GraphArtifact, ...],
    measurements: tuple[TaskMeasurement, ...],
    occurrence_keys: tuple[str, ...],
    profile_digests: tuple[str, ...],
    *,
    storage_contracts: Mapping[str, TaskStorageContract] | None,
    compiled_root_allocations: Mapping[str, tuple[ExecutableRootAllocation, ...]]
    | None,
) -> TaskProfileCatalog:
    return TaskProfileCatalog(
        {
            (
                artifact.compatibility_digest,
                occurrence_key,
            ): measurement
            for artifact, occurrence_key, measurement in zip(
                artifacts,
                occurrence_keys,
                measurements,
                strict=True,
            )
        },
        storage_contracts=storage_contracts,
        root_allocations=compiled_root_allocations,
        compatibility_digests={
            (
                artifact.compatibility_digest,
                occurrence_key,
            ): profile_digest
            for artifact, occurrence_key, profile_digest in zip(
                artifacts,
                occurrence_keys,
                profile_digests,
                strict=True,
            )
        },
        metadata_enabled=True,
    )


def _forward_profile_ids(
    profiles: TaskProfileCatalog,
    artifacts: tuple[GraphArtifact, ...],
    occurrence_keys: tuple[str, ...],
) -> tuple[str, ...]:
    return tuple(
        profiles.profile_id(
            artifact,
            profiles.additional_workspace_for_outputs(
                artifact,
                profiles.replacement_output_leaves(artifact),
                occurrence_key,
            ),
            metadata_digest=occurrence_key,
        )
        for artifact, occurrence_key in zip(
            artifacts,
            occurrence_keys,
            strict=True,
        )
    )


__all__ = ["lower_partitioned_forward_program", "resolve_forward_profiles"]
