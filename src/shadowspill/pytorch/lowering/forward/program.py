"""High-level orchestrator for partitioned forward lowering."""

from __future__ import annotations

from collections.abc import Mapping

import torch.nn as nn
from torch.utils._pytree import tree_flatten

from shadowspill.pytorch.capture.artifacts import GraphArtifact
from shadowspill.pytorch.capture.storage import TaskStorageContract
from shadowspill.pytorch.compilation.inductor import ExecutableRootAllocation
from shadowspill.pytorch.compilation.profiling import TaskMeasurement

from ...contracts import CaptureError
from ...partition import PartitionedExport
from ..profiles import TaskProfileCatalog
from ..program import execution_device_id, publish_program
from .artifacts import ForwardPhysicalLayout, LoweredForwardProgram
from .objects import register_forward_objects
from .residency import derive_forward_residency
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
    objects = register_forward_objects(model, partitioned, device_id=device_id)
    graph = emit_forward_tasks(
        partitioned,
        artifacts,
        objects,
        physical,
        device_id=device_id,
    )
    initial_residency, final_residency = derive_forward_residency(objects, graph)
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
    stage_count = len(partitioned.stages)
    if len(artifacts) != stage_count or len(measurements) != stage_count:
        raise CaptureError("stage, artifact, and measurement counts must match")
    profile_digests = (
        tuple(item.compatibility_digest for item in artifacts)
        if profile_compatibility_digests is None
        else profile_compatibility_digests
    )
    if len(profile_digests) != stage_count:
        raise CaptureError("profile identities must align with forward stages")
    occurrence_keys = tuple(
        f"forward_position_{index:06d}" for index in range(stage_count)
    )
    profile_catalog = TaskProfileCatalog(
        {
            (artifact.compatibility_digest, occurrence_key): measurement
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
            (artifact.compatibility_digest, occurrence_key): profile_digest
            for artifact, occurrence_key, profile_digest in zip(
                artifacts,
                occurrence_keys,
                profile_digests,
                strict=True,
            )
        },
        metadata_enabled=True,
    )
    contracts = tuple(profile_catalog.contract(artifact) for artifact in artifacts)
    layouts = tuple(
        profile_catalog.layout(artifact, occurrence_key)
        for artifact, occurrence_key in zip(
            artifacts,
            occurrence_keys,
            strict=True,
        )
    )
    profile_ids = tuple(
        profile_catalog.profile_id(
            artifact,
            profile_catalog.mutation_transition_bytes(artifact, occurrence_key),
            metadata_digest=occurrence_key,
        )
        for artifact, occurrence_key in zip(
            artifacts,
            occurrence_keys,
            strict=True,
        )
    )
    return ForwardPhysicalLayout(
        contracts,
        layouts,
        profile_catalog,
        profile_ids,
    )


__all__ = ["lower_partitioned_forward_program", "resolve_forward_profiles"]
