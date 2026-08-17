"""Deterministic construction of framework-facing planning diagnostics."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass

from shadowspill.ir import (
    AliasGroupSpec,
    ExecutionPlan,
    ObjectSpec,
    Program,
    TaskProfile,
    TaskSpec,
)
from shadowspill.pytorch.capture.artifacts import AotGraphPair, GraphArtifact
from shadowspill.pytorch.capture.storage import (
    MutationBinding,
    OutputView,
    StorageRoot,
    TaskStorageContract,
)
from shadowspill.pytorch.compilation.inductor import ExecutableTaskManifest
from shadowspill.pytorch.compilation.layout import (
    CompiledTaskLayout,
    reconcile_compiled_task_layout,
    replacement_transition_bytes,
)
from shadowspill.pytorch.diagnostics.plan import (
    PlanAllocationABIStep,
    PlanAllocationEvent,
    PlanCompiledOutputView,
    PlanCompiledRoot,
    PlanGraphPair,
    PlanGraphProfile,
    PlanMutationBinding,
    PlanObjectFootprint,
    PlanOutputView,
    PlanRepresentativeInput,
    PlanStorageRoot,
    PlanTaskStage,
    PlanUniqueStage,
)
from shadowspill.pytorch.graph_pairs import (
    DifferentiatedStage,
    GraphPairVariant,
    PartitionedTrainingCapture,
    saved_value_footprint,
)
from shadowspill.pytorch.lowering.forward import LoweredForwardProgram, TaskEntrypoint
from shadowspill.pytorch.lowering.profiles import ProfileMeasurementKey
from shadowspill.pytorch.lowering.training import (
    LoweredTrainingProgram,
    TrainingTaskEntrypoint,
)
from shadowspill.pytorch.optimizer import OptimizerTaskArtifact
from shadowspill.pytorch.profiling import TaskMeasurement
from shadowspill.pytorch.profiling.context import profile_input_context_digest


@dataclass(frozen=True, slots=True)
class _TrainingInventoryIndex:
    program: Program
    task_by_id: Mapping[str, TaskSpec]
    profile_by_id: Mapping[str, TaskProfile]
    entrypoint_by_key: Mapping[tuple[int, int, str, str], TrainingTaskEntrypoint]
    selected_ids: frozenset[str]
    execution_ordinal: Mapping[str, int]
    occurrence_keys: Mapping[tuple[int, int], str]
    stages_by_key: Mapping[str, tuple[tuple[int, int, DifferentiatedStage], ...]]
    unique_id_by_key: Mapping[str, str]
    chosen_by_occurrence: Mapping[tuple[int, int], str]


@dataclass(frozen=True, slots=True)
class _ForwardInventoryIndex:
    task_by_id: Mapping[str, TaskSpec]
    profile_by_id: Mapping[str, TaskProfile]
    selected_ids: frozenset[str]
    execution_ordinal: Mapping[str, int]
    unique_id_by_key: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class _GraphProfileContext:
    artifact: GraphArtifact
    direction: str
    task: TaskSpec
    profile: TaskProfile
    measurement: TaskMeasurement
    manifest: ExecutableTaskManifest
    layout: CompiledTaskLayout
    inputs: tuple[PlanObjectFootprint, ...]
    mutations: tuple[PlanObjectFootprint, ...]
    outputs: tuple[PlanObjectFootprint, ...]


def training_stage_inventory(
    captures: tuple[PartitionedTrainingCapture, ...],
    lowered: LoweredTrainingProgram,
    execution_plan: ExecutionPlan,
    measurements: Mapping[ProfileMeasurementKey, TaskMeasurement],
    manifests: Mapping[str, ExecutableTaskManifest],
    profiling_metadata_digests: tuple[str, ...] | None = None,
) -> tuple[tuple[PlanTaskStage, ...], tuple[PlanUniqueStage, ...]]:
    """Describe task occurrences and every legal structural graph pair."""

    index = _index_training_inventory(captures, lowered, execution_plan)
    task_map = _training_task_inventory(
        lowered,
        index,
        measurements,
        manifests,
        profiling_metadata_digests,
    )
    unique_stages = tuple(
        _training_unique_stage(
            structural_key,
            index,
            measurements,
            manifests,
            profiling_metadata_digests,
        )
        for structural_key in sorted(index.stages_by_key)
    )
    return task_map, unique_stages


def _index_training_inventory(
    captures: tuple[PartitionedTrainingCapture, ...],
    lowered: LoweredTrainingProgram,
    execution_plan: ExecutionPlan,
) -> _TrainingInventoryIndex:
    program = lowered.program
    selected = execution_plan.program.selected_tasks(execution_plan.selections)
    occurrence_keys, stages_by_key = _index_training_occurrences(captures)
    unique_id_by_key = {
        key: f"unique_stage_{index:04d}"
        for index, key in enumerate(sorted(stages_by_key))
    }
    selected_ids = frozenset(task.task_id for task in selected)
    return _TrainingInventoryIndex(
        program=program,
        task_by_id={task.task_id: task for task in program.tasks},
        profile_by_id={profile.profile_id: profile for profile in program.profiles},
        entrypoint_by_key={
            _entrypoint_key(entrypoint): entrypoint
            for entrypoint in lowered.entrypoints
            if entrypoint.stage_index is not None and entrypoint.variant is not None
        },
        selected_ids=selected_ids,
        execution_ordinal={task.task_id: index for index, task in enumerate(selected)},
        occurrence_keys=occurrence_keys,
        stages_by_key=stages_by_key,
        unique_id_by_key=unique_id_by_key,
        chosen_by_occurrence=_chosen_training_variants(lowered, selected_ids),
    )


def _index_training_occurrences(
    captures: tuple[PartitionedTrainingCapture, ...],
) -> tuple[
    dict[tuple[int, int], str],
    dict[str, tuple[tuple[int, int, DifferentiatedStage], ...]],
]:
    occurrence_keys: dict[tuple[int, int], str] = {}
    grouped: dict[str, list[tuple[int, int, DifferentiatedStage]]] = {}
    for microbatch, capture in enumerate(captures):
        for stage_index, stage in enumerate(capture.stages):
            key = _stage_key(stage)
            occurrence_keys[microbatch, stage_index] = key
            grouped.setdefault(key, []).append((microbatch, stage_index, stage))
    return occurrence_keys, {
        key: tuple(occurrences) for key, occurrences in grouped.items()
    }


def _chosen_training_variants(
    lowered: LoweredTrainingProgram,
    selected_ids: frozenset[str],
) -> dict[tuple[int, int], str]:
    chosen: dict[tuple[int, int], str] = {}
    for entrypoint in lowered.entrypoints:
        if (
            entrypoint.microbatch is not None
            and entrypoint.stage_index is not None
            and entrypoint.variant is not None
            and entrypoint.phase == "forward"
            and entrypoint.task_id in selected_ids
        ):
            chosen[entrypoint.microbatch, entrypoint.stage_index] = entrypoint.variant
    return chosen


def _training_task_inventory(
    lowered: LoweredTrainingProgram,
    index: _TrainingInventoryIndex,
    measurements: Mapping[ProfileMeasurementKey, TaskMeasurement],
    manifests: Mapping[str, ExecutableTaskManifest],
    metadata_digests: tuple[str, ...] | None,
) -> tuple[PlanTaskStage, ...]:
    auxiliary_ordinals: dict[str, int] = {}
    tasks: list[PlanTaskStage] = []
    for entrypoint in lowered.entrypoints:
        auxiliary_ordinal = auxiliary_ordinals.get(entrypoint.phase, 0)
        tasks.append(
            _training_task_stage(
                entrypoint,
                index,
                measurements,
                manifests,
                metadata_digests,
                auxiliary_ordinal,
            )
        )
        if entrypoint.microbatch is None or entrypoint.stage_index is None:
            auxiliary_ordinals[entrypoint.phase] = auxiliary_ordinal + 1
    return tuple(tasks)


def _training_task_stage(
    entrypoint: TrainingTaskEntrypoint,
    index: _TrainingInventoryIndex,
    measurements: Mapping[ProfileMeasurementKey, TaskMeasurement],
    manifests: Mapping[str, ExecutableTaskManifest],
    metadata_digests: tuple[str, ...] | None,
    auxiliary_ordinal: int,
) -> PlanTaskStage:
    artifact = entrypoint.artifact
    structural_contract = (
        artifact.compatibility_digest if artifact is not None else "opaque"
    )
    occurrence = _training_occurrence_identity(
        entrypoint, structural_contract, auxiliary_ordinal, index
    )
    contract_digests = _task_contract_digests(
        artifact,
        entrypoint,
        measurements,
        manifests,
        metadata_digests,
    )
    ordinal = index.execution_ordinal.get(entrypoint.task_id)
    task = index.task_by_id[entrypoint.task_id]
    profile = index.profile_by_id[task.profile_id]
    return PlanTaskStage(
        task_id=entrypoint.task_id,
        execution_ordinal=ordinal,
        execution_task_id=None if ordinal is None else f"execution_{ordinal:06d}",
        semantic_name=occurrence[0],
        phase=entrypoint.phase,
        microbatch=entrypoint.microbatch,
        stage_occurrence_id=occurrence[1],
        unique_stage_id=occurrence[2],
        structural_contract_key=structural_contract,
        semantic_contract_digest=contract_digests[0],
        executable_contract_digest=contract_digests[1],
        compiled_layout_digest=contract_digests[2],
        graph_pair_variant=entrypoint.variant,
        chosen_graph_pair_variant=occurrence[3],
        selected=entrypoint.task_id in index.selected_ids,
        profile_compatibility_digest=profile.compatibility_digest,
        profiling_metadata_digest=_metadata_for(entrypoint, metadata_digests),
    )


def _training_occurrence_identity(
    entrypoint: TrainingTaskEntrypoint,
    structural_contract: str,
    auxiliary_ordinal: int,
    index: _TrainingInventoryIndex,
) -> tuple[str, str | None, str, str | None]:
    if entrypoint.microbatch is None or entrypoint.stage_index is None:
        return (
            f"{entrypoint.phase}.component_{auxiliary_ordinal:04d}",
            None,
            f"auxiliary_contract_{structural_contract[:16]}",
            None,
        )
    occurrence = entrypoint.microbatch, entrypoint.stage_index
    stage_id = (
        f"microbatch_{entrypoint.microbatch:04d}.stage_{entrypoint.stage_index:04d}"
    )
    structural_key = index.occurrence_keys[occurrence]
    return (
        f"{stage_id}.{entrypoint.phase}.{entrypoint.variant}",
        stage_id,
        index.unique_id_by_key[structural_key],
        index.chosen_by_occurrence.get(occurrence),
    )


def _task_contract_digests(
    artifact: OptimizerTaskArtifact | None,
    entrypoint: TrainingTaskEntrypoint,
    measurements: Mapping[ProfileMeasurementKey, TaskMeasurement],
    manifests: Mapping[str, ExecutableTaskManifest],
    metadata_digests: tuple[str, ...] | None,
) -> tuple[str | None, str | None, str | None]:
    if not isinstance(artifact, GraphArtifact):
        return None, None, None
    manifest = manifests[artifact.compatibility_digest]
    measurement = _training_measurement(
        artifact, entrypoint, measurements, metadata_digests
    )
    layout = reconcile_compiled_task_layout(
        manifest.storage_contract,
        measurement,
        root_allocations=manifest.root_allocations,
    )
    return (
        artifact.storage_contract.compatibility_digest,
        manifest.storage_contract.compatibility_digest,
        layout.compatibility_digest,
    )


def _training_unique_stage(
    structural_key: str,
    index: _TrainingInventoryIndex,
    measurements: Mapping[ProfileMeasurementKey, TaskMeasurement],
    manifests: Mapping[str, ExecutableTaskManifest],
    metadata_digests: tuple[str, ...] | None,
) -> PlanUniqueStage:
    occurrences = index.stages_by_key[structural_key]
    microbatch, stage_index, representative = occurrences[0]
    graph_pairs = tuple(
        _training_graph_pair(
            option,
            microbatch,
            stage_index,
            index,
            measurements,
            manifests,
            metadata_digests,
        )
        for option in representative.graph_pairs.variants
    )
    return PlanUniqueStage(
        unique_stage_id=index.unique_id_by_key[structural_key],
        structural_key=structural_key,
        module_targets=tuple(
            dict.fromkeys(
                stage.example.stage.module_target for _, _, stage in occurrences
            )
        ),
        occurrence_count=len(occurrences),
        graph_pairs=graph_pairs,
    )


def _training_graph_pair(
    option: GraphPairVariant,
    microbatch: int,
    stage_index: int,
    index: _TrainingInventoryIndex,
    measurements: Mapping[ProfileMeasurementKey, TaskMeasurement],
    manifests: Mapping[str, ExecutableTaskManifest],
    metadata_digests: tuple[str, ...] | None,
) -> PlanGraphPair:
    variant = option.option_id
    pair = option.pair
    forward_entrypoint = index.entrypoint_by_key[
        microbatch, stage_index, variant, "forward"
    ]
    backward_entrypoint = index.entrypoint_by_key[
        microbatch, stage_index, variant, "backward"
    ]
    footprint = saved_value_footprint(pair)
    return PlanGraphPair(
        variant=variant,
        memory_budget=option.memory_budget,
        recomputation=pair.recomputation,
        saved_value_count=pair.saved_value_count,
        specialized_unit_tangent_count=pair.specialized_unit_tangent_count,
        saved_input_root_count=len(footprint.input_root_ids),
        saved_boundary_root_count=len(footprint.boundary_root_ids),
        saved_internal_root_count=len(footprint.internal_root_ids),
        saved_input_minimum_bytes=footprint.input_minimum_bytes,
        saved_boundary_minimum_bytes=footprint.boundary_minimum_bytes,
        saved_internal_minimum_bytes=footprint.internal_minimum_bytes,
        forward=_graph_profile(
            pair.forward,
            "forward",
            index.task_by_id[forward_entrypoint.task_id],
            index.program,
            _training_measurement(
                pair.forward,
                forward_entrypoint,
                measurements,
                metadata_digests,
            ),
            manifests[pair.forward.compatibility_digest],
        ),
        backward=_graph_profile(
            pair.backward,
            "backward",
            index.task_by_id[backward_entrypoint.task_id],
            index.program,
            _training_measurement(
                pair.backward,
                backward_entrypoint,
                measurements,
                metadata_digests,
            ),
            manifests[pair.backward.compatibility_digest],
        ),
    )


def _training_measurement(
    artifact: GraphArtifact,
    entrypoint: TrainingTaskEntrypoint,
    measurements: Mapping[ProfileMeasurementKey, TaskMeasurement],
    metadata_digests: tuple[str, ...] | None,
) -> TaskMeasurement:
    metadata = _metadata_for(entrypoint, metadata_digests)
    measurement = measurements.get(
        (
            artifact.compatibility_digest,
            metadata,
            profile_input_context_digest(artifact),
        )
    )
    if measurement is None and metadata_digests is None:
        measurement = measurements.get(artifact.compatibility_digest)
    if measurement is None:
        raise ValueError(
            "diagnostic profile is missing "
            f"artifact={artifact.compatibility_digest}, "
            f"profiling_metadata={metadata}"
        )
    return measurement


def _metadata_for(
    entrypoint: TrainingTaskEntrypoint,
    metadata_digests: tuple[str, ...] | None,
) -> str | None:
    if entrypoint.microbatch is None or metadata_digests is None:
        return None
    return metadata_digests[entrypoint.microbatch]


def forward_stage_inventory(
    lowered: LoweredForwardProgram,
    execution_plan: ExecutionPlan,
    measurements: Mapping[str, TaskMeasurement],
    manifests: Mapping[str, ExecutableTaskManifest],
    *,
    profiling_metadata_digest: str | None = None,
) -> tuple[tuple[PlanTaskStage, ...], tuple[PlanUniqueStage, ...]]:
    """Describe deduplicated inference stages and task occurrences."""

    index = _index_forward_inventory(lowered, execution_plan)
    tasks = tuple(
        _forward_task_stage(
            occurrence,
            entrypoint,
            lowered,
            index,
            measurements,
            manifests,
            profiling_metadata_digest,
        )
        for occurrence, entrypoint in enumerate(lowered.entrypoints)
    )
    unique_stages = tuple(
        _forward_unique_stage(key, lowered, index, measurements, manifests)
        for key in sorted(index.unique_id_by_key)
    )
    return tasks, unique_stages


def _index_forward_inventory(
    lowered: LoweredForwardProgram,
    execution_plan: ExecutionPlan,
) -> _ForwardInventoryIndex:
    selected = execution_plan.program.selected_tasks(execution_plan.selections)
    task_by_id = {task.task_id: task for task in lowered.program.tasks}
    profile_by_id = {
        profile.profile_id: profile for profile in lowered.program.profiles
    }
    keys = sorted(
        {
            profile_by_id[
                task_by_id[entrypoint.task_id].profile_id
            ].compatibility_digest
            for entrypoint in lowered.entrypoints
        }
    )
    return _ForwardInventoryIndex(
        task_by_id=task_by_id,
        profile_by_id=profile_by_id,
        selected_ids=frozenset(task.task_id for task in selected),
        execution_ordinal={task.task_id: index for index, task in enumerate(selected)},
        unique_id_by_key={
            key: f"unique_stage_{index:04d}" for index, key in enumerate(keys)
        },
    )


def _forward_task_stage(
    occurrence: int,
    entrypoint: TaskEntrypoint,
    lowered: LoweredForwardProgram,
    index: _ForwardInventoryIndex,
    measurements: Mapping[str, TaskMeasurement],
    manifests: Mapping[str, ExecutableTaskManifest],
    metadata_digest: str | None,
) -> PlanTaskStage:
    artifact_key = entrypoint.artifact.compatibility_digest
    manifest = manifests[artifact_key]
    task = index.task_by_id[entrypoint.task_id]
    profile = index.profile_by_id[task.profile_id]
    layout = reconcile_compiled_task_layout(
        manifest.storage_contract,
        measurements[profile.compatibility_digest],
        root_allocations=manifest.root_allocations,
    )
    ordinal = index.execution_ordinal.get(entrypoint.task_id)
    return PlanTaskStage(
        task_id=entrypoint.task_id,
        execution_ordinal=ordinal,
        execution_task_id=None if ordinal is None else f"execution_{ordinal:06d}",
        semantic_name=f"stage_{occurrence:04d}.forward.inference",
        phase="forward",
        microbatch=None,
        stage_occurrence_id=f"stage_{occurrence:04d}",
        unique_stage_id=index.unique_id_by_key[profile.compatibility_digest],
        structural_contract_key=artifact_key,
        semantic_contract_digest=(
            entrypoint.artifact.storage_contract.compatibility_digest
        ),
        executable_contract_digest=(manifest.storage_contract.compatibility_digest),
        compiled_layout_digest=layout.compatibility_digest,
        graph_pair_variant="inference",
        chosen_graph_pair_variant="inference",
        selected=entrypoint.task_id in index.selected_ids,
        profile_compatibility_digest=profile.compatibility_digest,
        profiling_metadata_digest=metadata_digest,
    )


def _forward_unique_stage(
    key: str,
    lowered: LoweredForwardProgram,
    index: _ForwardInventoryIndex,
    measurements: Mapping[str, TaskMeasurement],
    manifests: Mapping[str, ExecutableTaskManifest],
) -> PlanUniqueStage:
    occurrences = tuple(
        entrypoint
        for entrypoint in lowered.entrypoints
        if index.profile_by_id[
            index.task_by_id[entrypoint.task_id].profile_id
        ].compatibility_digest
        == key
    )
    representative = occurrences[0]
    task = index.task_by_id[representative.task_id]
    profile = _graph_profile(
        representative.artifact,
        "forward",
        task,
        lowered.program,
        measurements[key],
        manifests[representative.artifact.compatibility_digest],
    )
    return PlanUniqueStage(
        unique_stage_id=index.unique_id_by_key[key],
        structural_key=key,
        module_targets=tuple(
            dict.fromkeys(entrypoint.module_target for entrypoint in occurrences)
        ),
        occurrence_count=len(occurrences),
        graph_pairs=(
            PlanGraphPair(
                variant="inference",
                memory_budget=None,
                recomputation=False,
                saved_value_count=0,
                specialized_unit_tangent_count=0,
                saved_input_root_count=0,
                saved_boundary_root_count=0,
                saved_internal_root_count=0,
                saved_input_minimum_bytes=0,
                saved_boundary_minimum_bytes=0,
                saved_internal_minimum_bytes=0,
                forward=profile,
                backward=None,
            ),
        ),
    )


def _stage_key(stage: DifferentiatedStage) -> str:
    payload = {
        "roots": list(stage.differentiable_output_indices),
        "variants": [
            {
                "option_id": item.option_id,
                "memory_budget": item.memory_budget,
                "pair": _pair_key(item.pair),
            }
            for item in stage.graph_pairs.variants
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def _pair_key(pair: AotGraphPair) -> dict[str, object]:
    return {
        "forward": pair.forward.compatibility_digest,
        "backward": pair.backward.compatibility_digest,
        "recomputation": pair.recomputation,
        "saved_value_count": pair.saved_value_count,
        "specialized_unit_tangent_count": pair.specialized_unit_tangent_count,
    }


def _entrypoint_key(
    entrypoint: TrainingTaskEntrypoint,
) -> tuple[int, int, str, str]:
    if (
        entrypoint.microbatch is None
        or entrypoint.stage_index is None
        or entrypoint.variant is None
    ):
        raise ValueError("entrypoint has no graph-stage identity")
    return (
        entrypoint.microbatch,
        entrypoint.stage_index,
        entrypoint.variant,
        entrypoint.phase,
    )


def _graph_profile(
    artifact: GraphArtifact,
    direction: str,
    task: TaskSpec,
    program: Program,
    measurement: TaskMeasurement,
    manifest: ExecutableTaskManifest,
) -> PlanGraphProfile:
    inputs, mutations, outputs = _task_footprints(program, task)
    context = _GraphProfileContext(
        artifact=artifact,
        direction=direction,
        task=task,
        profile=_task_profile(program, task),
        measurement=measurement,
        manifest=manifest,
        layout=reconcile_compiled_task_layout(
            manifest.storage_contract,
            measurement,
            root_allocations=manifest.root_allocations,
        ),
        inputs=inputs,
        mutations=mutations,
        outputs=outputs,
    )
    return _build_graph_profile(context)


def _build_graph_profile(context: _GraphProfileContext) -> PlanGraphProfile:
    artifact = context.artifact
    contract = context.manifest.storage_contract
    measurement = context.measurement
    return PlanGraphProfile(
        direction=context.direction,
        structural_contract_key=artifact.compatibility_digest,
        semantic_contract_digest=artifact.storage_contract.compatibility_digest,
        semantic_contract_capture_ns=artifact.storage_contract_capture_ns,
        semantic_roots=_plan_storage_roots(artifact.storage_contract),
        semantic_output_views=_plan_output_views(artifact.storage_contract),
        semantic_mutations=_plan_mutations(artifact.storage_contract),
        executable_contract_digest=contract.compatibility_digest,
        executable_contract_capture_ns=context.manifest.contract_capture_ns,
        executable_roots=_plan_storage_roots(contract),
        executable_output_views=_plan_output_views(contract),
        executable_mutations=_plan_mutations(contract),
        compiled_layout_digest=context.layout.compatibility_digest,
        compiled_roots=_plan_compiled_roots(context.layout),
        compiled_output_views=_plan_compiled_views(context.layout),
        physical_profile_wall_time_ns=measurement.profiling_wall_time_ns,
        representative_task_id=context.task.task_id,
        runtime_ns=measurement.runtime_ns,
        samples_ns=measurement.samples_ns,
        provenance=measurement.provenance,
        representative_inputs=_plan_representative_inputs(measurement),
        profile_phase_timings_ns=measurement.phase_timings_ns,
        timing_relative_mad=measurement.timing_relative_mad,
        timing_half_drift=measurement.timing_half_drift,
        timing_unstable=measurement.timing_unstable,
        inputs=context.inputs,
        mutations=context.mutations,
        outputs=context.outputs,
        input_logical_bytes=_logical_bytes(context.inputs),
        input_allocation_bytes=_unique_allocation_bytes(context.inputs),
        mutation_logical_bytes=_logical_bytes(context.mutations),
        mutation_allocation_bytes=_unique_allocation_bytes(context.mutations),
        output_logical_bytes=_logical_bytes(context.outputs),
        output_allocation_bytes=_unique_allocation_bytes(context.outputs),
        workspace_requested_bytes=measurement.workspace_requested_bytes,
        workspace_charged_bytes=measurement.workspace_charged_bytes,
        replacement_transition_bytes=replacement_transition_bytes(
            contract, context.layout
        ),
        task_workspace_bytes=context.profile.workspace_bytes,
        workspace_extent_bytes=measurement.workspace_extent_bytes,
        persistent_extent_bytes=measurement.persistent_extent_bytes,
        allocation_contract_digest=(
            None
            if measurement.allocation_contract is None
            else measurement.allocation_contract.compatibility_digest
        ),
        allocation_contract=_plan_allocation_contract(measurement),
        allocation_timeline=_plan_allocation_timeline(measurement),
    )


def _task_profile(program: Program, task: TaskSpec) -> TaskProfile:
    return next(
        profile for profile in program.profiles if profile.profile_id == task.profile_id
    )


def _task_footprints(
    program: Program,
    task: TaskSpec,
) -> tuple[
    tuple[PlanObjectFootprint, ...],
    tuple[PlanObjectFootprint, ...],
    tuple[PlanObjectFootprint, ...],
]:
    objects = {item.object_id: item for item in program.objects}
    aliases = {item.alias_group_id: item for item in program.alias_groups}

    def resolve(object_ids: tuple[str, ...]) -> tuple[PlanObjectFootprint, ...]:
        return tuple(
            _footprint(objects[object_id], aliases[objects[object_id].alias_group_id])
            for object_id in object_ids
        )

    return (
        resolve(task.inputs),
        resolve(tuple(item.object_id for item in task.mutations)),
        resolve(task.outputs),
    )


def _plan_storage_roots(
    contract: TaskStorageContract,
) -> tuple[PlanStorageRoot, ...]:
    return tuple(_plan_storage_root(root) for root in contract.roots)


def _plan_storage_root(root: StorageRoot) -> PlanStorageRoot:
    return PlanStorageRoot(
        root_id=root.root_id,
        kind=root.kind.value,
        source_input=root.source_input,
        producer_node=root.producer_node,
        producer_target=root.producer_target,
        producer_result=root.producer_result,
        minimum_span_bytes=root.minimum_span_bytes,
    )


def _plan_output_views(
    contract: TaskStorageContract,
) -> tuple[PlanOutputView, ...]:
    return tuple(_plan_output_view(view) for view in contract.output_views)


def _plan_output_view(view: OutputView) -> PlanOutputView:
    return PlanOutputView(
        leaf_index=view.leaf_index,
        root_id=view.root_id,
        offset_bytes=view.offset_bytes,
        span_bytes=view.span_bytes,
        shape=view.shape,
        stride=view.stride,
        dtype=view.dtype,
        layout=view.layout,
    )


def _plan_mutations(
    contract: TaskStorageContract,
) -> tuple[PlanMutationBinding, ...]:
    return tuple(_plan_mutation(mutation) for mutation in contract.mutations)


def _plan_mutation(mutation: MutationBinding) -> PlanMutationBinding:
    return PlanMutationBinding(
        input_position=mutation.input_position,
        replacement_output_leaf=mutation.replacement_output_leaf,
        producer_node=mutation.producer_node,
        producer_target=mutation.producer_target,
        argument_name=mutation.argument_name,
    )


def _plan_compiled_roots(
    layout: CompiledTaskLayout,
) -> tuple[PlanCompiledRoot, ...]:
    return tuple(
        PlanCompiledRoot(
            root_id=root.root_id,
            allocation_ordinal=root.allocation_ordinal,
            requested_bytes=root.requested_bytes,
            charged_bytes=root.charged_bytes,
        )
        for root in layout.roots
    )


def _plan_compiled_views(
    layout: CompiledTaskLayout,
) -> tuple[PlanCompiledOutputView, ...]:
    return tuple(
        PlanCompiledOutputView(
            leaf_index=view.leaf_index,
            root_id=view.root_id,
            allocation_ordinal=view.allocation_ordinal,
            offset_bytes=view.offset_bytes,
        )
        for view in layout.output_views
    )


def _plan_representative_inputs(
    measurement: TaskMeasurement,
) -> tuple[PlanRepresentativeInput, ...]:
    return tuple(
        PlanRepresentativeInput(
            position=item.position,
            role=item.role.value,
            source=item.source,
            value_policy=item.value_policy,
            dtype=item.dtype,
            shape=item.shape,
            stride=item.stride,
            storage_offset=item.storage_offset,
            alias_group=item.alias_group,
            consumer_targets=item.consumer_targets,
        )
        for item in measurement.representative_inputs
    )


def _plan_allocation_timeline(
    measurement: TaskMeasurement,
) -> tuple[PlanAllocationEvent, ...]:
    return tuple(
        PlanAllocationEvent(
            allocation_ordinal=event.allocation_ordinal,
            operation=event.operation.value,
            requested_bytes=event.requested_bytes,
            charged_bytes=event.charged_bytes,
            output_leaf_indices=event.output_leaf_indices,
            output_view_offsets=event.output_view_offsets,
            reuses_ordinal=event.reuses_ordinal,
        )
        for event in measurement.allocation_trace
    )


def _plan_allocation_contract(
    measurement: TaskMeasurement,
) -> tuple[PlanAllocationABIStep, ...]:
    if measurement.allocation_contract is None:
        return ()
    return tuple(
        PlanAllocationABIStep(
            operation_index=step.operation_index,
            allocation_ordinal=step.allocation_ordinal,
            operation=step.operation.value,
            requested_bytes=step.requested_bytes,
            charged_bytes=step.charged_bytes,
            alignment_bytes=step.alignment_bytes,
            output_leaf_indices=step.output_leaf_indices,
            mutation_input_positions=step.mutation_input_positions,
            persistent_after_task=step.persistent_after_task,
        )
        for step in measurement.allocation_contract.steps
    )


def _logical_bytes(values: tuple[PlanObjectFootprint, ...]) -> int:
    return sum(item.logical_size_bytes for item in values)


def _footprint(
    object_spec: ObjectSpec, alias_spec: AliasGroupSpec
) -> PlanObjectFootprint:
    return PlanObjectFootprint(
        object_id=object_spec.object_id,
        alias_group_id=object_spec.alias_group_id,
        role=object_spec.role.value,
        logical_size_bytes=object_spec.size_bytes,
        allocation_size_bytes=alias_spec.size_bytes,
        offset_bytes=object_spec.offset_bytes,
    )


def _unique_allocation_bytes(values: tuple[PlanObjectFootprint, ...]) -> int:
    return sum(
        value.allocation_size_bytes
        for value in {item.alias_group_id: item for item in values}.values()
    )


__all__ = ["forward_stage_inventory", "training_stage_inventory"]
