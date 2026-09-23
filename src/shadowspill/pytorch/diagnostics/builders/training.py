"""What a training plan report says about its stages, pairs and profiles."""

from collections.abc import Mapping
from dataclasses import dataclass

from shadowspill.ir import (
    ExecutionPlan,
    ShadowSpillProgram,
    TaskProfile,
    TaskSpec,
)
from shadowspill.planner.diagnostics.plan import (
    PlanGraphPair,
    PlanTaskStage,
    PlanUniqueStage,
)
from shadowspill.pytorch.capture.artifacts import GraphArtifact
from shadowspill.pytorch.graph_pairs import (
    DifferentiatedStage,
    GraphPairVariant,
    PartitionedTrainingCapture,
    saved_value_footprint,
)
from shadowspill.pytorch.lowering.profiles import ProfileMeasurementKey
from shadowspill.pytorch.lowering.training import (
    LoweredTrainingProgram,
)
from shadowspill.pytorch.optimizer import OptimizerTaskArtifact
from shadowspill.pytorch.profiling import TaskMeasurement
from shadowspill.step import StepDataOrdering
from shadowspill.task.entrypoints import TaskEntrypoint
from shadowspill.task.layout import (
    reconcile_compiled_task_layout,
)
from shadowspill.task.manifest import ExecutableTaskManifest

from .graphs import _graph_profile
from .keys import _entrypoint_key, _stage_key


@dataclass(frozen=True, slots=True)
class _TrainingInventoryIndex:
    program: ShadowSpillProgram
    task_by_id: Mapping[str, TaskSpec]
    profile_by_id: Mapping[str, TaskProfile]
    entrypoint_by_key: Mapping[tuple[int, int, str, str], TaskEntrypoint]
    selected_ids: frozenset[str]
    execution_ordinal: Mapping[str, int]
    occurrence_keys: Mapping[tuple[int, int], str]
    stages_by_key: Mapping[str, tuple[tuple[int, int, DifferentiatedStage], ...]]
    unique_id_by_key: Mapping[str, str]
    chosen_by_occurrence: Mapping[tuple[int, int], str]


def training_stage_inventory(
    captures: tuple[PartitionedTrainingCapture, ...],
    lowered: LoweredTrainingProgram,
    execution_plan: ExecutionPlan,
    measurements: Mapping[ProfileMeasurementKey, TaskMeasurement],
    manifests: Mapping[str, ExecutableTaskManifest],
    profiling_metadata_digests: tuple[str, ...] | None = None,
    *,
    data_ordering: StepDataOrdering,
) -> tuple[tuple[PlanTaskStage, ...], tuple[PlanUniqueStage, ...]]:
    """Describe task occurrences and every legal structural graph pair."""

    index = _index_training_inventory(captures, lowered, execution_plan)
    stage_counts = tuple(len(capture.stages) for capture in captures)
    task_map = _plan_task_stages(
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
            data_ordering,
            stage_counts,
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
            if entrypoint.options.stage_index is not None
            and entrypoint.options.variant is not None
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
            entrypoint.options.repetition is not None
            and entrypoint.options.stage_index is not None
            and entrypoint.options.variant is not None
            and entrypoint.options.phase == "forward"
            and entrypoint.task_id in selected_ids
        ):
            chosen[entrypoint.options.repetition, entrypoint.options.stage_index] = (
                entrypoint.options.variant
            )
    return chosen


def _training_task_stage(
    entrypoint: TaskEntrypoint,
    lowered: LoweredTrainingProgram,
    index: _TrainingInventoryIndex,
    measurements: Mapping[ProfileMeasurementKey, TaskMeasurement],
    manifests: Mapping[str, ExecutableTaskManifest],
    metadata_digests: tuple[str, ...] | None,
    auxiliary_ordinal: int,
) -> PlanTaskStage:
    artifact = lowered.executables.get(entrypoint.task_id)
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
        phase=entrypoint.options.phase,
        microbatch=entrypoint.options.repetition,
        stage_occurrence_id=occurrence[1],
        unique_stage_id=occurrence[2],
        structural_contract_key=structural_contract,
        semantic_contract_digest=contract_digests[0],
        executable_contract_digest=contract_digests[1],
        compiled_layout_digest=contract_digests[2],
        graph_pair_variant=entrypoint.options.variant,
        chosen_graph_pair_variant=occurrence[3],
        selected=entrypoint.task_id in index.selected_ids,
        profile_compatibility_digest=profile.compatibility_digest,
        profiling_metadata_digest=_metadata_for(entrypoint, metadata_digests),
    )


def _training_occurrence_identity(
    entrypoint: TaskEntrypoint,
    structural_contract: str,
    auxiliary_ordinal: int,
    index: _TrainingInventoryIndex,
) -> tuple[str, str | None, str, str | None]:
    if entrypoint.options.repetition is None or entrypoint.options.stage_index is None:
        return (
            f"{entrypoint.options.phase}.component_{auxiliary_ordinal:04d}",
            None,
            f"auxiliary_contract_{structural_contract[:16]}",
            None,
        )
    occurrence = entrypoint.options.repetition, entrypoint.options.stage_index
    microbatch, stage_index = occurrence
    stage_id = f"microbatch_{microbatch:04d}.stage_{stage_index:04d}"
    structural_key = index.occurrence_keys[occurrence]
    return (
        f"{stage_id}.{entrypoint.options.phase}.{entrypoint.options.variant}",
        stage_id,
        index.unique_id_by_key[structural_key],
        index.chosen_by_occurrence.get(occurrence),
    )


def _task_contract_digests(
    artifact: OptimizerTaskArtifact | None,
    entrypoint: TaskEntrypoint,
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
        layout.contract_digest,
        layout.compatibility_digest,
    )


def _training_unique_stage(
    structural_key: str,
    index: _TrainingInventoryIndex,
    measurements: Mapping[ProfileMeasurementKey, TaskMeasurement],
    manifests: Mapping[str, ExecutableTaskManifest],
    metadata_digests: tuple[str, ...] | None,
    data_ordering: StepDataOrdering,
    stage_counts: tuple[int, ...],
) -> PlanUniqueStage:
    occurrences = index.stages_by_key[structural_key]
    microbatch, stage_index, representative = occurrences[0]
    accumulates = not data_ordering.creates(
        microbatch, stage_index, stage_count=stage_counts[microbatch]
    )
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
        for option in representative.graph_pairs.options(accumulates=accumulates)
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
    entrypoint: TaskEntrypoint,
    measurements: Mapping[ProfileMeasurementKey, TaskMeasurement],
    metadata_digests: tuple[str, ...] | None,
) -> TaskMeasurement:
    metadata = _metadata_for(entrypoint, metadata_digests)
    measurement = measurements.get(
        (
            artifact.compatibility_digest,
            metadata,
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
    entrypoint: TaskEntrypoint,
    metadata_digests: tuple[str, ...] | None,
) -> str | None:
    if entrypoint.options.repetition is None or metadata_digests is None:
        return None
    return metadata_digests[entrypoint.options.repetition]


def _plan_task_stages(
    lowered: LoweredTrainingProgram,
    index: _TrainingInventoryIndex,
    measurements: Mapping[ProfileMeasurementKey, TaskMeasurement],
    manifests: Mapping[str, ExecutableTaskManifest],
    metadata_digests: tuple[str, ...] | None,
) -> tuple[PlanTaskStage, ...]:
    auxiliary_ordinals: dict[str, int] = {}
    tasks: list[PlanTaskStage] = []
    for entrypoint in lowered.entrypoints:
        auxiliary_ordinal = auxiliary_ordinals.get(entrypoint.options.phase, 0)
        tasks.append(
            _training_task_stage(
                entrypoint,
                lowered,
                index,
                measurements,
                manifests,
                metadata_digests,
                auxiliary_ordinal,
            )
        )
        if (
            entrypoint.options.repetition is None
            or entrypoint.options.stage_index is None
        ):
            auxiliary_ordinals[entrypoint.options.phase] = auxiliary_ordinal + 1
    return tuple(tasks)
