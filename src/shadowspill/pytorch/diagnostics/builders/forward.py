"""What a forward plan report says about its stages."""

from collections.abc import Mapping
from dataclasses import dataclass

from shadowspill.ir import (
    ExecutionPlan,
    TaskProfile,
    TaskSpec,
)
from shadowspill.planner.diagnostics.plan import (
    PlanGraphPair,
    PlanTaskStage,
    PlanUniqueStage,
)
from shadowspill.pytorch.lowering.forward import LoweredForwardProgram, TaskEntrypoint
from shadowspill.pytorch.profiling import TaskMeasurement
from shadowspill.task.layout import (
    reconcile_compiled_task_layout,
)
from shadowspill.task.manifest import ExecutableTaskManifest

from .graphs import _graph_profile


@dataclass(frozen=True, slots=True)
class _ForwardInventoryIndex:
    task_by_id: Mapping[str, TaskSpec]
    profile_by_id: Mapping[str, TaskProfile]
    selected_ids: frozenset[str]
    execution_ordinal: Mapping[str, int]
    unique_id_by_key: Mapping[str, str]


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
    artifact_key = lowered.executables[entrypoint.task_id].compatibility_digest
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
            lowered.executables[
                entrypoint.task_id
            ].storage_contract.compatibility_digest
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
        lowered.executables[representative.task_id],
        "forward",
        task,
        lowered.program,
        measurements[key],
        manifests[lowered.executables[representative.task_id].compatibility_digest],
    )
    return PlanUniqueStage(
        unique_stage_id=index.unique_id_by_key[key],
        structural_key=key,
        module_targets=tuple(
            dict.fromkeys(
                entrypoint.options.target or entrypoint.task_id
                for entrypoint in occurrences
            )
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
