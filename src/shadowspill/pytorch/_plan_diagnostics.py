"""Deterministic construction of framework-facing planning diagnostics."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

from shadowspill.ir import (
    AliasGroupSpec,
    ExecutionPlan,
    ObjectSpec,
    Program,
    TaskSpec,
)

from .capture import AotGraphPair, GraphArtifact
from .lowering import LoweredForwardProgram
from .partition import PartitionedTrainingCapture, TrainingStage
from .profiling import TaskMeasurement
from .public import (
    PlanAllocationEvent,
    PlanGraphPair,
    PlanGraphProfile,
    PlanObjectFootprint,
    PlanTaskStage,
    PlanUniqueStage,
)
from .training_lowering import LoweredTrainingProgram, TrainingTaskEntrypoint


def training_stage_inventory(
    captures: tuple[PartitionedTrainingCapture, ...],
    lowered: LoweredTrainingProgram,
    execution_plan: ExecutionPlan,
    measurements: Mapping[str, TaskMeasurement],
) -> tuple[tuple[PlanTaskStage, ...], tuple[PlanUniqueStage, ...]]:
    """Describe task occurrences and all legal structural graph pairs."""

    program = lowered.program
    task_by_id = {item.task_id: item for item in program.tasks}
    entrypoint_by_key = {
        _entrypoint_key(item): item
        for item in lowered.entrypoints
        if item.stage_index is not None and item.variant is not None
    }
    selected_tasks = execution_plan.program.selected_tasks(execution_plan.selections)
    selected_ids = {item.task_id for item in selected_tasks}
    execution_ordinal = {
        item.task_id: index for index, item in enumerate(selected_tasks)
    }
    occurrence_keys: dict[tuple[int, int], str] = {}
    stage_by_key: dict[str, list[tuple[int, int, TrainingStage]]] = {}
    for microbatch, capture in enumerate(captures):
        for stage_index, stage in enumerate(capture.stages):
            structural_key = _stage_key(stage)
            occurrence_keys[(microbatch, stage_index)] = structural_key
            stage_by_key.setdefault(structural_key, []).append(
                (microbatch, stage_index, stage)
            )
    unique_id_by_key = {
        key: f"unique_stage_{index:04d}"
        for index, key in enumerate(sorted(stage_by_key))
    }
    chosen_by_occurrence: dict[tuple[int, int], str] = {}
    for entrypoint in lowered.entrypoints:
        if (
            entrypoint.microbatch is None
            or entrypoint.stage_index is None
            or entrypoint.variant is None
            or entrypoint.phase != "forward"
            or entrypoint.task_id not in selected_ids
        ):
            continue
        chosen_by_occurrence[(entrypoint.microbatch, entrypoint.stage_index)] = (
            entrypoint.variant
        )

    task_map: list[PlanTaskStage] = []
    auxiliary_ordinals: dict[str, int] = {}
    for entrypoint in lowered.entrypoints:
        artifact = entrypoint.artifact
        structural_abi = (
            artifact.compatibility_digest if artifact is not None else "opaque"
        )
        if entrypoint.microbatch is not None and entrypoint.stage_index is not None:
            occurrence = (entrypoint.microbatch, entrypoint.stage_index)
            structural_key = occurrence_keys[occurrence]
            stage_occurrence_id = (
                f"microbatch_{entrypoint.microbatch:04d}."
                f"stage_{entrypoint.stage_index:04d}"
            )
            unique_stage_id = unique_id_by_key[structural_key]
            chosen = chosen_by_occurrence.get(occurrence)
            semantic_name = (
                f"{stage_occurrence_id}.{entrypoint.phase}."
                f"{entrypoint.variant}"
            )
        else:
            stage_occurrence_id = None
            unique_stage_id = f"auxiliary_abi_{structural_abi[:16]}"
            chosen = None
            auxiliary_ordinal = auxiliary_ordinals.get(entrypoint.phase, 0)
            auxiliary_ordinals[entrypoint.phase] = auxiliary_ordinal + 1
            semantic_name = (
                f"{entrypoint.phase}.component_{auxiliary_ordinal:04d}"
            )
        ordinal = execution_ordinal.get(entrypoint.task_id)
        task_map.append(
            PlanTaskStage(
                task_id=entrypoint.task_id,
                execution_ordinal=ordinal,
                execution_task_id=(
                    None if ordinal is None else f"execution_{ordinal:06d}"
                ),
                semantic_name=semantic_name,
                phase=entrypoint.phase,
                microbatch=entrypoint.microbatch,
                stage_occurrence_id=stage_occurrence_id,
                unique_stage_id=unique_stage_id,
                structural_abi_key=structural_abi,
                graph_pair_variant=entrypoint.variant,
                chosen_graph_pair_variant=chosen,
                selected=entrypoint.task_id in selected_ids,
            )
        )

    unique_stages: list[PlanUniqueStage] = []
    for structural_key in sorted(stage_by_key):
        occurrences = stage_by_key[structural_key]
        microbatch, stage_index, representative = occurrences[0]
        graph_pairs: list[PlanGraphPair] = []
        for variant, pair in (
            ("save", representative.save_pair),
            ("recompute", representative.recompute_pair),
        ):
            forward_entrypoint = entrypoint_by_key[
                (microbatch, stage_index, variant, "forward")
            ]
            backward_entrypoint = entrypoint_by_key[
                (microbatch, stage_index, variant, "backward")
            ]
            graph_pairs.append(
                PlanGraphPair(
                    variant=variant,
                    recomputation=pair.recomputation,
                    saved_value_count=pair.saved_value_count,
                    specialized_unit_tangent_count=(
                        pair.specialized_unit_tangent_count
                    ),
                    forward=_graph_profile(
                        pair.forward,
                        "forward",
                        task_by_id[forward_entrypoint.task_id],
                        program,
                        measurements[pair.forward.compatibility_digest],
                    ),
                    backward=_graph_profile(
                        pair.backward,
                        "backward",
                        task_by_id[backward_entrypoint.task_id],
                        program,
                        measurements[pair.backward.compatibility_digest],
                    ),
                )
            )
        unique_stages.append(
            PlanUniqueStage(
                unique_stage_id=unique_id_by_key[structural_key],
                structural_key=structural_key,
                module_targets=tuple(
                    dict.fromkeys(
                        stage.example.module_target for _, _, stage in occurrences
                    )
                ),
                occurrence_count=len(occurrences),
                graph_pairs=tuple(graph_pairs),
            )
        )
    return tuple(task_map), tuple(unique_stages)


def forward_stage_inventory(
    lowered: LoweredForwardProgram,
    execution_plan: ExecutionPlan,
    measurements: Mapping[str, TaskMeasurement],
) -> tuple[tuple[PlanTaskStage, ...], tuple[PlanUniqueStage, ...]]:
    """Describe deduplicated inference stages and task occurrences."""

    task_by_id = {item.task_id: item for item in lowered.program.tasks}
    selected_tasks = execution_plan.program.selected_tasks(execution_plan.selections)
    selected_ids = {item.task_id for item in selected_tasks}
    execution_ordinal = {
        item.task_id: index for index, item in enumerate(selected_tasks)
    }
    keys = sorted({item.artifact.compatibility_digest for item in lowered.entrypoints})
    unique_id_by_key = {
        key: f"unique_stage_{index:04d}" for index, key in enumerate(keys)
    }
    task_map = tuple(
        PlanTaskStage(
            task_id=entrypoint.task_id,
            execution_ordinal=execution_ordinal.get(entrypoint.task_id),
            execution_task_id=(
                f"execution_{execution_ordinal[entrypoint.task_id]:06d}"
                if entrypoint.task_id in execution_ordinal
                else None
            ),
            semantic_name=f"stage_{index:04d}.forward.inference",
            phase="forward",
            microbatch=None,
            stage_occurrence_id=f"stage_{index:04d}",
            unique_stage_id=unique_id_by_key[entrypoint.artifact.compatibility_digest],
            structural_abi_key=entrypoint.artifact.compatibility_digest,
            graph_pair_variant="inference",
            chosen_graph_pair_variant="inference",
            selected=entrypoint.task_id in selected_ids,
        )
        for index, entrypoint in enumerate(lowered.entrypoints)
    )
    unique_stages: list[PlanUniqueStage] = []
    for key in keys:
        occurrences = tuple(
            item
            for item in lowered.entrypoints
            if item.artifact.compatibility_digest == key
        )
        representative = occurrences[0]
        unique_stages.append(
            PlanUniqueStage(
                unique_stage_id=unique_id_by_key[key],
                structural_key=key,
                module_targets=tuple(
                    dict.fromkeys(item.module_target for item in occurrences)
                ),
                occurrence_count=len(occurrences),
                graph_pairs=(
                    PlanGraphPair(
                        variant="inference",
                        recomputation=False,
                        saved_value_count=0,
                        specialized_unit_tangent_count=0,
                        forward=_graph_profile(
                            representative.artifact,
                            "forward",
                            task_by_id[representative.task_id],
                            lowered.program,
                            measurements[key],
                        ),
                        backward=None,
                    ),
                ),
            )
        )
    return task_map, tuple(unique_stages)


def _stage_key(stage: TrainingStage) -> str:
    payload = {
        "roots": list(stage.differentiable_output_indices),
        "save": _pair_key(stage.save_pair),
        "recompute": _pair_key(stage.recompute_pair),
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
) -> PlanGraphProfile:
    objects = {item.object_id: item for item in program.objects}
    aliases = {item.alias_group_id: item for item in program.alias_groups}

    def footprints(object_ids: tuple[str, ...]) -> tuple[PlanObjectFootprint, ...]:
        return tuple(
            _footprint(objects[object_id], aliases[objects[object_id].alias_group_id])
            for object_id in object_ids
        )

    inputs = footprints(task.inputs)
    mutations = footprints(tuple(item.object_id for item in task.mutations))
    outputs = footprints(task.outputs)
    profile = next(
        item for item in program.profiles if item.profile_id == task.profile_id
    )
    return PlanGraphProfile(
        direction=direction,
        structural_abi_key=artifact.compatibility_digest,
        representative_task_id=task.task_id,
        runtime_ns=measurement.runtime_ns,
        samples_ns=measurement.samples_ns,
        provenance=measurement.provenance,
        inputs=inputs,
        mutations=mutations,
        outputs=outputs,
        input_logical_bytes=sum(item.logical_size_bytes for item in inputs),
        input_allocation_bytes=_unique_allocation_bytes(inputs),
        mutation_logical_bytes=sum(item.logical_size_bytes for item in mutations),
        mutation_allocation_bytes=_unique_allocation_bytes(mutations),
        output_logical_bytes=sum(item.logical_size_bytes for item in outputs),
        output_allocation_bytes=_unique_allocation_bytes(outputs),
        workspace_requested_bytes=measurement.workspace_requested_bytes,
        workspace_charged_bytes=measurement.workspace_charged_bytes,
        task_workspace_bytes=profile.workspace_bytes,
        workspace_extent_bytes=measurement.workspace_extent_bytes,
        persistent_extent_bytes=measurement.persistent_extent_bytes,
        allocation_timeline=tuple(
            PlanAllocationEvent(
                allocation_ordinal=event.allocation_ordinal,
                operation=event.operation.value,
                requested_bytes=event.requested_bytes,
                charged_bytes=event.charged_bytes,
                output_leaf_indices=event.output_leaf_indices,
                reuses_ordinal=event.reuses_ordinal,
            )
            for event in measurement.allocation_trace
        ),
    )


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
