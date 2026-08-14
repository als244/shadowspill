"""Immutable, predecoded records used by repeated training execution."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from shadowspill.ir import ExecutionPlan, MemoryAction, MemoryActionKind, TaskSpec
from shadowspill.pytorch.capture.artifacts import GraphArtifact
from shadowspill.pytorch.lowering.training import (
    LoweredTrainingProgram,
    TrainingTaskEntrypoint,
)
from shadowspill.pytorch.runtime_adapter.bridge import RuntimeBridge, actions_by_task
from shadowspill.pytorch.runtime_adapter.failures import ExecutionTaskIdentity


@dataclass(frozen=True, slots=True)
class ExecutionTaskRecord:
    """One selected task with all repeated-path relationships predecoded."""

    entrypoint: TrainingTaskEntrypoint
    task: TaskSpec
    input_aliases: tuple[str, ...]
    actions: tuple[MemoryAction, ...]
    dense_task_id: int
    execution_ordinal: int
    semantic_name: str
    trace_label: str
    function: Callable[..., object] | None
    argument_template: tuple[object, ...] | None
    forward_outputs: tuple[ForwardOutputRecord, ...]
    gradient_outputs: tuple[GradientOutputRecord, ...]
    optimizer_argument_object_ids: tuple[str | None, ...]
    handoff_source_aliases: frozenset[str]
    dematerialize_aliases: tuple[str, ...]
    released_ephemeral: tuple[tuple[str, tuple[str, ...]], ...]
    native_handle: int = 0

    @property
    def identity(self) -> ExecutionTaskIdentity:
        """Return the three task identities used in public diagnostics."""

        return ExecutionTaskIdentity(
            execution_task_id=f"execution_{self.execution_ordinal:06d}",
            semantic_name=self.semantic_name,
            canonical_task_id=self.task.task_id,
        )


@dataclass(frozen=True, slots=True)
class ForwardOutputRecord:
    """One forward output leaf and its planned alias bundle."""

    leaf_index: int
    object_id: str
    alias_id: str
    adopt: bool
    replace: bool


@dataclass(frozen=True, slots=True)
class GradientOutputRecord:
    """All contribution leaves accumulated into one planned gradient."""

    object_id: str
    alias_id: str
    leaf_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PlanRun:
    """One immutable initial or recurrent training execution program."""

    lowered: LoweredTrainingProgram
    plan: ExecutionPlan
    expected_task_seconds: Mapping[str, float]
    execution: tuple[ExecutionTaskRecord, ...]
    initial_device_aliases: tuple[str, ...]
    public_by_microbatch: tuple[tuple[str, ...], ...]


def build_plan_run(
    lowered: LoweredTrainingProgram,
    plan: ExecutionPlan,
    *,
    bridge: RuntimeBridge,
    functions: Mapping[str, Callable[..., object]],
) -> PlanRun:
    """Predecode one selected plan into allocation-free repeated-path records."""

    tasks = {
        item.task_id: item for item in plan.program.selected_tasks(plan.selections)
    }
    profiles = {item.profile_id: item for item in plan.program.profiles}
    entrypoints = tuple(item for item in lowered.entrypoints if item.task_id in tasks)
    action_index = actions_by_task(plan.schedule.actions)
    aliases = _input_aliases(tasks, bridge)
    object_ids = _object_ids_by_alias(plan)
    ephemeral = _ephemeral_aliases(plan)
    optimizer_objects = _optimizer_objects(lowered)
    identities = _execution_identities(entrypoints)
    execution = tuple(
        _build_task_record(
            entrypoint,
            task=tasks[entrypoint.task_id],
            actions=action_index.get(entrypoint.task_id, ()),
            input_aliases=aliases[entrypoint.task_id],
            object_ids_by_alias=object_ids,
            ephemeral_aliases=ephemeral,
            optimizer_objects=optimizer_objects,
            identity=identities[entrypoint.task_id],
            bridge=bridge,
            functions=functions,
        )
        for entrypoint in entrypoints
    )
    return PlanRun(
        lowered=lowered,
        plan=plan,
        expected_task_seconds={
            task_id: profiles[task.profile_id].runtime_ns / 1e9
            for task_id, task in tasks.items()
        },
        execution=execution,
        initial_device_aliases=tuple(
            item.alias_group_id
            for item in plan.schedule.initial_residency
            if item.location.value == "device"
        ),
        public_by_microbatch=_public_outputs(entrypoints, bridge),
    )


def _build_task_record(
    entrypoint: TrainingTaskEntrypoint,
    *,
    task: TaskSpec,
    actions: tuple[MemoryAction, ...],
    input_aliases: tuple[str, ...],
    object_ids_by_alias: Mapping[str, tuple[str, ...]],
    ephemeral_aliases: frozenset[str],
    optimizer_objects: Mapping[str, str],
    identity: tuple[int, str],
    bridge: RuntimeBridge,
    functions: Mapping[str, Callable[..., object]],
) -> ExecutionTaskRecord:
    artifact = entrypoint.artifact
    function = (
        functions[artifact.compatibility_digest]
        if isinstance(artifact, GraphArtifact)
        else None
    )
    argument_template = (
        tuple(artifact.example_arguments)
        if isinstance(artifact, GraphArtifact) and entrypoint.phase != "optimizer"
        else None
    )
    outputs = _forward_outputs(entrypoint, input_aliases, bridge)
    handoff_aliases = frozenset(
        bridge.alias_for_object(item.source_object_id)
        for item in entrypoint.storage_handoffs
        if item.destination_object_id in task.outputs
    )
    execution_ordinal, semantic_name = identity
    return ExecutionTaskRecord(
        entrypoint=entrypoint,
        task=task,
        input_aliases=input_aliases,
        actions=actions,
        dense_task_id=int(entrypoint.task_id.removeprefix("task_")),
        execution_ordinal=execution_ordinal,
        semantic_name=semantic_name,
        trace_label=f"execution_{execution_ordinal:06d}.{semantic_name}",
        function=function,
        argument_template=argument_template,
        forward_outputs=outputs,
        gradient_outputs=_gradient_outputs(entrypoint, bridge),
        optimizer_argument_object_ids=tuple(
            optimizer_objects.get(name) for name in entrypoint.optimizer_binding_names
        ),
        handoff_source_aliases=handoff_aliases,
        dematerialize_aliases=tuple(
            item.alias_group_id
            for item in actions
            if item.kind in {MemoryActionKind.RELEASE, MemoryActionKind.OFFLOAD}
            and item.alias_group_id not in handoff_aliases
        ),
        released_ephemeral=tuple(
            (item.alias_group_id, object_ids_by_alias[item.alias_group_id])
            for item in actions
            if item.kind is MemoryActionKind.RELEASE
            and item.alias_group_id in ephemeral_aliases
        ),
    )


def _forward_outputs(
    entrypoint: TrainingTaskEntrypoint,
    input_aliases: tuple[str, ...],
    bridge: RuntimeBridge,
) -> tuple[ForwardOutputRecord, ...]:
    result: list[ForwardOutputRecord] = []
    produced: set[str] = set()
    replacement_leaves = set(entrypoint.replacement_output_leaves)
    for slot in entrypoint.output_slots:
        alias_id = bridge.alias_for_object(slot.object_id)
        replace = slot.leaf_index in replacement_leaves
        adopt = (replace or alias_id not in input_aliases) and alias_id not in produced
        if adopt:
            produced.add(alias_id)
        result.append(
            ForwardOutputRecord(
                slot.leaf_index,
                slot.object_id,
                alias_id,
                adopt,
                replace,
            )
        )
    return tuple(result)


def _gradient_outputs(
    entrypoint: TrainingTaskEntrypoint,
    bridge: RuntimeBridge,
) -> tuple[GradientOutputRecord, ...]:
    grouped: dict[str, tuple[str, list[int]]] = {}
    for slot in entrypoint.gradient_output_slots:
        alias_id = bridge.alias_for_object(slot.object_id)
        grouped.setdefault(alias_id, (slot.object_id, []))[1].append(slot.leaf_index)
    return tuple(
        GradientOutputRecord(object_id, alias_id, tuple(indices))
        for alias_id, (object_id, indices) in grouped.items()
    )


def _input_aliases(
    tasks: Mapping[str, TaskSpec], bridge: RuntimeBridge
) -> dict[str, tuple[str, ...]]:
    return {
        task_id: tuple(
            dict.fromkeys(
                bridge.alias_for_object(object_id) for object_id in task.inputs
            )
        )
        for task_id, task in tasks.items()
    }


def _object_ids_by_alias(plan: ExecutionPlan) -> dict[str, tuple[str, ...]]:
    return {
        group.alias_group_id: tuple(
            item.object_id
            for item in plan.program.objects
            if item.alias_group_id == group.alias_group_id
        )
        for group in plan.program.alias_groups
    }


def _ephemeral_aliases(plan: ExecutionPlan) -> frozenset[str]:
    initial = {item.alias_group_id for item in plan.schedule.initial_residency}
    return frozenset(
        item.alias_group_id
        for item in plan.program.alias_groups
        if item.alias_group_id not in initial
    )


def _optimizer_objects(lowered: LoweredTrainingProgram) -> dict[str, str]:
    return {
        **{item.parameter_name: item.parameter_object_id for item in lowered.gradients},
        **{
            f"gradient.{item.parameter_name}": item.gradient_object_id
            for item in lowered.gradients
        },
        **{item.name: item.object_id for item in lowered.optimizer_objects},
    }


def _execution_identities(
    entrypoints: tuple[TrainingTaskEntrypoint, ...],
) -> dict[str, tuple[int, str]]:
    result: dict[str, tuple[int, str]] = {}
    phase_ordinals: dict[str, int] = {}
    for execution_ordinal, entrypoint in enumerate(entrypoints):
        if entrypoint.microbatch is not None and entrypoint.stage_index is not None:
            semantic_name = (
                f"microbatch_{entrypoint.microbatch:04d}."
                f"stage_{entrypoint.stage_index:04d}."
                f"{entrypoint.phase}.{entrypoint.variant}"
            )
        else:
            phase_ordinal = phase_ordinals.get(entrypoint.phase, 0)
            phase_ordinals[entrypoint.phase] = phase_ordinal + 1
            semantic_name = f"{entrypoint.phase}.component_{phase_ordinal:04d}"
        result[entrypoint.task_id] = (execution_ordinal, semantic_name)
    return result


def _public_outputs(
    entrypoints: tuple[TrainingTaskEntrypoint, ...], bridge: RuntimeBridge
) -> tuple[tuple[str, ...], ...]:
    result: dict[int, tuple[str, ...]] = {}
    for entrypoint in entrypoints:
        if entrypoint.phase != "forward" or entrypoint.microbatch is None:
            continue
        result[entrypoint.microbatch] = tuple(
            bridge.alias_for_object(
                next(
                    slot.object_id
                    for slot in entrypoint.output_slots
                    if slot.leaf_index == leaf_index
                )
            )
            for leaf_index in entrypoint.public_output_leaves
        )
    return tuple(result[index] for index in range(len(result)))


__all__ = [
    "ExecutionTaskRecord",
    "ForwardOutputRecord",
    "GradientOutputRecord",
    "PlanRun",
    "build_plan_run",
]
