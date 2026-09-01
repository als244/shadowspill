"""Immutable, predecoded records used by repeated training execution."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from shadowspill.ir import ExecutionPlan, MemoryAction, MemoryActionKind, TaskSpec
from shadowspill.ir.schedule import first_use_initial_order
from shadowspill.pytorch.capture.artifacts import GraphArtifact
from shadowspill.pytorch.lowering.training import (
    LoweredTrainingProgram,
    TrainingTaskEntrypoint,
)
from shadowspill.pytorch.runtime_adapter.bridge import (
    RuntimeBridge,
    TaskMemoryEnvelope,
    TaskPublication,
    actions_by_task,
)
from shadowspill.pytorch.runtime_adapter.failures import ExecutionTaskIdentity
from shadowspill.simulator import SimulationResult


@dataclass(frozen=True, slots=True)
class ExecutionTaskRecord:
    """One selected task with all repeated-path relationships predecoded."""

    entrypoint: TrainingTaskEntrypoint
    task: TaskSpec
    input_aliases: tuple[str, ...]
    input_storage_aliases: tuple[str, ...]
    actions: tuple[MemoryAction, ...]
    task_index: int
    execution_ordinal: int
    semantic_name: str
    trace_label: str
    function: Callable[..., object] | None
    argument_template: tuple[object, ...] | None
    forward_outputs: tuple[ForwardOutputRecord, ...]
    gradient_outputs: tuple[GradientOutputRecord, ...]
    optimizer_outputs: tuple[OptimizerOutputRecord, ...]
    publications: tuple[TaskPublication, ...]
    optimizer_argument_object_ids: tuple[str | None, ...]
    handoff_source_aliases: frozenset[str]
    dematerialize_aliases: tuple[str, ...]
    released_ephemeral: tuple[tuple[str, tuple[str, ...]], ...]
    memory_envelope: TaskMemoryEnvelope
    task_handle: int = 0

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
    publication_ordinal: int | None


@dataclass(frozen=True, slots=True)
class GradientOutputRecord:
    """All contribution leaves accumulated into one planned gradient."""

    object_id: str
    alias_id: str
    leaf_indices: tuple[int, ...]
    publication_ordinal: int | None


@dataclass(frozen=True, slots=True)
class OptimizerOutputRecord:
    """One lazily created optimizer tensor published by an initial task."""

    name: str
    object_id: str
    alias_id: str
    publication_ordinal: int | None


@dataclass(frozen=True, slots=True)
class PlanRun:
    """One immutable initial or recurrent training execution program."""

    lowered: LoweredTrainingProgram
    plan: ExecutionPlan
    simulation: SimulationResult
    expected_task_seconds: Mapping[str, float]
    execution: tuple[ExecutionTaskRecord, ...]
    initial_prefetches: tuple[str, ...]
    public_by_microbatch: tuple[tuple[str, ...], ...]
    initial_task_id: int | None = None
    caller_acquisition_handle: int = 0


def build_plan_run(
    lowered: LoweredTrainingProgram,
    plan: ExecutionPlan,
    simulation: SimulationResult,
    *,
    bridge: RuntimeBridge,
    functions: Mapping[str, Callable[..., object]],
    memory_envelopes: Mapping[str, TaskMemoryEnvelope],
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
            memory_envelope=memory_envelopes.get(
                entrypoint.task_id, TaskMemoryEnvelope()
            ),
        )
        for entrypoint in entrypoints
    )
    return PlanRun(
        lowered=lowered,
        plan=plan,
        simulation=simulation,
        expected_task_seconds={
            task_id: profiles[task.profile_id].runtime_ns / 1e9
            for task_id, task in tasks.items()
        },
        execution=execution,
        initial_prefetches=tuple(
            alias_group_id
            for alias_group_id in first_use_initial_order(plan.program, plan.schedule)
            if bridge.requires_storage(alias_group_id)
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
    memory_envelope: TaskMemoryEnvelope,
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
    outputs = (
        ()
        if entrypoint.phase == "optimizer"
        else _forward_outputs(entrypoint, input_aliases, bridge)
    )
    gradient_outputs = _gradient_outputs(entrypoint, bridge)
    optimizer_outputs = _optimizer_outputs_for_entrypoint(entrypoint, bridge)
    handoff_aliases = frozenset(
        bridge.alias_for_object(item.source_object_id)
        for item in entrypoint.storage_handoffs
        if item.destination_object_id in task.outputs
    )
    execution_ordinal, semantic_name = identity
    publications = _task_publications(outputs, gradient_outputs, optimizer_outputs)
    return ExecutionTaskRecord(
        entrypoint=entrypoint,
        task=task,
        input_aliases=input_aliases,
        input_storage_aliases=tuple(
            alias_id for alias_id in input_aliases if bridge.requires_storage(alias_id)
        ),
        actions=actions,
        task_index=int(entrypoint.task_id.removeprefix("task_")),
        execution_ordinal=execution_ordinal,
        semantic_name=semantic_name,
        trace_label=f"execution_{execution_ordinal:06d}.{semantic_name}",
        function=function,
        argument_template=argument_template,
        forward_outputs=outputs,
        gradient_outputs=gradient_outputs,
        optimizer_outputs=optimizer_outputs,
        publications=publications,
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
        memory_envelope=memory_envelope,
    )


def _forward_outputs(
    entrypoint: TrainingTaskEntrypoint,
    input_aliases: tuple[str, ...],
    bridge: RuntimeBridge,
) -> tuple[ForwardOutputRecord, ...]:
    result: list[ForwardOutputRecord] = []
    produced: set[str] = set()
    next_publication = 0
    replacement_leaves = set(entrypoint.replacement_output_leaves)
    for slot in entrypoint.output_slots:
        alias_id = bridge.alias_for_object(slot.object_id)
        replace = slot.leaf_index in replacement_leaves
        adopt = (replace or alias_id not in input_aliases) and alias_id not in produced
        publication_ordinal = None
        if adopt:
            produced.add(alias_id)
            if bridge.requires_storage(alias_id):
                publication_ordinal = next_publication
                next_publication += 1
        result.append(
            ForwardOutputRecord(
                slot.leaf_index,
                slot.object_id,
                alias_id,
                adopt,
                replace,
                publication_ordinal,
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
    result: list[GradientOutputRecord] = []
    next_publication = 0
    for alias_id, (object_id, indices) in grouped.items():
        publication_ordinal = None
        if bridge.requires_storage(alias_id):
            publication_ordinal = next_publication
            next_publication += 1
        result.append(
            GradientOutputRecord(
                object_id, alias_id, tuple(indices), publication_ordinal
            )
        )
    return tuple(result)


def _optimizer_outputs_for_entrypoint(
    entrypoint: TrainingTaskEntrypoint,
    bridge: RuntimeBridge,
) -> tuple[OptimizerOutputRecord, ...]:
    if entrypoint.phase != "optimizer":
        return ()
    if len(entrypoint.optimizer_output_names) != len(entrypoint.output_slots):
        raise ValueError("optimizer output names and tensor slots must align")
    result: list[OptimizerOutputRecord] = []
    next_publication = 0
    seen: set[str] = set()
    for name, slot in zip(
        entrypoint.optimizer_output_names,
        entrypoint.output_slots,
        strict=True,
    ):
        alias_id = bridge.alias_for_object(slot.object_id)
        publication_ordinal = None
        if alias_id not in seen:
            seen.add(alias_id)
            if bridge.requires_storage(alias_id):
                publication_ordinal = next_publication
                next_publication += 1
        result.append(
            OptimizerOutputRecord(
                name,
                slot.object_id,
                alias_id,
                publication_ordinal,
            )
        )
    return tuple(result)


def _task_publications(
    forward: tuple[ForwardOutputRecord, ...],
    gradients: tuple[GradientOutputRecord, ...],
    optimizer: tuple[OptimizerOutputRecord, ...],
) -> tuple[TaskPublication, ...]:
    """Return the one ordered publication table for this task phase."""

    indexed: list[tuple[int, TaskPublication]] = []
    indexed.extend(
        (
            item.publication_ordinal,
            TaskPublication(item.alias_id, replace_lease=item.replace),
        )
        for item in forward
        if item.adopt and item.publication_ordinal is not None
    )
    indexed.extend(
        (item.publication_ordinal, TaskPublication(item.alias_id))
        for item in gradients
        if item.publication_ordinal is not None
    )
    indexed.extend(
        (item.publication_ordinal, TaskPublication(item.alias_id))
        for item in optimizer
        if item.publication_ordinal is not None
    )
    indexed.sort(key=lambda item: item[0])
    if tuple(index for index, _item in indexed) != tuple(range(len(indexed))):
        raise ValueError("task publication ordinals must be contiguous and ordered")
    return tuple(item for _index, item in indexed)


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
    "OptimizerOutputRecord",
    "PlanRun",
    "build_plan_run",
]
