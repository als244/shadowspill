"""Immutable, predecoded records used by repeated training execution."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from shadowspill.ir import ExecutionPlan, MemoryAction, MemoryActionKind, TaskSpec
from shadowspill.ir.schedule import first_use_initial_order
from shadowspill.pytorch.capture.artifacts import GraphArtifact
from shadowspill.pytorch.lowering.training import (
    LoweredTrainingProgram,
    optimizer_object_ids,
)
from shadowspill.pytorch.optimizer import OptimizerTaskArtifact
from shadowspill.runtime.failures import ExecutionTaskIdentity
from shadowspill.runtime.plan import (
    RuntimeBridge,
    TaskMemoryEnvelope,
    TaskPublication,
    actions_by_task,
)
from shadowspill.simulator import SimulationResult
from shadowspill.task.entrypoints import TaskEntrypoint

from .timing import TracedInvocation, TracedTask, alias_accesses


@dataclass(frozen=True, slots=True)
class ExecutionTaskRecord:
    """One selected task with all repeated-path relationships predecoded."""

    entrypoint: TaskEntrypoint
    #: What to call for this task. The entrypoint is neutral; this is not.
    artifact: GraphArtifact | OptimizerTaskArtifact | None
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
class PlanRun:
    """The training step's program, predecoded for repeated execution."""

    lowered: LoweredTrainingProgram
    plan: ExecutionPlan
    simulation: SimulationResult
    expected_task_seconds: Mapping[str, float]
    execution: tuple[ExecutionTaskRecord, ...]
    initial_fetches: tuple[str, ...]
    public_by_microbatch: tuple[tuple[str, ...], ...]
    #: Every alias group's objects in this run's program, which is what names
    #: the objects a task publishes.
    object_ids_by_alias: Mapping[str, tuple[str, ...]]
    initial_task_id: int | None = None
    caller_acquisition_handle: int = 0

    def traced_invocation(self) -> TracedInvocation:
        """This run in the terms a runtime trace is taken in."""

        return TracedInvocation(
            tasks=tuple(
                TracedTask(
                    record.entrypoint,
                    self.expected_task_seconds[record.task.task_id],
                    record.execution_ordinal,
                    record.semantic_name,
                )
                for record in self.execution
            ),
            actions=(
                tuple(
                    MemoryAction("task_000000", alias_id, MemoryActionKind.FETCH)
                    for alias_id in self.initial_fetches
                )
                + self.plan.schedule.actions
            ),
            simulation=self.simulation,
            alias_accesses=alias_accesses(
                self.plan.program,
                ((record.execution_ordinal, record.task) for record in self.execution),
            ),
        )


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
    optimizer_objects = optimizer_object_ids(
        lowered.gradients, lowered.optimizer_objects
    )
    identities = _execution_identities(entrypoints)
    execution = tuple(
        _build_task_record(
            entrypoint,
            task=tasks[entrypoint.task_id],
            actions=action_index.get(entrypoint.task_id, ()),
            input_aliases=aliases[entrypoint.task_id],
            object_ids_by_alias=object_ids,
            ephemeral_aliases=ephemeral,
            artifact=lowered.executables.get(entrypoint.task_id),
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
        initial_fetches=tuple(
            alias_group_id
            for alias_group_id in first_use_initial_order(plan.program, plan.schedule)
            if bridge.objects.requires_storage(alias_group_id)
        ),
        public_by_microbatch=_public_outputs(entrypoints, bridge),
        object_ids_by_alias=object_ids,
    )


def _build_task_record(
    entrypoint: TaskEntrypoint,
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
    artifact: GraphArtifact | OptimizerTaskArtifact | None,
) -> ExecutionTaskRecord:
    function = (
        functions[artifact.compatibility_digest]
        if isinstance(artifact, GraphArtifact)
        else None
    )
    argument_template = (
        tuple(artifact.example_arguments)
        if isinstance(artifact, GraphArtifact)
        and entrypoint.options.phase != "optimizer"
        else None
    )
    outputs = (
        ()
        if entrypoint.options.phase == "optimizer"
        else _forward_outputs(entrypoint, input_aliases, bridge)
    )
    gradient_outputs = _gradient_outputs(entrypoint, bridge)
    handoff_aliases = frozenset(
        bridge.objects.alias_for_object(item.source_object_id)
        for item in entrypoint.storage_handoffs
        if item.destination_object_id in task.outputs
    )
    execution_ordinal, semantic_name = identity
    publications = _task_publications(outputs, gradient_outputs)
    return ExecutionTaskRecord(
        entrypoint=entrypoint,
        artifact=artifact,
        task=task,
        input_aliases=input_aliases,
        input_storage_aliases=tuple(
            alias_id
            for alias_id in input_aliases
            if bridge.objects.requires_storage(alias_id)
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
        publications=publications,
        optimizer_argument_object_ids=tuple(
            optimizer_objects.get(name) for name in entrypoint.options.named_inputs
        ),
        handoff_source_aliases=handoff_aliases,
        dematerialize_aliases=tuple(
            item.alias_group_id
            for item in actions
            if item.kind in {MemoryActionKind.RELEASE, MemoryActionKind.EVICT}
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
    entrypoint: TaskEntrypoint,
    input_aliases: tuple[str, ...],
    bridge: RuntimeBridge,
) -> tuple[ForwardOutputRecord, ...]:
    result: list[ForwardOutputRecord] = []
    produced: set[str] = set()
    next_publication = 0
    replacement_leaves = set(entrypoint.replacement_output_leaves)
    for slot in entrypoint.output_slots:
        alias_id = bridge.objects.alias_for_object(slot.object_id)
        replace = slot.leaf_index in replacement_leaves
        adopt = (replace or alias_id not in input_aliases) and alias_id not in produced
        publication_ordinal = None
        if adopt:
            produced.add(alias_id)
            if bridge.objects.requires_storage(alias_id):
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
    entrypoint: TaskEntrypoint,
    bridge: RuntimeBridge,
) -> tuple[GradientOutputRecord, ...]:
    grouped: dict[str, tuple[str, list[int]]] = {}
    for slot in entrypoint.options.contribution_slots:
        alias_id = bridge.objects.alias_for_object(slot.object_id)
        grouped.setdefault(alias_id, (slot.object_id, []))[1].append(slot.leaf_index)
    result: list[GradientOutputRecord] = []
    next_publication = 0
    for alias_id, (object_id, indices) in grouped.items():
        publication_ordinal = None
        if bridge.objects.requires_storage(alias_id):
            publication_ordinal = next_publication
            next_publication += 1
        result.append(
            GradientOutputRecord(
                object_id, alias_id, tuple(indices), publication_ordinal
            )
        )
    return tuple(result)


def _task_publications(
    forward: tuple[ForwardOutputRecord, ...],
    gradients: tuple[GradientOutputRecord, ...],
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
                bridge.objects.alias_for_object(object_id) for object_id in task.inputs
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


def _execution_identities(
    entrypoints: tuple[TaskEntrypoint, ...],
) -> dict[str, tuple[int, str]]:
    result: dict[str, tuple[int, str]] = {}
    phase_ordinals: dict[str, int] = {}
    for execution_ordinal, entrypoint in enumerate(entrypoints):
        if (
            entrypoint.options.repetition is not None
            and entrypoint.options.stage_index is not None
        ):
            semantic_name = (
                f"microbatch_{entrypoint.options.repetition:04d}."
                f"stage_{entrypoint.options.stage_index:04d}."
                f"{entrypoint.options.phase}.{entrypoint.options.variant}"
            )
        else:
            phase_ordinal = phase_ordinals.get(entrypoint.options.phase, 0)
            phase_ordinals[entrypoint.options.phase] = phase_ordinal + 1
            semantic_name = f"{entrypoint.options.phase}.component_{phase_ordinal:04d}"
        result[entrypoint.task_id] = (execution_ordinal, semantic_name)
    return result


def _public_outputs(
    entrypoints: tuple[TaskEntrypoint, ...], bridge: RuntimeBridge
) -> tuple[tuple[str, ...], ...]:
    result: dict[int, tuple[str, ...]] = {}
    for entrypoint in entrypoints:
        if (
            entrypoint.options.phase != "forward"
            or entrypoint.options.repetition is None
        ):
            continue
        result[entrypoint.options.repetition] = tuple(
            bridge.objects.alias_for_object(
                next(
                    slot.object_id
                    for slot in entrypoint.output_slots
                    if slot.leaf_index == leaf_index
                )
            )
            for leaf_index in entrypoint.options.public_output_leaves
        )
    return tuple(result[index] for index in range(len(result)))


__all__ = [
    "ExecutionTaskRecord",
    "ForwardOutputRecord",
    "GradientOutputRecord",
    "PlanRun",
    "build_plan_run",
]
