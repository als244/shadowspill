"""Persistent object bindings exposed by compiled execution tasks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from shadowspill.ir import Program, TaskSpec
from shadowspill.planner import (
    AdmissionTopology,
    StorageHandoff,
    TaskAdmissionSpec,
    TaskAllocationStep,
    TaskAllocationStepKind,
)
from shadowspill.pytorch.profiling import (
    TaskAllocationEvent,
    TaskAllocationOperation,
)

from ...lowering.forward import TaskEntrypoint
from ...lowering.training import TrainingTaskEntrypoint


@dataclass(frozen=True, slots=True)
class TaskOutputBinding:
    """Map one returned tensor leaf to its persistent Program alias group."""

    leaf_index: int
    alias_group_id: str
    replacement: bool = False
    source_alias_group_id: str | None = None

    def __post_init__(self) -> None:
        if self.leaf_index < 0:
            raise ValueError("output leaf index must be non-negative")
        if not self.alias_group_id:
            raise ValueError("output alias group ID must be non-empty")
        if self.source_alias_group_id == self.alias_group_id:
            raise ValueError("storage handoff source and destination must differ")


def output_bindings_for_entrypoints(
    tasks: Sequence[TaskSpec],
    entrypoints: Sequence[TaskEntrypoint | TrainingTaskEntrypoint],
    alias_by_object: Mapping[str, str],
) -> dict[str, tuple[TaskOutputBinding, ...]]:
    """Describe which returned tensor allocations become persistent outputs."""

    task_by_id = {task.task_id: task for task in tasks}
    result: dict[str, tuple[TaskOutputBinding, ...]] = {}
    for entrypoint in entrypoints:
        task = task_by_id.get(entrypoint.task_id)
        if task is None:
            continue
        slots = (
            entrypoint.gradient_output_slots
            if isinstance(entrypoint, TrainingTaskEntrypoint)
            and entrypoint.phase == "backward"
            else entrypoint.output_slots
        )
        output_objects = set(task.outputs)
        replacement_leaves = set(entrypoint.replacement_output_leaves)
        handoff_by_leaf = {
            item.leaf_index: item for item in entrypoint.storage_handoffs
        }
        seen_aliases: set[str] = set()
        bindings: list[TaskOutputBinding] = []
        for slot in slots:
            replacement = slot.leaf_index in replacement_leaves
            if slot.object_id not in output_objects and not replacement:
                continue
            alias_id = alias_by_object[slot.object_id]
            if alias_id in seen_aliases:
                continue
            bindings.append(
                TaskOutputBinding(
                    slot.leaf_index,
                    alias_id,
                    replacement,
                    (
                        alias_by_object[
                            handoff_by_leaf[slot.leaf_index].source_object_id
                        ]
                        if slot.leaf_index in handoff_by_leaf
                        else None
                    ),
                )
            )
            seen_aliases.add(alias_id)
        result[task.task_id] = tuple(bindings)
    return result


def build_admission_topology(
    program: Program,
    *,
    execution_pool_bytes: int,
    object_capacity_bytes: int,
    output_bindings: Mapping[str, tuple[TaskOutputBinding, ...]] | None = None,
    workspace_extents_by_compatibility: Mapping[str, tuple[int, ...]] | None = None,
    allocation_traces_by_compatibility: Mapping[
        str, tuple[TaskAllocationEvent, ...]
    ]
    | None = None,
    alignment: int = 256,
) -> AdmissionTopology:
    """Normalize executable output ownership into planner admission facts."""

    if len(program.devices) != 1:
        raise ValueError(
            "one admission topology currently describes one execution pool; "
            f"Program has {len(program.devices)} devices"
        )
    alias_by_object = {
        item.object_id: item.alias_group_id for item in program.objects
    }
    alias_size = {
        item.alias_group_id: item.size_bytes for item in program.alias_groups
    }
    profile_by_id = {item.profile_id: item for item in program.profiles}
    profiled_extents = dict(workspace_extents_by_compatibility or {})
    profiled_traces = dict(allocation_traces_by_compatibility or {})
    bindings_by_task = dict(output_bindings or {})
    task_specs: list[TaskAdmissionSpec] = []
    for task in program.tasks:
        bindings = bindings_by_task.get(task.task_id, ())
        replacements = tuple(
            dict.fromkeys(
                item.alias_group_id
                for item in bindings
                if item.replacement and alias_size[item.alias_group_id] != 0
            )
        )
        handoffs = tuple(
            StorageHandoff(item.source_alias_group_id, item.alias_group_id)
            for item in bindings
            if item.source_alias_group_id is not None
        )
        handoff_destinations = {
            item.destination_alias_group_id for item in handoffs
        }
        fresh = dict.fromkeys(alias_by_object[item] for item in task.outputs)
        for binding in bindings:
            if not binding.replacement and binding.source_alias_group_id is None:
                fresh.setdefault(binding.alias_group_id, None)
        fresh_aliases = tuple(
            alias_id
            for alias_id in fresh
            if alias_id not in replacements
            and alias_id not in handoff_destinations
            and alias_size[alias_id] != 0
        )
        replacement_bytes = sum(alias_size[item] for item in replacements)
        profile = profile_by_id[task.profile_id]
        profiled_workspace = profile.workspace_bytes
        if replacement_bytes > profiled_workspace:
            raise ValueError(
                f"task {task.task_id} replacement bytes exceed its workspace "
                f"charge: replacements={replacement_bytes}, "
                f"workspace={profiled_workspace}"
            )
        anonymous_workspace = profiled_workspace - replacement_bytes
        workspace_extents = profiled_extents.get(profile.compatibility_digest)
        if workspace_extents is None:
            workspace_extents = (
                () if anonymous_workspace == 0 else (anonymous_workspace,)
            )
        else:
            measured_workspace = sum(workspace_extents)
            if measured_workspace > anonymous_workspace:
                raise ValueError(
                    f"task {task.task_id} workspace extents exceed its "
                    "anonymous workspace charge: "
                    f"extents={measured_workspace}, "
                    f"workspace={anonymous_workspace}"
                )
            residual = anonymous_workspace - measured_workspace
            if residual:
                workspace_extents = (
                    *workspace_extents,
                    *_unclassified_workspace_extents(
                        task,
                        residual,
                        alias_by_object=alias_by_object,
                        alias_size=alias_size,
                        persistent_destinations=(
                            set(replacements) | handoff_destinations
                        ),
                    ),
                )
        task_specs.append(
            TaskAdmissionSpec(
                task_id=task.task_id,
                workspace_extents=workspace_extents,
                fresh_output_aliases=fresh_aliases,
                replacement_aliases=replacements,
                storage_handoffs=handoffs,
                allocation_steps=_task_allocation_steps(
                    task.task_id,
                    profiled_traces.get(profile.compatibility_digest),
                    bindings,
                    persistent_aliases=set(fresh_aliases) | set(replacements),
                ),
            )
        )
    topology = AdmissionTopology(
        device_id=program.devices[0].device_id,
        pool_capacity_bytes=execution_pool_bytes,
        object_capacity_bytes=object_capacity_bytes,
        minimum_alignment=alignment,
        tasks=tuple(task_specs),
    )
    topology.validate(program)
    return topology


def _task_allocation_steps(
    task_id: str,
    trace: tuple[TaskAllocationEvent, ...] | None,
    bindings: tuple[TaskOutputBinding, ...],
    *,
    persistent_aliases: set[str],
) -> tuple[TaskAllocationStep, ...]:
    """Project one physical profile without making it a runtime ABI.

    The profiled order is used only by offline dynamic-pool admission.  Output
    leaves that do not become Program objects are released at task completion;
    persistent output allocations remain live for ownership publication.
    """

    if trace is None:
        return ()
    alias_by_leaf = {item.leaf_index: item.alias_group_id for item in bindings}
    live: dict[int, str | None] = {}
    steps: list[TaskAllocationStep] = []
    observed_aliases: set[str] = set()
    for event in trace:
        ordinal = event.allocation_ordinal
        if event.operation is TaskAllocationOperation.ALLOCATE:
            aliases = {
                alias_by_leaf[leaf]
                for leaf in event.output_leaf_indices
                if leaf in alias_by_leaf
            }
            if len(aliases) > 1:
                raise ValueError(
                    f"task {task_id} allocation {ordinal} backs distinct "
                    f"persistent aliases {sorted(aliases)!r}"
                )
            alias_id = next(iter(aliases), None)
            if alias_id is not None:
                observed_aliases.add(alias_id)
            live[ordinal] = alias_id
            steps.append(
                TaskAllocationStep(
                    ordinal,
                    TaskAllocationStepKind.ALLOCATE,
                    event.charged_bytes,
                    alias_id,
                    event.reuses_ordinal,
                )
            )
            continue
        if ordinal not in live:
            raise ValueError(
                f"task {task_id} allocation trace releases unknown ordinal "
                f"{ordinal}"
            )
        del live[ordinal]
        steps.append(TaskAllocationStep(ordinal, TaskAllocationStepKind.RELEASE))
    missing = persistent_aliases - observed_aliases
    unexpected = observed_aliases - persistent_aliases
    if missing or unexpected:
        raise ValueError(
            f"task {task_id} physical output bindings disagree with Program "
            f"ownership: missing={sorted(missing)!r}, "
            f"unexpected={sorted(unexpected)!r}"
        )
    # The profiler intentionally omits logical frees for returned tensors.
    # Unselected returned leaves are ordinary task-local allocations and end
    # at the task boundary; declared Program outputs remain live.
    steps.extend(
        TaskAllocationStep(ordinal, TaskAllocationStepKind.RELEASE)
        for ordinal, alias_id in live.items()
        if alias_id is None
    )
    return tuple(steps)


def _unclassified_workspace_extents(
    task: TaskSpec,
    residual_bytes: int,
    *,
    alias_by_object: Mapping[str, str],
    alias_size: Mapping[str, int],
    persistent_destinations: set[str],
) -> tuple[int, ...]:
    """Recover transient mutation extents hidden by a scalar task profile.

    Recurrent gradient accumulation keeps the existing gradient objects live
    while a compiled backward returns one contribution allocation per mutated
    alias. Training lowering charges those contributions in ``workspace_bytes``
    so the simulator sees the correct total overlap. They remain separate
    allocator requests and must not become one synthetic contiguous lease.

    Replacement and handoff destinations are admitted as persistent object
    generations elsewhere. Any remaining bytes are physical transition
    padding whose individual geometry is not otherwise represented.
    """

    mutation_aliases = tuple(
        dict.fromkeys(
            alias_by_object[item.object_id]
            for item in task.mutations
            if alias_by_object[item.object_id] not in persistent_destinations
            and alias_size[alias_by_object[item.object_id]] != 0
        )
    )
    mutation_extents = tuple(alias_size[item] for item in mutation_aliases)
    classified_bytes = sum(mutation_extents)
    if classified_bytes > residual_bytes:
        # The mutations are in-place or their transition storage is already
        # represented by another physical binding. Preserve the conservative
        # residual without inventing a partial decomposition.
        return (residual_bytes,)
    padding = residual_bytes - classified_bytes
    return (*mutation_extents, *((padding,) if padding else ()))


__all__ = [
    "TaskOutputBinding",
    "build_admission_topology",
    "output_bindings_for_entrypoints",
]
