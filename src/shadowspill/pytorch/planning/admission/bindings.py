"""Persistent object bindings exposed by compiled execution tasks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from shadowspill.ir import Program, TaskSpec, shared_residency_footprint
from shadowspill.planner import (
    AdmissionFacts,
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


def build_admission_facts(
    program: Program,
    *,
    execution_pool_bytes: int,
    object_capacity_bytes: int,
    output_bindings: Mapping[str, tuple[TaskOutputBinding, ...]] | None = None,
    allocation_traces_by_compatibility: Mapping[str, tuple[TaskAllocationEvent, ...]]
    | None = None,
    alignment: int = 256,
) -> AdmissionFacts:
    """Normalize executable output ownership into planner admission facts."""

    if len(program.devices) != 1:
        raise ValueError(
            "one admission facts currently describes one execution pool; "
            f"Program has {len(program.devices)} devices"
        )
    alias_by_object = {item.object_id: item.alias_group_id for item in program.objects}
    alias_size = {item.alias_group_id: item.size_bytes for item in program.alias_groups}
    shared = shared_residency_footprint(program)
    shared_execution_bytes = shared.for_device(program.devices[0].device_id)
    movable_pool_bytes = execution_pool_bytes - shared_execution_bytes
    movable_object_bytes = object_capacity_bytes - shared_execution_bytes
    if movable_pool_bytes <= 0 or movable_object_bytes <= 0:
        raise ValueError(
            "shared residency leaves no positive callable-owned capacity: "
            f"pool={execution_pool_bytes}, object={object_capacity_bytes}, "
            f"shared={shared_execution_bytes}"
        )
    profile_by_id = {item.profile_id: item for item in program.profiles}
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
        handoff_destinations = {item.destination_alias_group_id for item in handoffs}
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
        profile = profile_by_id[task.profile_id]
        profiled_workspace = profile.workspace_bytes
        try:
            trace = profiled_traces[profile.compatibility_digest]
        except KeyError as error:
            raise ValueError(
                f"task {task.task_id} lacks explicit physical allocation "
                f"evidence for profile {profile.compatibility_digest!r}"
            ) from error
        allocation_steps = _task_allocation_steps(
            task.task_id,
            trace,
            bindings,
            persistent_aliases=set(fresh_aliases) | set(replacements),
        )
        workspace_extents = _anonymous_peak_extents(allocation_steps)
        task_workspace_extents = _workspace_peak_extents(
            allocation_steps,
            transient_aliases=set(replacements),
        )
        if sum(task_workspace_extents) != profiled_workspace:
            raise ValueError(
                f"task {task.task_id} physical allocation trace and Program "
                "workspace disagree: "
                f"trace_peak={sum(task_workspace_extents)}, "
                f"program_workspace={profiled_workspace}, "
                f"phase={task.phase!r}, profile={profile.profile_id!r}, "
                f"compatibility={profile.compatibility_digest!r}, "
                f"profiled_workspace={profiled_workspace}, "
                f"fresh_aliases={fresh_aliases!r}, "
                f"replacement_aliases={replacements!r}, "
                f"workspace_extents={task_workspace_extents!r}"
            )
        task_specs.append(
            TaskAdmissionSpec(
                task_id=task.task_id,
                workspace_extents=workspace_extents,
                fresh_output_aliases=fresh_aliases,
                replacement_aliases=replacements,
                storage_handoffs=handoffs,
                allocation_steps=allocation_steps,
            )
        )
    facts = AdmissionFacts(
        device_id=program.devices[0].device_id,
        pool_capacity_bytes=movable_pool_bytes,
        object_capacity_bytes=movable_object_bytes,
        minimum_alignment=alignment,
        tasks=tuple(task_specs),
    )
    facts.validate(program)
    return facts


def _task_allocation_steps(
    task_id: str,
    trace: tuple[TaskAllocationEvent, ...],
    bindings: tuple[TaskOutputBinding, ...],
    *,
    persistent_aliases: set[str],
) -> tuple[TaskAllocationStep, ...]:
    """Project one physical profile without making it a runtime contract.

    The profiled order is used only by offline dynamic-pool admission.  Output
    leaves that do not become Program objects are released at task completion;
    persistent output allocations remain live for ownership publication.
    """

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
                f"task {task_id} allocation trace releases unknown ordinal {ordinal}"
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


def _anonymous_peak_extents(
    steps: tuple[TaskAllocationStep, ...],
) -> tuple[int, ...]:
    """Return the exact anonymous live-set peak encoded by one task trace."""

    return _workspace_peak_extents(steps, transient_aliases=set())


def _workspace_peak_extents(
    steps: tuple[TaskAllocationStep, ...],
    *,
    transient_aliases: set[str],
) -> tuple[int, ...]:
    """Return the live-set peak not charged as fresh Program output storage."""

    live: dict[int, int] = {}
    peak: tuple[int, ...] = ()
    live_bytes = 0
    peak_bytes = 0
    for step in steps:
        if step.kind is TaskAllocationStepKind.ALLOCATE:
            if (
                step.output_alias_group_id is None
                or step.output_alias_group_id in transient_aliases
            ):
                live[step.allocation_ordinal] = step.charged_bytes
                live_bytes += step.charged_bytes
        else:
            live_bytes -= live.pop(step.allocation_ordinal, 0)
        if live_bytes > peak_bytes:
            peak_bytes = live_bytes
            peak = tuple(sorted(live.values()))
    return peak


__all__ = [
    "TaskOutputBinding",
    "build_admission_facts",
    "output_bindings_for_entrypoints",
]
