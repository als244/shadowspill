"""The IR with its identifiers replaced by stable integer indices.

Every C entry point takes flat arrays rather than objects, so a program,
a schedule and an execution plan each have an indexed form: the same
facts, with string ids resolved to positions and rows flattened. The
indices are assigned once, deterministically, and every later array is
in that order, so the C side never looks anything up by name.
"""

from __future__ import annotations

from dataclasses import dataclass

from .execution import ExecutionPlan
from .program import (
    ObjectRole,
    Persistence,
    ResourceKind,
    ShadowSpillProgram,
    SharedResidencyPolicy,
)
from .schedule import MemoryActionKind, MemoryLocation, MemorySchedule

RESOURCE_KIND_CODE = {
    ResourceKind.COMPUTE: 0,
    ResourceKind.COMMUNICATION: 1,
    ResourceKind.CONTROL: 2,
}
OBJECT_ROLE_CODE = {value: index for index, value in enumerate(ObjectRole)}
PERSISTENCE_CODE = {value: index for index, value in enumerate(Persistence)}
SHARED_RESIDENCY_CODE = {
    None: 0,
    SharedResidencyPolicy.SHARED_READ_ONLY: 1,
    SharedResidencyPolicy.SHARED_WRITABLE_CAUSAL: 2,
    SharedResidencyPolicy.SHARED_WRITABLE_UNORDERED: 3,
}
MEMORY_LOCATION_CODE = {
    MemoryLocation.DEVICE: 0,
    MemoryLocation.SPILL: 1,
}
MEMORY_ACTION_CODE = {
    MemoryActionKind.RELEASE: 0,
    MemoryActionKind.EVICT: 1,
    MemoryActionKind.FETCH: 2,
    MemoryActionKind.WRITE_BACK: 3,
}


@dataclass(frozen=True, slots=True)
class IndexedProgram:
    """A program as flat arrays, with every identity replaced by its position.

    `index_program` returns it. A record's index is its position in the
    program, and every array is in that order. A ragged pair ``X_offsets, X``
    holds row ``i`` as ``X[X_offsets[i]:X_offsets[i + 1]]``, with one row per
    task or option.

    Attributes
    ----------
    device_ids, alias_group_ids, object_ids, profile_ids, task_ids : tuple of str
        Each record's identity, in program order.
    task_alternative_group_ids : tuple of str
        Each alternative group's identity.
    task_alternative_option_ids : tuple of str
        Each option's identity, group by group. An option's index is its
        position here.
    alias_device : tuple of int
        Each alias group's device.
    alias_size_bytes, alias_initial_version : tuple of int
        Each alias group's size and initial version.
    alias_retain_spill_copy : tuple of bool
        Whether each alias group retains its spill copy.
    alias_shared_residency : tuple of int
        Each alias group's shared-residency policy: 0 for none, then 1 to 3 for
        the members of `SharedResidencyPolicy` in declaration order.
    object_alias_group : tuple of int
        Each object's alias group.
    object_offset_bytes, object_size_bytes : tuple of int
        Each object's byte range in its group.
    object_role, object_persistence : tuple of int
        Each object's role and persistence, as the member's position in
        `ObjectRole` or `Persistence`.
    profile_runtime_ns, profile_workspace_bytes : tuple of int
        Each profile's duration and workspace.
    profile_compatibility_digests : tuple of str
        Each profile's contract identity.
    task_device : tuple of int
        Each task's device.
    task_resource_kind : tuple of int
        Each task's resource kind: 0 compute, 1 communication, 2 control.
    task_resource_lane : tuple of int
        Each task's lane.
    task_profile : tuple of int
        Each task's profile.
    task_phases : tuple of str
        Each task's phase label.
    task_requires_entrypoint : tuple of bool
        Whether each task requires an entrypoint.
    dependency_offsets, dependencies : tuple of int
        Each task's dependencies, as tasks.
    input_offsets, inputs, output_offsets, outputs : tuple of int
        Each task's inputs and outputs, as objects.
    mutation_offsets, mutation_objects : tuple of int
        Each task's mutations, as objects.
    mutation_version_deltas : tuple of int
        Each mutation's version delta, in the order of ``mutation_objects``.
    group_option_offsets : tuple of int
        Each group's options, as a range of option indices.
    option_active_task_offsets, option_active_tasks : tuple of int
        Each option's active tasks.
    option_retained_alias_offsets, option_retained_aliases : tuple of int
        Each option's retained alias groups.
    """

    device_ids: tuple[str, ...]
    alias_group_ids: tuple[str, ...]
    object_ids: tuple[str, ...]
    profile_ids: tuple[str, ...]
    task_ids: tuple[str, ...]
    task_alternative_group_ids: tuple[str, ...]
    task_alternative_option_ids: tuple[str, ...]
    alias_device: tuple[int, ...]
    alias_size_bytes: tuple[int, ...]
    alias_initial_version: tuple[int, ...]
    alias_retain_spill_copy: tuple[bool, ...]
    alias_shared_residency: tuple[int, ...]
    object_alias_group: tuple[int, ...]
    object_offset_bytes: tuple[int, ...]
    object_size_bytes: tuple[int, ...]
    object_role: tuple[int, ...]
    object_persistence: tuple[int, ...]
    profile_runtime_ns: tuple[int, ...]
    profile_workspace_bytes: tuple[int, ...]
    profile_compatibility_digests: tuple[str, ...]
    task_device: tuple[int, ...]
    task_resource_kind: tuple[int, ...]
    task_resource_lane: tuple[int, ...]
    task_profile: tuple[int, ...]
    task_phases: tuple[str, ...]
    task_requires_entrypoint: tuple[bool, ...]
    dependency_offsets: tuple[int, ...]
    dependencies: tuple[int, ...]
    input_offsets: tuple[int, ...]
    inputs: tuple[int, ...]
    output_offsets: tuple[int, ...]
    outputs: tuple[int, ...]
    mutation_offsets: tuple[int, ...]
    mutation_objects: tuple[int, ...]
    mutation_version_deltas: tuple[int, ...]
    group_option_offsets: tuple[int, ...]
    option_active_task_offsets: tuple[int, ...]
    option_active_tasks: tuple[int, ...]
    option_retained_alias_offsets: tuple[int, ...]
    option_retained_aliases: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class IndexedMemorySchedule:
    """A schedule as flat arrays over a program's indices.

    `index_memory_schedule` returns it. Alias groups and tasks are positions
    in the program, counting all its tasks.

    Attributes
    ----------
    initial_aliases, initial_locations : tuple of int
        Each initial residency entry's alias group and location: 0 for the
        device, 1 for spill.
    action_trigger_tasks, action_aliases : tuple of int
        Each action's trigger task and alias group.
    action_kinds : tuple of int
        Each action's kind: 0 release, 1 evict, 2 fetch, 3 write-back.
    final_aliases, final_locations : tuple of int
        Each final residency entry's alias group and location.
    """

    initial_aliases: tuple[int, ...]
    initial_locations: tuple[int, ...]
    action_trigger_tasks: tuple[int, ...]
    action_aliases: tuple[int, ...]
    action_kinds: tuple[int, ...]
    final_aliases: tuple[int, ...]
    final_locations: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class IndexedExecutionPlan:
    """An execution plan as flat arrays and values, as the runtime admits it.

    `index_execution_plan` returns it. Tasks, groups and options are indices
    into ``program``.

    Attributes
    ----------
    program : IndexedProgram
        The plan's program.
    schedule : IndexedMemorySchedule
        The plan's schedule.
    selection_groups, selection_options : tuple of int
        Each selection's group and option.
    entrypoint_tasks : tuple of int
        Each entrypoint's task.
    entrypoint_ids, entrypoint_executor_ids : tuple of str
        Each entrypoint's function and executor.
    entrypoint_contract_digests : tuple of str
        Each entrypoint's contract digest.
    device_budget_bytes, spill_budget_bytes, baseline_bytes : int
        As in `PhysicalAdmission`.
    external_headroom_bytes, slab_bytes, workspace_reserve_bytes : int
        As in `PhysicalAdmission`.
    spill_reservation_bytes, predicted_fragmentation_bytes : int
        As in `PhysicalAdmission`.
    reject_overbudget : bool
        As in `PhysicalAdmission`.
    predicted_device_peak_bytes, predicted_spill_peak_bytes : int
        The `PlanPrediction` peaks.
    predicted_makespan_ns : int
        The `PlanPrediction` makespan.
    """

    program: IndexedProgram
    schedule: IndexedMemorySchedule
    selection_groups: tuple[int, ...]
    selection_options: tuple[int, ...]
    entrypoint_tasks: tuple[int, ...]
    entrypoint_ids: tuple[str, ...]
    entrypoint_executor_ids: tuple[str, ...]
    entrypoint_contract_digests: tuple[str, ...]
    device_budget_bytes: int
    spill_budget_bytes: int
    baseline_bytes: int
    external_headroom_bytes: int
    reject_overbudget: bool
    slab_bytes: int
    workspace_reserve_bytes: int
    spill_reservation_bytes: int
    predicted_fragmentation_bytes: int
    predicted_device_peak_bytes: int
    predicted_spill_peak_bytes: int
    predicted_makespan_ns: int


def _flatten_strings(
    rows: tuple[tuple[str, ...], ...],
    indices: dict[str, int],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    offsets = [0]
    values: list[int] = []
    for row in rows:
        values.extend(indices[value] for value in row)
        offsets.append(len(values))
    return tuple(offsets), tuple(values)


def flatten_rows(
    rows: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Pack ragged rows into one values array and its row offsets.

    Row `i` is `values[offsets[i]:offsets[i + 1]]`. The indexed forms in
    both this package and the simulator are built this way, so it lives
    here, at the layer below.
    """

    offsets = [0]
    values: list[int] = []
    for row in rows:
        values.extend(row)
        offsets.append(len(values))
    return tuple(offsets), tuple(values)


def index_program(program: ShadowSpillProgram) -> IndexedProgram:
    """Flatten a program into arrays over its record positions.

    Parameters
    ----------
    program : ShadowSpillProgram
        The program.

    Returns
    -------
    IndexedProgram
        The same facts, in the program's order.
    """

    device_ids = tuple(item.device_id for item in program.devices)
    alias_ids = tuple(item.alias_group_id for item in program.alias_groups)
    object_ids = tuple(item.object_id for item in program.objects)
    profile_ids = tuple(item.profile_id for item in program.profiles)
    task_ids = tuple(item.task_id for item in program.tasks)
    group_ids = tuple(item.group_id for item in program.task_alternative_groups)
    options = tuple(
        option for group in program.task_alternative_groups for option in group.options
    )
    option_ids = tuple(option.option_id for option in options)
    device_index = {value: index for index, value in enumerate(device_ids)}
    alias_index = {value: index for index, value in enumerate(alias_ids)}
    object_index = {value: index for index, value in enumerate(object_ids)}
    profile_index = {value: index for index, value in enumerate(profile_ids)}
    task_index = {value: index for index, value in enumerate(task_ids)}
    dependency_offsets, dependencies = _flatten_strings(
        tuple(task.dependencies for task in program.tasks), task_index
    )
    input_offsets, inputs = _flatten_strings(
        tuple(task.inputs for task in program.tasks), object_index
    )
    output_offsets, outputs = _flatten_strings(
        tuple(task.outputs for task in program.tasks), object_index
    )
    mutation_offsets, mutation_objects = flatten_rows(
        tuple(
            tuple(object_index[item.object_id] for item in task.mutations)
            for task in program.tasks
        )
    )
    group_option_offsets = [0]
    for group in program.task_alternative_groups:
        group_option_offsets.append(group_option_offsets[-1] + len(group.options))
    option_active_task_offsets, option_active_tasks = _flatten_strings(
        tuple(option.active_task_ids for option in options), task_index
    )
    option_retained_alias_offsets, option_retained_aliases = _flatten_strings(
        tuple(option.retained_alias_group_ids for option in options), alias_index
    )
    return IndexedProgram(
        device_ids=device_ids,
        alias_group_ids=alias_ids,
        object_ids=object_ids,
        profile_ids=profile_ids,
        task_ids=task_ids,
        task_alternative_group_ids=group_ids,
        task_alternative_option_ids=option_ids,
        alias_device=tuple(
            device_index[item.device_id] for item in program.alias_groups
        ),
        alias_size_bytes=tuple(item.size_bytes for item in program.alias_groups),
        alias_initial_version=tuple(
            item.initial_version for item in program.alias_groups
        ),
        alias_retain_spill_copy=tuple(
            item.retain_spill_copy for item in program.alias_groups
        ),
        alias_shared_residency=tuple(
            SHARED_RESIDENCY_CODE[item.shared_residency]
            for item in program.alias_groups
        ),
        object_alias_group=tuple(
            alias_index[item.alias_group_id] for item in program.objects
        ),
        object_offset_bytes=tuple(item.offset_bytes for item in program.objects),
        object_size_bytes=tuple(item.size_bytes for item in program.objects),
        object_role=tuple(OBJECT_ROLE_CODE[item.role] for item in program.objects),
        object_persistence=tuple(
            PERSISTENCE_CODE[item.persistence] for item in program.objects
        ),
        profile_runtime_ns=tuple(item.runtime_ns for item in program.profiles),
        profile_workspace_bytes=tuple(
            item.workspace_bytes for item in program.profiles
        ),
        profile_compatibility_digests=tuple(
            item.compatibility_digest for item in program.profiles
        ),
        task_device=tuple(
            device_index[task.resource.device_id] for task in program.tasks
        ),
        task_resource_kind=tuple(
            RESOURCE_KIND_CODE[task.resource.kind] for task in program.tasks
        ),
        task_resource_lane=tuple(task.resource.lane for task in program.tasks),
        task_profile=tuple(profile_index[task.profile_id] for task in program.tasks),
        task_phases=tuple(task.phase for task in program.tasks),
        task_requires_entrypoint=tuple(
            task.requires_entrypoint for task in program.tasks
        ),
        dependency_offsets=dependency_offsets,
        dependencies=dependencies,
        input_offsets=input_offsets,
        inputs=inputs,
        output_offsets=output_offsets,
        outputs=outputs,
        mutation_offsets=mutation_offsets,
        mutation_objects=mutation_objects,
        mutation_version_deltas=tuple(
            mutation.version_delta
            for task in program.tasks
            for mutation in task.mutations
        ),
        group_option_offsets=tuple(group_option_offsets),
        option_active_task_offsets=option_active_task_offsets,
        option_active_tasks=option_active_tasks,
        option_retained_alias_offsets=option_retained_alias_offsets,
        option_retained_aliases=option_retained_aliases,
    )


def index_memory_schedule(
    program: ShadowSpillProgram,
    schedule: MemorySchedule,
) -> IndexedMemorySchedule:
    """Flatten a schedule into arrays over a program's record positions.

    Parameters
    ----------
    program : ShadowSpillProgram
        The program whose positions the indices are.
    schedule : MemorySchedule
        The schedule.

    Returns
    -------
    IndexedMemorySchedule
        The same facts, in the schedule's order.

    Raises
    ------
    KeyError
        If the schedule names an alias group or task the program lacks.
    """

    alias_index = {
        item.alias_group_id: index for index, item in enumerate(program.alias_groups)
    }
    task_index = {item.task_id: index for index, item in enumerate(program.tasks)}
    return IndexedMemorySchedule(
        initial_aliases=tuple(
            alias_index[item.alias_group_id] for item in schedule.initial_residency
        ),
        initial_locations=tuple(
            MEMORY_LOCATION_CODE[item.location] for item in schedule.initial_residency
        ),
        action_trigger_tasks=tuple(
            task_index[item.trigger_task_id] for item in schedule.actions
        ),
        action_aliases=tuple(
            alias_index[item.alias_group_id] for item in schedule.actions
        ),
        action_kinds=tuple(MEMORY_ACTION_CODE[item.kind] for item in schedule.actions),
        final_aliases=tuple(
            alias_index[item.alias_group_id] for item in schedule.final_residency
        ),
        final_locations=tuple(
            MEMORY_LOCATION_CODE[item.location] for item in schedule.final_residency
        ),
    )


def index_execution_plan(plan: ExecutionPlan) -> IndexedExecutionPlan:
    """Flatten an execution plan, its program and its schedule into arrays.

    Parameters
    ----------
    plan : ExecutionPlan
        The plan.

    Returns
    -------
    IndexedExecutionPlan
        The same facts, with admission and prediction as plain values.
    """

    indexed_program = index_program(plan.program)
    task_index = {
        task_id: index for index, task_id in enumerate(indexed_program.task_ids)
    }
    group_index = {
        group_id: index
        for index, group_id in enumerate(indexed_program.task_alternative_group_ids)
    }
    option_index: dict[tuple[str, str], int] = {}
    next_option = 0
    for group in plan.program.task_alternative_groups:
        for option in group.options:
            option_index[(group.group_id, option.option_id)] = next_option
            next_option += 1
    admission = plan.admission
    prediction = plan.prediction
    return IndexedExecutionPlan(
        program=indexed_program,
        schedule=index_memory_schedule(plan.program, plan.schedule),
        selection_groups=tuple(group_index[item.group_id] for item in plan.selections),
        selection_options=tuple(
            option_index[(item.group_id, item.option_id)] for item in plan.selections
        ),
        entrypoint_tasks=tuple(task_index[item.task_id] for item in plan.entrypoints),
        entrypoint_ids=tuple(item.entrypoint_id for item in plan.entrypoints),
        entrypoint_executor_ids=tuple(item.executor_id for item in plan.entrypoints),
        entrypoint_contract_digests=tuple(
            item.contract_digest for item in plan.entrypoints
        ),
        device_budget_bytes=admission.device_budget_bytes,
        spill_budget_bytes=admission.spill_budget_bytes,
        baseline_bytes=admission.baseline_bytes,
        external_headroom_bytes=admission.external_headroom_bytes,
        reject_overbudget=admission.reject_overbudget,
        slab_bytes=admission.slab_bytes,
        workspace_reserve_bytes=admission.workspace_reserve_bytes,
        spill_reservation_bytes=admission.spill_reservation_bytes,
        predicted_fragmentation_bytes=admission.predicted_fragmentation_bytes,
        predicted_device_peak_bytes=prediction.device_peak_bytes,
        predicted_spill_peak_bytes=prediction.spill_peak_bytes,
        predicted_makespan_ns=prediction.makespan_ns,
    )


__all__ = [
    "IndexedExecutionPlan",
    "IndexedMemorySchedule",
    "IndexedProgram",
    "index_execution_plan",
    "index_memory_schedule",
    "index_program",
]
