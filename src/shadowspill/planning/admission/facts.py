"""Translate task lifetimes into the existing physical admission engine."""

from collections.abc import Mapping

from shadowspill.ir import ShadowSpillProgram, shared_residency_footprint
from shadowspill.planner.admission import (
    AdmissionFacts,
    StorageHandoff,
    TaskAdmissionSpec,
)
from shadowspill.task.allocations.events import TaskAllocationEvent

from .outputs import TaskOutputBinding
from .traces import (
    _anonymous_peak_extents,
    _task_allocation_steps,
    _workspace_peak_extents,
)


def build_admission_facts(
    program: ShadowSpillProgram,
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
            f"ShadowSpillProgram has {len(program.devices)} devices"
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
        # A boundary without a callable cannot make task-local allocations.
        # Keep the ownership/workspace checks below: an empty trace must not
        # silently admit storage attributed to a task that does no work.
        trace: tuple[TaskAllocationEvent, ...] = ()
        if task.requires_entrypoint:
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
                f"task {task.task_id} physical allocation trace and ShadowSpillProgram "
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
