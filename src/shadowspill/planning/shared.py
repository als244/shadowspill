"""CPU admission for sequential saved Plans that share model tensors."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from shadowspill.artifacts.plan import Plan
from shadowspill.artifacts.tensors import shared_model_aliases
from shadowspill.ir import MemoryLocation


@dataclass(frozen=True)
class SharedMemory:
    aliases: Mapping[str, Mapping[str, str]]
    fixed_slice_bytes: int
    execution_bytes: int
    spill_bytes: int


def shared_memory(primary: Plan, plans: Mapping[str, Plan]) -> SharedMemory:
    """Validate layouts and conservative joint capacity before opening pools.

    Model storage is shared by registration name. Other tensors remain private.
    Plans execute sequentially, so fixed GPU workspace takes the largest slice.
    Fixed provider overhead is conservatively summed across distinct Programs;
    no undocumented resource sharing is assumed.
    """
    named = {"": primary, **plans}
    if "" in plans or any(not isinstance(name, str) for name in plans):
        raise ValueError("additional Plan names must be nonempty strings")
    mappings: dict[str, dict[str, str]] = {"": {}}
    for name, plan in named.items():
        if not isinstance(plan, Plan):
            raise TypeError("every admitted plan must be a Plan")
        if plan.simulation_only_reasons:
            raise ValueError(
                f"Plan {name or 'default'} is simulation-only: "
                + "; ".join(plan.simulation_only_reasons)
            )
        if plan.memory_budgets != primary.memory_budgets:
            raise ValueError("shared Plans must declare the same memory budgets")
        if plan.program.setup != primary.program.setup:
            raise ValueError("shared Plans require the same build and target")
        if plan.program.devices != primary.program.devices:
            raise ValueError("shared Plans require the same logical devices")
        if name:
            if plan.program.bindings["kind"] != "forward":
                raise ValueError("additional Plans currently support forward only")
            mapping = shared_model_aliases(primary.program, plan.program)
            _read_only(plan, mapping)
            mappings[name] = mapping
    paths = [
        (name, p)
        for name, plan in named.items()
        for p in (
            (plan,)
            if plan._accumulation_plan is None
            else (plan, plan._accumulation_plan)
        )
    ]
    layouts = [plan.memory_layout for _, plan in paths]
    assert all(layout is not None for layout in layouts)
    fixed = max(layout.fixed_slice_bytes for layout in layouts if layout is not None)
    dynamic = max(
        layout.required_bytes - layout.fixed_slice_bytes
        for layout in layouts
        if layout is not None
    )
    unique = {plan.program.digest: plan for plan in named.values()}
    overhead = sum(plan.fixed_device_bytes for plan in unique.values())
    # These plans run in one worker. Its explicitly reserved headroom is shared;
    # provider allocations remain conservatively summed because their sharing
    # is not part of the execution contract.
    headrooms = [
        p.profiled_program.metadata.get("external_headroom_bytes", 0)
        for p in unique.values()
    ]
    overhead -= sum(headrooms) - max(headrooms)
    execution = fixed + dynamic + overhead
    spill = _spill_bytes(paths, mappings)
    if execution > primary.memory_budgets.execution_bytes:
        raise ValueError(
            f"shared Plans require {execution} GPU bytes including overhead; "
            f"budget is {primary.memory_budgets.execution_bytes}"
        )
    if spill > primary.memory_budgets.spill_bytes:
        raise ValueError(
            f"shared Plans require {spill} spill bytes; "
            f"budget is {primary.memory_budgets.spill_bytes}"
        )
    return SharedMemory(mappings, fixed, execution, spill)


def _read_only(plan: Plan, mapping: Mapping[str, str]) -> None:
    aliases = {o.object_id: o.alias_group_id for o in plan.program.objects}
    for task in plan._costed_program.selected_tasks(plan.selections):
        writes = (*task.outputs, *(m.object_id for m in task.mutations))
        if any(aliases[object_id] in mapping for object_id in writes):
            raise ValueError("additional forward Plan mutates shared model tensors")


def _spill_bytes(
    paths: Sequence[tuple[str, Plan]], mappings: Mapping[str, Mapping[str, str]]
) -> int:
    # At a call boundary, all other plans keep their persistent spill storage.
    # Subtract only initial retained leases guaranteed to be in the active
    # plan's peak already. This may overestimate; it never discounts a lease
    # merely because it exists only at the other end of the invocation.
    persistent = {}
    active = []
    for name, plan in paths:
        groups = {a.alias_group_id: a for a in plan.program.alias_groups}
        mapping = mappings[name]
        keys = {
            alias: ("", mapping[alias]) if alias in mapping else (name, alias)
            for alias in groups
        }
        entries = (*plan.schedule.initial_residency, *plan.program.final_residency)
        for entry in entries:
            if entry.location is MemoryLocation.SPILL:
                persistent[keys[entry.alias_group_id]] = groups[
                    entry.alias_group_id
                ].spill_size_bytes
        guaranteed = {
            keys[r.alias_group_id]
            for r in plan.schedule.initial_residency
            if r.location is MemoryLocation.SPILL
            and groups[r.alias_group_id].retain_spill_copy
        }
        active.append((plan, guaranteed))
    total = sum(persistent.values())
    return max(
        plan.timeline.spill_peak_bytes + total - sum(persistent[k] for k in included)
        for plan, included in active
    )
