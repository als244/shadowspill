"""CPU admission for sequential saved Plans sharing persistent operation state."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from shadowspill.artifacts.plan import Plan
from shadowspill.errors import PlanInfeasibleError
from shadowspill.ir import MemoryLocation

from .shared_state import StateLocation, shared_state
from .worker_memory import shared_fixed_overhead


@dataclass(frozen=True)
class SharedMemory:
    aliases: Mapping[str, Mapping[str, StateLocation]]
    host: Mapping[str, Mapping[str, StateLocation]]
    fixed_slice_bytes: int
    execution_bytes: int
    spill_bytes: int


def shared_memory(primary: Plan, plans: Mapping[str, Plan]) -> SharedMemory:
    """Validate layouts and conservative joint capacity before opening pools.

    Persistent model/optimizer/gradient state has one canonical owner. Other
    objects remain private. Host counters share the same logical inventory.
    Plans execute sequentially, so fixed GPU workspace takes the largest slice,
    and spill holds persistent state once beside the largest transient spill
    peak, plus whatever transient spill the Plans leave at their ends.
    Fixed overhead follows the worker's external/headroom/retained components;
    their maxima can come from different Plans. Measured communicator startup
    shares a reservation when one compatible calibration covers the worker's
    request union; other reservations remain additive, as during execution.
    """
    named = {"": primary, **plans}
    if "" in plans or any(not isinstance(name, str) for name in plans):
        raise ValueError("additional Plan names must be nonempty strings")
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
        if plan.program.parallelism != primary.program.parallelism:
            raise ValueError("shared Plans require the same logical parallelism")
        if name and plan.program.bindings["kind"] not in {
            "forward",
            "forward_backward",
            "step",
            "optim_step",
        }:
            raise ValueError("additional Plan has an unsupported operation kind")
    state = shared_state({name: plan.program for name, plan in named.items()})
    mappings = state.aliases
    for name, plan in plans.items():
        if (
            plan.program.bindings["kind"] == "forward"
            and plan.program.bindings.get("operation_mode") != "training"
        ):
            _read_only(plan, mappings[name])
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
    overhead = shared_fixed_overhead([plan for _, plan in paths])
    execution = fixed + dynamic + overhead
    spill = _spill_bytes(paths, mappings)
    if execution > primary.effective_execution_bytes:
        raise PlanInfeasibleError(
            f"shared Plans require {execution} GPU bytes including overhead; "
            f"effective allowance is {primary.effective_execution_bytes}",
            kind="shared_execution_memory",
            required_bytes=execution,
            capacity_bytes=primary.effective_execution_bytes,
        )
    if spill > primary.memory_budgets.spill_bytes:
        raise PlanInfeasibleError(
            f"shared Plans require {spill} spill bytes; "
            f"budget is {primary.memory_budgets.spill_bytes}",
            kind="shared_spill_memory",
            required_bytes=spill,
            capacity_bytes=primary.memory_budgets.spill_bytes,
        )
    return SharedMemory(mappings, state.host, fixed, execution, spill)


def _read_only(plan: Plan, mapping: Mapping[str, StateLocation]) -> None:
    aliases = {o.object_id: o.alias_group_id for o in plan.program.objects}
    for task in plan._costed_program.selected_tasks(plan.selections):
        writes = (*task.outputs, *(m.object_id for m in task.mutations))
        if any(aliases[object_id] in mapping for object_id in writes):
            raise ValueError("additional forward Plan mutates shared model tensors")


def _spill_bytes(
    paths: Sequence[tuple[str, Plan]],
    mappings: Mapping[str, Mapping[str, StateLocation]],
) -> int:
    # Plans run one at a time. A persistent lease (an alias the initial or
    # final residency places in spill, or one that retains its spill copy) is
    # held for the whole run, once per object, however many Plans name it. A
    # transient lease is held only while its Plan runs, except what a Plan
    # still holds when it ends, which stays held from then on. The simulator
    # classifies leases the same way for each Plan's transient peak.
    persistent = {}
    left = 0
    for name, plan in paths:
        groups = {a.alias_group_id: a for a in plan.program.alias_groups}
        mapping = mappings[name]
        keys = {alias: mapping.get(alias, (name, alias)) for alias in groups}
        entries = (*plan.schedule.initial_residency, *plan.program.final_residency)
        held = {e.alias_group_id for e in entries if e.location is MemoryLocation.SPILL}
        held |= {alias for alias, group in groups.items() if group.retain_spill_copy}
        for alias in held:
            persistent[keys[alias]] = groups[alias].spill_size_bytes
        left += plan.timeline.spill_transient_end_bytes
    transient = max(plan.timeline.spill_transient_peak_bytes for _, plan in paths)
    return sum(persistent.values()) + left + transient
