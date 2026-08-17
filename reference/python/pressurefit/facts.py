"""Framework-neutral facts derived once for one recomputation selection."""

from __future__ import annotations

from dataclasses import dataclass

from shadowspill.ir import (
    MemoryLocation,
    Program,
    RecomputationSelection,
    ResidencySpec,
    TaskSpec,
)
from shadowspill.planner.model import PressureFitInfeasibleError
from shadowspill.simulator import SimulationConfig


@dataclass(frozen=True, slots=True)
class PlanningFacts:
    program: Program
    selections: tuple[RecomputationSelection, ...]
    tasks: tuple[TaskSpec, ...]
    task_index: dict[str, int]
    alias_ids: tuple[str, ...]
    alias_index: dict[str, int]
    alias_sizes: tuple[int, ...]
    alias_devices: tuple[str, ...]
    alias_retain_spill_copy: tuple[bool, ...]
    initial_locations: tuple[MemoryLocation | None, ...]
    final_locations: tuple[MemoryLocation | None, ...]
    anchors: tuple[frozenset[int], ...]
    access_events: tuple[tuple[tuple[int, int], ...], ...]
    production_boundaries: tuple[frozenset[int], ...]
    write_boundaries: tuple[tuple[int, ...], ...]
    output_reservations: tuple[tuple[int, ...], ...]
    input_tasks: tuple[tuple[int, ...], ...]
    first_input_tasks: tuple[int, ...]
    object_capacity_by_device: dict[str, int]
    object_capacity_by_boundary: dict[str, tuple[int, ...]]
    task_ideal_end_ns: tuple[int, ...]

    @property
    def last_boundary(self) -> int:
        return len(self.tasks) - 1

    def profile_workspace(self, task: TaskSpec) -> int:
        profiles = {item.profile_id: item for item in self.program.profiles}
        return profiles[task.profile_id].workspace_bytes


def _residency_map(
    values: tuple[ResidencySpec, ...],
    *,
    field: str,
) -> dict[str, MemoryLocation]:
    result: dict[str, MemoryLocation] = {}
    for index, value in enumerate(values):
        if value.alias_group_id in result:
            raise ValueError(
                f"{field}[{index}] duplicates alias group {value.alias_group_id!r}"
            )
        result[value.alias_group_id] = value.location
    return result


def build_facts(
    program: Program,
    selections: tuple[RecomputationSelection, ...],
    initial_residency: tuple[ResidencySpec, ...],
    final_residency: tuple[ResidencySpec, ...],
    config: SimulationConfig,
) -> PlanningFacts:
    """Validate planner inputs and derive dense alias/task boundary facts."""

    tasks = program.selected_tasks(selections)
    task_index = {task.task_id: index for index, task in enumerate(tasks)}
    aliases = program.alias_groups
    alias_ids = tuple(item.alias_group_id for item in aliases)
    alias_index = {value: index for index, value in enumerate(alias_ids)}
    zero_aliases = {
        index for index, item in enumerate(aliases) if item.size_bytes == 0
    }
    initial = _residency_map(initial_residency, field="initial_residency")
    final = _residency_map(final_residency, field="final_residency")
    for field, values in (("initial_residency", initial), ("final_residency", final)):
        unknown = set(values) - set(alias_ids)
        if unknown:
            raise ValueError(f"{field} contains unknown aliases: {sorted(unknown)}")

    configured = {device.device_id: device for device in config.devices}
    program_devices = {device.device_id for device in program.devices}
    if set(configured) != program_devices:
        raise ValueError(
            "simulation config devices must exactly match the program devices: "
            f"expected {sorted(program_devices)}, got {sorted(configured)}"
        )

    object_alias = {
        item.object_id: alias_index[item.alias_group_id] for item in program.objects
    }
    profile_by_id = {profile.profile_id: profile for profile in program.profiles}
    anchor_sets: list[set[int]] = [set() for _ in aliases]
    accesses: list[set[tuple[int, int]]] = [set() for _ in aliases]
    productions: list[set[int]] = [set() for _ in aliases]
    writes: list[set[int]] = [set() for _ in aliases]
    reservations: list[tuple[int, ...]] = []
    consumers: list[list[int]] = [[] for _ in aliases]
    produced_aliases: set[int] = set()

    for alias_id, location in initial.items():
        alias_number = alias_index[alias_id]
        if location is MemoryLocation.DEVICE and alias_number not in zero_aliases:
            anchor_sets[alias_number].add(-1)

    ideal_time = 0
    ideal_end: list[int] = []
    for index, task in enumerate(tasks):
        input_aliases = {
            alias
            for object_id in task.inputs
            for alias in (object_alias[object_id],)
            if alias not in zero_aliases
        }
        output_aliases = {
            alias
            for object_id in task.outputs
            for alias in (object_alias[object_id],)
            if alias not in zero_aliases
        }
        mutation_aliases = {
            alias
            for mutation in task.mutations
            for alias in (object_alias[mutation.object_id],)
            if alias not in zero_aliases
        }
        for alias_number in input_aliases | mutation_aliases:
            anchor_sets[alias_number].add(index - 1)
            accesses[alias_number].add((index - 1, index))
        for alias_number in input_aliases:
            consumers[alias_number].append(index)
        for alias_number in output_aliases:
            anchor_sets[alias_number].add(index)
            accesses[alias_number].add((index, index))
            productions[alias_number].add(index)
            writes[alias_number].add(index)
            produced_aliases.add(alias_number)
        for alias_number in mutation_aliases:
            anchor_sets[alias_number].add(index)
            writes[alias_number].add(index)
        fresh_outputs = tuple(
            alias
            for alias in sorted(output_aliases)
            if alias not in input_aliases and alias not in mutation_aliases
        )
        reservations.append(fresh_outputs)
        ideal_time += profile_by_id[task.profile_id].runtime_ns
        ideal_end.append(ideal_time)

    for alias_id, location in final.items():
        alias_number = alias_index[alias_id]
        if location is MemoryLocation.DEVICE and alias_number not in zero_aliases:
            anchor_sets[alias_number].add(len(tasks) - 1)

    for task in tasks:
        for object_id in task.inputs:
            alias_number = object_alias[object_id]
            if alias_number in zero_aliases:
                continue
            if not anchor_sets[alias_number]:
                continue
            first_anchor = min(anchor_sets[alias_number])
            if first_anchor == -1 and alias_number not in produced_aliases:
                alias_id = alias_ids[alias_number]
                if initial.get(alias_id) is None:
                    raise ValueError(
                        f"input alias {alias_id!r} has no initial residency"
                    )

    object_capacity = {
        device_id: device.capacity_bytes for device_id, device in configured.items()
    }
    boundary_capacity = {
        device_id: [device.capacity_bytes] * (len(tasks) + 1)
        for device_id, device in configured.items()
    }
    for task_index_, task in enumerate(tasks):
        workspace = profile_by_id[task.profile_id].workspace_bytes
        device_id = task.resource.device_id
        capacity = object_capacity[device_id]
        if workspace > capacity:
            raise PressureFitInfeasibleError(
                f"task workspace {workspace} exceeds capacity "
                f"{capacity} on {device_id!r}",
                kind="workspace_capacity",
                device_id=device_id,
                boundary_task_id=task.task_id,
                required_bytes=workspace,
                capacity_bytes=capacity,
            )
        # Boundary ``task_index_ - 1`` is the point at which this task's
        # inputs, fresh outputs, and anonymous workspace must coexist.  Do
        # not subtract the largest workspace globally: workspace belonging
        # to another sequential task is not live at this boundary.
        boundary_capacity[device_id][task_index_] = capacity - workspace

    return PlanningFacts(
        program=program,
        selections=selections,
        tasks=tasks,
        task_index=task_index,
        alias_ids=alias_ids,
        alias_index=alias_index,
        alias_sizes=tuple(item.size_bytes for item in aliases),
        alias_devices=tuple(item.device_id for item in aliases),
        alias_retain_spill_copy=tuple(item.retain_spill_copy for item in aliases),
        initial_locations=tuple(initial.get(alias_id) for alias_id in alias_ids),
        final_locations=tuple(final.get(alias_id) for alias_id in alias_ids),
        anchors=tuple(frozenset(values) for values in anchor_sets),
        access_events=tuple(tuple(sorted(values)) for values in accesses),
        production_boundaries=tuple(frozenset(values) for values in productions),
        write_boundaries=tuple(tuple(sorted(values)) for values in writes),
        output_reservations=tuple(reservations),
        input_tasks=tuple(tuple(values) for values in consumers),
        first_input_tasks=tuple(
            min(
                (task for boundary, task in values if boundary == task - 1),
                default=len(tasks),
            )
            for values in accesses
        ),
        object_capacity_by_device=object_capacity,
        object_capacity_by_boundary={
            device_id: tuple(values)
            for device_id, values in boundary_capacity.items()
        },
        task_ideal_end_ns=tuple(ideal_end),
    )


__all__ = ["PlanningFacts", "build_facts"]
