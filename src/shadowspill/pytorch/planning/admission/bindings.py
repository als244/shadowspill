"""Persistent object bindings exposed by compiled execution tasks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from shadowspill.ir import Program, TaskSpec
from shadowspill.planner import AdmissionTopology, StorageHandoff, TaskAdmissionSpec

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
    bindings_by_task = dict(output_bindings or {})
    task_specs: list[TaskAdmissionSpec] = []
    for task in program.tasks:
        bindings = bindings_by_task.get(task.task_id, ())
        replacements = tuple(
            dict.fromkeys(
                item.alias_group_id for item in bindings if item.replacement
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
        profiled_workspace = profile_by_id[task.profile_id].workspace_bytes
        if replacement_bytes > profiled_workspace:
            raise ValueError(
                f"task {task.task_id} replacement bytes exceed its workspace "
                f"charge: replacements={replacement_bytes}, "
                f"workspace={profiled_workspace}"
            )
        task_specs.append(
            TaskAdmissionSpec(
                task_id=task.task_id,
                workspace_bytes=profiled_workspace - replacement_bytes,
                fresh_output_aliases=fresh_aliases,
                replacement_aliases=replacements,
                storage_handoffs=handoffs,
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


__all__ = [
    "TaskOutputBinding",
    "build_admission_topology",
    "output_bindings_for_entrypoints",
]
