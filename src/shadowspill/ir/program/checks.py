"""What a program must be true of, one named check at a time.

Each check takes the program and the identities indexed from it, and raises
through the same `require`/`fail` vocabulary the specs use. They run in the
order below because each one relies on what the ones above it established:
nothing is looked up by id before the ids are known to be unique.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..validation import (
    fail,
    index_unique,
    require,
    require_tuple,
)
from .enums import SharedResidencyPolicy

if TYPE_CHECKING:
    from . import ShadowSpillProgram


@dataclass(frozen=True, slots=True)
class Identities:
    """Every id in the program, indexed once for the checks that follow."""

    device_ids: Mapping[str, int]
    alias_ids: Mapping[str, int]
    object_ids: Mapping[str, int]
    profile_ids: Mapping[str, int]
    task_ids: Mapping[str, int]
    group_ids: Mapping[str, int]
    alias_by_id: Mapping[str, Any]
    object_by_id: Mapping[str, Any]


def require_shapes(program: ShadowSpillProgram) -> None:
    """Every collection is a tuple, and a program has at least one device."""

    for name, value in (
        ("devices", program.devices),
        ("alias_groups", program.alias_groups),
        ("objects", program.objects),
        ("profiles", program.profiles),
        ("tasks", program.tasks),
        ("task_alternative_groups", program.task_alternative_groups),
    ):
        require_tuple(value, f"program.{name}")
    require(bool(program.devices), "program.devices", "must not be empty")


def index_identities(program: ShadowSpillProgram) -> Identities:
    """Index every id, refusing duplicates, and keep the two lookups checks need."""

    device_ids = index_unique(
        (device.device_id for device in program.devices), "program.devices"
    )
    alias_ids = index_unique(
        (group.alias_group_id for group in program.alias_groups),
        "program.alias_groups",
    )
    object_ids = index_unique(
        (item.object_id for item in program.objects), "program.objects"
    )
    profile_ids = index_unique(
        (profile.profile_id for profile in program.profiles), "program.profiles"
    )
    task_ids = index_unique((task.task_id for task in program.tasks), "program.tasks")
    group_ids = index_unique(
        (group.group_id for group in program.task_alternative_groups),
        "program.task_alternative_groups",
    )
    alias_by_id = {
        alias_group.alias_group_id: alias_group for alias_group in program.alias_groups
    }
    object_by_id = {item.object_id: item for item in program.objects}
    return Identities(
        device_ids=device_ids,
        alias_ids=alias_ids,
        object_ids=object_ids,
        profile_ids=profile_ids,
        task_ids=task_ids,
        group_ids=group_ids,
        alias_by_id=alias_by_id,
        object_by_id=object_by_id,
    )


def check_alias_groups(program: ShadowSpillProgram, ids: Identities) -> None:
    """Every alias group names a device the program declares."""

    for index, alias_group in enumerate(program.alias_groups):
        require(
            alias_group.device_id in ids.device_ids,
            f"program.alias_groups[{index}].device_id",
            f"unknown device {alias_group.device_id!r}",
        )


def check_objects(program: ShadowSpillProgram, ids: Identities) -> None:
    """Every object sits inside the alias group it names."""

    for index, item in enumerate(program.objects):
        path = f"program.objects[{index}]"
        if item.alias_group_id not in ids.alias_by_id:
            fail(
                f"{path}.alias_group_id",
                f"unknown alias group {item.alias_group_id!r}",
            )
        extent = ids.alias_by_id[item.alias_group_id].size_bytes
        require(
            item.offset_bytes + item.size_bytes <= extent,
            path,
            f"object {item.object_id!r} ({item.role.value}, "
            f"{item.persistence.value}) spans "
            f"[{item.offset_bytes}, {item.offset_bytes + item.size_bytes}) of "
            f"alias group {item.alias_group_id!r}, which is {extent} bytes",
        )


def check_alternative_groups(
    program: ShadowSpillProgram, ids: Identities
) -> set[frozenset[str]]:
    """Each group's options name known tasks, and no task is in two groups.

    Returns the pairs of tasks that two options make mutually exclusive,
    which is what lets the task check accept two writers of one object.
    """

    tasks_in_groups: set[str] = set()
    exclusive_pairs: set[frozenset[str]] = set()
    for group_index, task_alternative_group in enumerate(
        program.task_alternative_groups
    ):
        group_path = f"program.task_alternative_groups[{group_index}]"
        require(
            task_alternative_group.group_id in ids.group_ids,
            group_path,
            "invalid group identity",
        )
        group_tasks: set[str] = set()
        option_tasks: list[set[str]] = []
        for option_index, option in enumerate(task_alternative_group.options):
            option_path = f"{group_path}.options[{option_index}]"
            active = set(option.active_task_ids)
            option_tasks.append(active)
            for task_id in active:
                require(
                    task_id in ids.task_ids,
                    f"{option_path}.active_task_ids",
                    f"unknown task {task_id!r}",
                )
                group_tasks.add(task_id)
            for alias_id in option.retained_alias_group_ids:
                require(
                    alias_id in ids.alias_ids,
                    f"{option_path}.retained_alias_group_ids",
                    f"unknown alias group {alias_id!r}",
                )
                require(
                    ids.alias_by_id[alias_id].shared_residency is None,
                    f"{option_path}.retained_alias_group_ids",
                    (
                        f"shared alias group {alias_id!r} is runtime-resident and "
                        "cannot be retained by a task-alternative option"
                    ),
                )
        overlap = tasks_in_groups & group_tasks
        require(
            not overlap,
            group_path,
            f"tasks occur in several task-alternative groups: {sorted(overlap)}",
        )
        tasks_in_groups.update(group_tasks)
        for left_index, left in enumerate(option_tasks):
            for right in option_tasks[left_index + 1 :]:
                for left_task in left - right:
                    for right_task in right - left:
                        exclusive_pairs.add(frozenset((left_task, right_task)))
    return exclusive_pairs


def check_tasks(
    program: ShadowSpillProgram,
    ids: Identities,
    exclusive_pairs: set[frozenset[str]],
) -> None:
    """Every task names known resources, depends on its producers, and writes alone."""

    produced_by: dict[str, list[str]] = {}
    seen_tasks: set[str] = set()
    for index, task in enumerate(program.tasks):
        path = f"program.tasks[{index}]"
        require(
            task.resource.device_id in ids.device_ids,
            f"{path}.resource.device_id",
            f"unknown device {task.resource.device_id!r}",
        )
        require(
            task.profile_id in ids.profile_ids,
            f"{path}.profile_id",
            f"unknown profile {task.profile_id!r}",
        )
        for dependency in task.dependencies:
            require(
                dependency in seen_tasks,
                f"{path}.dependencies",
                f"dependency {dependency!r} must precede the task",
            )
        for relation, values in (
            ("inputs", task.inputs),
            ("outputs", task.outputs),
        ):
            for object_id in values:
                require(
                    object_id in ids.object_ids,
                    f"{path}.{relation}",
                    f"unknown object {object_id!r}",
                )
        for output in task.outputs:
            output_alias = ids.object_by_id[output].alias_group_id
            output_sharing = ids.alias_by_id[output_alias].shared_residency
            if output_sharing not in {
                None,
                SharedResidencyPolicy.SHARED_WRITABLE_CAUSAL,
            }:
                fail(
                    f"{path}.outputs",
                    f"shared alias group {output_alias!r} uses "
                    f"{output_sharing.value!r} and cannot publish task outputs",
                )
            previous_writers = produced_by.setdefault(output, [])
            require(
                all(
                    frozenset((writer, task.task_id)) in exclusive_pairs
                    for writer in previous_writers
                ),
                f"{path}.outputs",
                f"object {output!r} has simultaneously active writers",
            )
            previous_writers.append(task.task_id)
        for input_id in task.inputs:
            producers = produced_by.get(input_id, [])
            if producers:
                require(
                    any(producer in task.dependencies for producer in producers),
                    f"{path}.dependencies",
                    (
                        f"must include a producer for input {input_id!r}; "
                        f"candidates are {producers}"
                    ),
                )
        for mutation_index, mutation in enumerate(task.mutations):
            require(
                mutation.object_id in task.inputs,
                f"{path}.mutations[{mutation_index}].object_id",
                "mutated object must be an input",
            )
            mutation_alias = ids.object_by_id[mutation.object_id].alias_group_id
            require(
                ids.alias_by_id[mutation_alias].shared_residency
                is not SharedResidencyPolicy.SHARED_READ_ONLY,
                f"{path}.mutations[{mutation_index}].object_id",
                f"shared alias group {mutation_alias!r} is read-only",
            )
        seen_tasks.add(task.task_id)
