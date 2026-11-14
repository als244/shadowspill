"""CPU count expansion of saved forward tasks, including output-only calls."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import replace
from typing import Any

from shadowspill.artifacts._values import plain
from shadowspill.artifacts.examples import input_groups, output_groups
from shadowspill.artifacts.program import Program
from shadowspill.artifacts.state_inventory import program_state
from shadowspill.ir import TaskSpec

from .example_storage import clone_storage, identities, references, remap


def _templates(program: Program) -> tuple[tuple[TaskSpec, ...], ...]:
    saved = program.bindings
    required = {
        "root_inputs",
        "public_outputs",
        "input_signature",
        "root_sources",
        "entrypoints",
        "output_tree",
        "output_literals",
    }
    if not required <= saved.keys() or not program.tasks:
        raise ValueError("example instantiation requires saved forward bindings")
    if program.task_alternative_groups:
        raise ValueError("forward count expansion requires resolved task alternatives")
    tasks = {task.task_id: task for task in program.tasks}
    groups = saved.get("forward_templates", (tuple(tasks),))
    flat = [name for group in groups for name in group]
    if (
        not groups
        or any(not group for group in groups)
        or len(flat) != len(tasks)
        or set(flat) != tasks.keys()
        or len(input_groups(saved)) != len(groups)
        or len(output_groups(saved)) != len(groups)
        or saved.get("num_examples", 1) != len(groups)
    ):
        raise ValueError("invalid saved forward example groups")
    return tuple(tuple(tasks[name] for name in group) for group in groups)


def _bind_tasks(
    template: Sequence[TaskSpec],
    entries: Mapping[str, Any],
    object_map: Mapping[str, str],
    alias_map: Mapping[str, str],
    new_ids: Iterator[str],
    previous: tuple[str, ...],
    repetition: int,
) -> tuple[list[TaskSpec], list[Any], tuple[str, ...]]:
    task_map = {task.task_id: next(new_ids) for task in template}
    tasks, bound_entries = [], []
    for task in template:
        dependencies = [
            task_map[name] for name in task.dependencies if name in task_map
        ]
        # Each example starts after all terminals of the preceding example. This
        # preserves shared buffer mutation and collective order, including when
        # a previously expanded example is selected in a different position.
        if not dependencies or any(name not in task_map for name in task.dependencies):
            dependencies.extend(previous)
        tasks.append(
            replace(
                task,
                task_id=task_map[task.task_id],
                dependencies=tuple(dict.fromkeys(dependencies)),
                inputs=tuple(object_map.get(value, value) for value in task.inputs),
                outputs=tuple(object_map.get(value, value) for value in task.outputs),
                mutations=tuple(
                    replace(
                        item, object_id=object_map.get(item.object_id, item.object_id)
                    )
                    for item in task.mutations
                ),
            )
        )
        if task.task_id in entries:
            entry = remap(entries[task.task_id], object_map, alias_map)
            entry["task_id"] = task_map[task.task_id]
            entry.setdefault("options", {})["repetition"] = repetition
            bound_entries.append(entry)
    depended = {name for task in template for name in task.dependencies}
    terminals = tuple(
        task_map[task.task_id] for task in template if task.task_id not in depended
    )
    return tasks, bound_entries, terminals


def instantiate_forward(program: Program, template_indices: Sequence[int]) -> Program:
    """Return grouped forward results without recapturing any task form."""
    templates = _templates(program)
    indices = tuple(template_indices)
    if not indices or any(
        type(i) is not int or not 0 <= i < len(templates) for i in indices
    ):
        raise ValueError("template_indices must name at least one captured example")
    saved = program.bindings
    objects = {obj.object_id: obj for obj in program.objects}
    entries = {entry["task_id"]: entry for entry in saved["entrypoints"]}
    roots, outputs = input_groups(saved), output_groups(saved)
    example_aliases = []
    for index, template in enumerate(templates):
        refs = {value for task in template for value in (*task.inputs, *task.outputs)}
        refs.update(
            references(
                [entries.get(task.task_id, {}) for task in template], set(objects)
            )
        )
        refs.update(slot["object_id"] for slot in roots[index])
        refs.update(value for value in outputs[index] if value is not None)
        example_aliases.append({objects[value].alias_group_id for value in refs})
    shared = {
        binding.alias_group_id
        for binding in program_state(program).bindings
        if binding.alias_group_id is not None
    }
    if any(
        objects[value].alias_group_id in shared
        for group in outputs
        for value in group
        if value is not None
    ):
        raise ValueError(
            "forward count outputs must own storage separate from model state"
        )
    shared.update(
        {alias.alias_group_id for alias in program.alias_groups}
        - set().union(*example_aliases)
    )
    private_groups = [group - shared for group in example_aliases]
    seen: set[str] = set()
    for private in private_groups:
        if private & seen:
            raise ValueError("forward examples share private storage")
        seen.update(private)
    storage = clone_storage(
        program, shared, [private_groups[index] for index in indices]
    )
    bindings = plain(saved)
    bindings.update(
        num_examples=len(indices),
        forward_templates=[],
        root_inputs=[],
        public_outputs=[],
        input_signatures=[],
        example_root_sources=[],
        entrypoints=[],
        task_names={},
        transient_outputs={},
        fixed_inputs=[
            item
            for item in saved.get("fixed_inputs", ())
            if objects[item["object_id"]].alias_group_id in shared
        ],
        example_instances={
            "version": 1,
            "source_program": program.digest,
            "template_indices": indices,
        },
    )
    signatures = saved.get(
        "input_signatures", (saved["input_signature"],) * len(templates)
    )
    root_sources = saved.get(
        "example_root_sources", (saved["root_sources"],) * len(templates)
    )
    tasks: list[TaskSpec] = []
    previous: tuple[str, ...] = ()
    new_ids = identities(set(), "task_")
    for destination, source in enumerate(indices):
        object_map, alias_map = (
            storage.object_maps[destination],
            storage.alias_maps[destination],
        )
        group, bound_entries, previous = _bind_tasks(
            templates[source],
            entries,
            object_map,
            alias_map,
            new_ids,
            previous,
            destination,
        )
        tasks.extend(group)
        bindings["forward_templates"].append([task.task_id for task in group])
        bindings["entrypoints"].extend(bound_entries)
        for number, (old, new) in enumerate(zip(templates[source], group, strict=True)):
            bindings["task_names"][new.task_id] = (
                f"example[{destination}].forward[{number}]"
            )
            if old.task_id in saved.get("transient_outputs", {}):
                bindings["transient_outputs"][new.task_id] = saved["transient_outputs"][
                    old.task_id
                ]
        bindings["root_inputs"].append(remap(roots[source], object_map, alias_map))
        bindings["public_outputs"].append(
            [
                object_map.get(value, value) if value is not None else None
                for value in outputs[source]
            ]
        )
        bindings["input_signatures"].append(signatures[source])
        bindings["example_root_sources"].append(root_sources[source])
        bindings["fixed_inputs"].extend(
            remap(item, object_map, alias_map)
            for item in saved.get("fixed_inputs", ())
            if item["object_id"] in object_map
        )
        for slot in roots[source]:
            alias = objects[slot["object_id"]].alias_group_id
            if alias in alias_map:
                name = root_sources[source][slot["leaf_index"]]["name"]
                storage.sources[alias_map[alias]] = [f"example[{destination}].{name}"]
    bindings["input_signature"], bindings["root_sources"] = (
        bindings["input_signatures"][0],
        bindings["example_root_sources"][0],
    )
    if "collective_tasks" in bindings:
        active = {task.profile_id for task in tasks}
        bindings["collective_tasks"] = {
            key: value
            for key, value in bindings["collective_tasks"].items()
            if key in active
        }
    return replace(
        program,
        tasks=tuple(tasks),
        objects=tuple(storage.objects),
        alias_groups=tuple(storage.aliases),
        initial_residency=tuple(storage.initial),
        final_residency=tuple(storage.final),
        bindings=bindings,
        sources=storage.sources,
    )
