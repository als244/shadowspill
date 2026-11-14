"""Instantiate example/count variants using saved storage and task forms only."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import replace
from itertools import count
from typing import Any

from shadowspill.artifacts._values import plain
from shadowspill.artifacts.program import Program
from shadowspill.artifacts.state_inventory import program_state

from .bind import bind_training

_OBJECT_FIELDS = {
    "object_id",
    "source_object_id",
    "destination_object_id",
    "source",
    "stored",
    "padding",
    "destination",
    "mutations",
}


def _references(value: Any, identities: set[str], *, key: str = "") -> set[str]:
    if isinstance(value, str):
        return (
            {value} & identities if key in _OBJECT_FIELDS | {"saved_aliases"} else set()
        )
    if isinstance(value, Mapping):
        return set().union(
            *(_references(child, identities, key=name) for name, child in value.items())
        )
    if isinstance(value, (tuple, list)):
        return set().union(
            *(_references(child, identities, key=key) for child in value)
        )
    return set()


def _template(
    value: Any, objects: Mapping[str, str], aliases: Mapping[str, str], *, key: str = ""
) -> Any:
    """Rewrite only the documented object/alias-bearing template fields.

    Profile keys, variant names and literal metadata must never be rewritten
    merely because a string happens to equal an IR identity.
    """
    if isinstance(value, Mapping):
        return {
            name: _template(child, objects, aliases, key=name)
            for name, child in value.items()
        }
    if isinstance(value, (list, tuple)):
        if key == "saved_aliases":
            return [aliases.get(item, item) for item in value]
        if key == "mutations":
            return [objects.get(item, item) for item in value]
        return [_template(child, objects, aliases, key=key) for child in value]
    if key in _OBJECT_FIELDS and isinstance(value, str):
        return objects.get(value, value)
    return value


def _identities(existing: set[str], prefix: str) -> Iterator[str]:
    # Native execution binds the numeric suffix of canonical plan-local IDs.
    # Continue that namespace while preserving every original state identity.
    start = 1 + max(
        (
            int(name.removeprefix(prefix))
            for name in existing
            if name.startswith(prefix) and name.removeprefix(prefix).isdigit()
        ),
        default=-1,
    )
    return (f"{prefix}{value:06d}" for value in count(start))


def instantiate_examples(program: Program, template_indices: Sequence[int]) -> Program:
    """Select/repeat captured physical example forms without build or profiling.

    Each occurrence gets independent input/activation/output storage. Model,
    gradient, optimizer and registered state retain their existing identities.
    Missing fresh/additive task forms fail; this never compiles a missing form.
    The result still needs CPU planning, shared admission and capacity/coverage
    validation. Selecting a count alone does not establish a legal batch size.
    """
    saved = program.bindings
    templates = saved.get("training_templates")
    if saved.get("kind") not in ("step", "forward_backward") or not templates:
        raise ValueError("example instantiation requires saved training templates")
    indices = tuple(template_indices)
    if not indices or any(
        type(i) is not int or not 0 <= i < len(templates) for i in indices
    ):
        raise ValueError("template_indices must name at least one captured example")
    objects = {obj.object_id: obj for obj in program.objects}
    aliases = {alias.alias_group_id: alias for alias in program.alias_groups}
    by_alias: dict[str, list[str]] = {}
    for obj in program.objects:
        by_alias.setdefault(obj.alias_group_id, []).append(obj.object_id)
    object_ids, alias_ids = set(objects), set(aliases)
    example_aliases = []
    for index, template in enumerate(templates):
        refs = _references(template, object_ids)
        refs.update(slot["object_id"] for slot in saved["root_inputs"][index])
        refs.update(
            value for value in saved["public_outputs"][index] if value is not None
        )
        example_aliases.append(
            {objects[value].alias_group_id for value in refs}
            | _references(template, alias_ids)
        )
    # The canonical inventory owns state identity. Gradients used by complete
    # steps and optimizer-private scratch/controls also stay shared across counts.
    shared = {
        binding.alias_group_id
        for binding in program_state(program).bindings
        if binding.alias_group_id is not None
    }
    shared.update(
        objects[item["gradient_object_id"]].alias_group_id
        for item in saved["gradients"]
    )
    common_refs = _references(saved.get("optimizer_templates", ()), object_ids)
    common_refs.update(_references(saved.get("gradient_templates", ()), object_ids))
    shared.update(objects[value].alias_group_id for value in common_refs)
    shared.update(alias_ids - set().union(*example_aliases))
    new_aliases = _identities(alias_ids, "alias_")
    new_objects = _identities(object_ids, "object_")
    kept_objects = [obj for obj in program.objects if obj.alias_group_id in shared]
    kept_aliases = [
        alias for alias in program.alias_groups if alias.alias_group_id in shared
    ]
    bindings = plain(saved)
    bindings.update(
        num_examples=len(indices),
        training_templates=[],
        root_inputs=[],
        public_outputs=[],
        input_signatures=[],
        example_root_sources=[],
        fixed_inputs=[
            item
            for item in bindings.get("fixed_inputs", ())
            if objects[item["object_id"]].alias_group_id in shared
        ],
        entrypoints=[],
        task_names={},
        transient_outputs={},
        example_instances={
            "version": 1,
            "source_program": program.digest,
            "template_indices": list(indices),
        },
    )
    for key in ("ordering", "accumulate_gradients"):
        bindings.pop(key, None)
    # Collective descriptions are form-keyed, but Program validates them against
    # actual tasks. Restore them after the control-only intermediate is rebound.
    collectives = bindings.pop("collective_tasks", {})
    sources = {
        alias: plain(names)
        for alias, names in program.sources.items()
        if alias in shared
    }
    initial = {
        item.alias_group_id: item
        for item in program.initial_residency
        if item.alias_group_id in shared
    }
    final = {
        item.alias_group_id: item
        for item in program.final_residency
        if item.alias_group_id in shared
    }
    signatures = saved.get(
        "input_signatures", [saved["input_signature"]] * len(templates)
    )
    root_sources = saved.get(
        "example_root_sources", [saved["root_sources"]] * len(templates)
    )
    for destination, source in enumerate(indices):
        private = example_aliases[source] - shared
        alias_map = {alias: next(new_aliases) for alias in sorted(private)}
        object_map = {
            value: next(new_objects)
            for value in sorted(value for alias in private for value in by_alias[alias])
        }
        kept_aliases.extend(
            replace(aliases[alias], alias_group_id=alias_map[alias])
            for alias in sorted(private)
        )
        kept_objects.extend(
            replace(
                objects[value],
                object_id=mapped,
                alias_group_id=alias_map[objects[value].alias_group_id],
            )
            for value, mapped in sorted(object_map.items())
        )
        bindings["training_templates"].append(
            _template(templates[source], object_map, alias_map)
        )
        roots = _template(saved["root_inputs"][source], object_map, alias_map)
        bindings["root_inputs"].append(roots)
        bindings["public_outputs"].append(
            [object_map.get(value, value) for value in saved["public_outputs"][source]]
        )
        bindings["input_signatures"].append(plain(signatures[source]))
        bindings["example_root_sources"].append(plain(root_sources[source]))
        bindings["fixed_inputs"].extend(
            _template(item, object_map, alias_map)
            for item in saved.get("fixed_inputs", ())
            if item["object_id"] in object_map
        )
        for original, updated in alias_map.items():
            if original in program.sources:
                sources[updated] = plain(program.sources[original])
        for old, new in zip(saved["root_inputs"][source], roots, strict=True):
            alias = objects[old["object_id"]].alias_group_id
            if alias in private:
                name = root_sources[source][new["leaf_index"]]["name"]
                sources[alias_map[alias]] = [f"example[{destination}].{name}"]
        for entries, target in (
            (program.initial_residency, initial),
            (program.final_residency, final),
        ):
            for item in entries:
                if item.alias_group_id in private:
                    mapped = alias_map[item.alias_group_id]
                    target[mapped] = replace(item, alias_group_id=mapped)
    bindings["input_signature"] = bindings["input_signatures"][0]
    bindings["root_sources"] = bindings["example_root_sources"][0]
    # A control-only intermediate lets the existing binder assemble all fresh/
    # additive dependencies and optimizer frontiers without an alternate lowering.
    control = next(
        (task for task in program.tasks if task.profile_id == "invocation_start"), None
    )
    if control is None:
        raise ValueError("training Program has no invocation-start task")
    intermediate = replace(
        program,
        objects=tuple(kept_objects),
        alias_groups=tuple(kept_aliases),
        tasks=(control,),
        task_alternative_groups=(),
        initial_residency=tuple(initial.values()),
        final_residency=tuple(final.values()),
        bindings=bindings,
        sources=sources,
    )
    result = bind_training(
        intermediate,
        optimizer_ordering=saved.get("optimizer_ordering", "stage_interleaved"),
    )
    if collectives:
        active = {task.profile_id for task in result.tasks}
        result = replace(
            result,
            bindings={
                **result.bindings,
                "collective_tasks": {
                    k: v for k, v in collectives.items() if k in active
                },
            },
        )
    return result
