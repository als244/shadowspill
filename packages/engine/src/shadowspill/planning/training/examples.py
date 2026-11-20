"""Instantiate example/count variants using saved storage and task forms only."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from shadowspill.artifacts._values import plain
from shadowspill.artifacts.program import Program
from shadowspill.artifacts.state_inventory import program_state

from ..example_storage import clone_storage, references, remap
from .bind import bind_training


def instantiate_examples(program: Program, template_indices: Sequence[int]) -> Program:
    """Derive a Program that runs captured example forms in a chosen sequence.

    Each entry of ``template_indices`` adds one example of the form captured at
    that index: ``[0, 0, 0]`` runs the first form three times, ``[1, 0]`` the
    second and then the first. Each example gets its own input, activation and
    output storage; model, gradient, optimizer and other registered state stay
    shared. Task forms and their profile keys are unchanged and nothing is
    compiled, so existing profiles apply. A derived forward Program returns a
    tuple of one result per example.

    Parameters
    ----------
    program : Program
        A complete-step, forward/backward or forward Program, derived already
        or not.
    template_indices : sequence of int
        The captured form of each example; at least one.

    Returns
    -------
    Program
        The derived Program.

    Raises
    ------
    ValueError
        If ``program`` has no saved example forms, or an index names no
        captured example.
    """
    saved = program.bindings
    if saved.get("kind") == "forward":
        from ..forward_examples import instantiate_forward

        return instantiate_forward(program, template_indices)
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
    object_ids, alias_ids = set(objects), set(aliases)
    example_aliases = []
    for index, template in enumerate(templates):
        refs = references(template, object_ids)
        refs.update(slot["object_id"] for slot in saved["root_inputs"][index])
        refs.update(
            value for value in saved["public_outputs"][index] if value is not None
        )
        example_aliases.append(
            {objects[value].alias_group_id for value in refs}
            | references(template, alias_ids)
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
    common_refs = references(saved.get("optimizer_templates", ()), object_ids)
    common_refs.update(references(saved.get("gradient_templates", ()), object_ids))
    shared.update(objects[value].alias_group_id for value in common_refs)
    shared.update(alias_ids - set().union(*example_aliases))
    storage = clone_storage(
        program, shared, [example_aliases[index] - shared for index in indices]
    )
    # Only the replaced lists below are edited. Keep the other immutable nested
    # bindings intact until Program validates and freezes the assembled result.
    bindings = dict(saved)
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
    if "objective_schema" in saved:
        # Static metric values belong to the captured form each example runs.
        schema = plain(saved["objective_schema"])
        schema["static_leaves"] = [schema["static_leaves"][i] for i in indices]
        bindings["objective_schema"] = schema
    # Collective descriptions are form-keyed, but Program validates them against
    # actual tasks. Restore them after the control-only intermediate is rebound.
    collectives = bindings.pop("collective_tasks", {})
    signatures = saved.get(
        "input_signatures", [saved["input_signature"]] * len(templates)
    )
    root_sources = saved.get(
        "example_root_sources", [saved["root_sources"]] * len(templates)
    )
    for destination, source in enumerate(indices):
        private = example_aliases[source] - shared
        alias_map = storage.alias_maps[destination]
        object_map = storage.object_maps[destination]
        bindings["training_templates"].append(
            remap(templates[source], object_map, alias_map)
        )
        roots = remap(saved["root_inputs"][source], object_map, alias_map)
        bindings["root_inputs"].append(roots)
        bindings["public_outputs"].append(
            [object_map.get(value, value) for value in saved["public_outputs"][source]]
        )
        bindings["input_signatures"].append(plain(signatures[source]))
        bindings["example_root_sources"].append(plain(root_sources[source]))
        bindings["fixed_inputs"].extend(
            remap(item, object_map, alias_map)
            for item in saved.get("fixed_inputs", ())
            if item["object_id"] in object_map
        )
        for old, new in zip(saved["root_inputs"][source], roots, strict=True):
            alias = objects[old["object_id"]].alias_group_id
            if alias in private:
                name = root_sources[source][new["leaf_index"]]["name"]
                storage.sources[alias_map[alias]] = [f"example[{destination}].{name}"]
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
        objects=tuple(storage.objects),
        alias_groups=tuple(storage.aliases),
        tasks=(control,),
        task_alternative_groups=(),
        initial_residency=tuple(storage.initial),
        final_residency=tuple(storage.final),
        bindings=bindings,
        sources=storage.sources,
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
