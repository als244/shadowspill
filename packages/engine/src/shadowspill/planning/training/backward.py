"""Derive fresh/additive backward work from one saved complete-step capture."""

from dataclasses import replace

from shadowspill.artifacts._values import plain
from shadowspill.artifacts.program import Program
from shadowspill.artifacts.tensors import tensor_geometry
from shadowspill.ir import MemoryLocation, Persistence, ResidencySpec

from ..example_storage import references
from .bind import bind_training
from .operations import _persistent_gradients


def backward_program(step: Program) -> Program:
    """Retain captured model/backward forms and persistent producer gradients.

    Optimizer state, controls, reductions and publications belong to the derived
    optimizer-only path. Gradient spill/fetch transfer declarations stay on the
    original producer aliases. This uses the existing training binder and never
    captures or compiles another model. The result still needs planning and
    joint admission with its optimizer and other operation Plans.
    """
    saved = step.bindings
    if (
        saved.get("kind") != "step"
        or not saved.get("training_templates")
        or not saved.get("optimizer_templates")
    ):
        raise ValueError("backward_program requires saved complete-step templates")
    objects = {obj.object_id: obj for obj in step.objects}
    used = references(saved["training_templates"], set(objects))
    used.update(item["object_id"] for item in saved["registrations"])
    used.update(
        item["object_id"]
        for item in saved.get("registered_state", ())
        if item.get("object_id") is not None
    )
    used.update(slot["object_id"] for group in saved["root_inputs"] for slot in group)
    used.update(item["object_id"] for item in saved.get("fixed_inputs", ()))
    used.update(
        value
        for group in saved["public_outputs"]
        for value in group
        if value is not None
    )
    gradients = _persistent_gradients(step)
    geometry = tensor_geometry(step)
    gradient_ids = {item["gradient_object_id"] for item in gradients}
    used.update(gradient_ids)
    aliases = {objects[value].alias_group_id for value in used}
    # Keep complete alias closures, including registered ties and public views
    # that do not themselves occur as task inputs or outputs.
    kept = {obj.object_id for obj in step.objects if obj.alias_group_id in aliases}
    bindings = plain(saved)
    for name in tuple(bindings):
        if name.startswith("optimizer_") or name in {
            "gradient_templates",
            "gradient_scale_control",
            "owned_parameter_views",
            "ordering",
            "accumulate_gradients",
            "example_instances",
        }:
            del bindings[name]
    collectives = bindings.pop("collective_tasks", {})
    bindings.update(
        kind="forward_backward",
        optimizer_templates=[],
        persistent_gradients=gradients,
        gradient_geometry={
            item["gradient_object_id"]: geometry[item["gradient_object_id"]]
            for item in gradients
        },
        entrypoints=[],
        task_names={},
        transient_outputs={},
        operation_derivation={
            "kind": "forward_backward",
            "source_program": step.digest,
        },
    )
    if "distributed_training_lowered" in bindings:
        bindings["distributed_training_lowered"] = "gradients"
    for name in ("weight_shards", "gradient_shards"):
        if name in bindings:
            bindings[name] = {
                key: value for key, value in bindings[name].items() if key in aliases
            }
    final = {
        item.alias_group_id: item
        for item in step.final_residency
        if item.alias_group_id in aliases
    }
    # Keep the saved declaration order: set iteration changes Program identity
    # across fresh workers, even though residency membership is unchanged.
    for gradient in gradients:
        object_id = gradient["gradient_object_id"]
        alias = objects[object_id].alias_group_id
        final[alias] = ResidencySpec(alias, MemoryLocation.SPILL)
    control = next(
        (task for task in step.tasks if task.profile_id == "invocation_start"), None
    )
    if control is None:
        raise ValueError("complete-step Program has no invocation-start task")
    intermediate = replace(
        step,
        name=step.name + " backward",
        objects=tuple(
            replace(obj, persistence=Persistence.RUN)
            if obj.object_id in gradient_ids
            else obj
            for obj in step.objects
            if obj.object_id in kept
        ),
        alias_groups=tuple(
            alias for alias in step.alias_groups if alias.alias_group_id in aliases
        ),
        tasks=(control,),
        task_alternative_groups=(),
        initial_residency=tuple(
            item for item in step.initial_residency if item.alias_group_id in aliases
        ),
        final_residency=tuple(final.values()),
        sources={key: value for key, value in step.sources.items() if key in aliases},
        bindings=bindings,
    )
    result = bind_training(intermediate)
    # Fresh one-example tasks do not mention the additive forms needed by later
    # contributions/count expansion. Retain every captured training variant.
    profiles = {control.profile_id} | {
        form[phase]["profile_key"]
        for example in saved["training_templates"]
        for stage in example
        for form in stage
        for phase in ("forward", "backward")
    }
    return replace(
        result,
        task_definitions=tuple(
            item for item in result.task_definitions if item.profile_key in profiles
        ),
        executables={
            key: value for key, value in result.executables.items() if key in profiles
        },
        bindings={
            **result.bindings,
            "collective_tasks": {
                key: value for key, value in collectives.items() if key in profiles
            },
        },
    )
