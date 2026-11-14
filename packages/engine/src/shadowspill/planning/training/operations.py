"""Derive an optimizer invocation from saved complete-step task components."""

from dataclasses import replace
from math import prod

from shadowspill.artifacts._values import plain
from shadowspill.artifacts.program import Program
from shadowspill.artifacts.tensors import tensor_geometry
from shadowspill.ir import (
    MemoryLocation,
    Persistence,
    ResidencySpec,
    ResourceKind,
    ResourceSpec,
    TaskSpec,
)

from .communication import order_collectives
from .identity import canonicalize_tasks
from .optimizer import place_optimizer
from .tasks import TrainingSequence


def optimizer_program(step: Program) -> Program:
    """Reuse captured updates, reductions and publications without model capture.

    The result accepts no examples. Its initial gradient storage has exactly
    the producer layout declared by the complete-step Program; existing generic
    reduction/transfer contracts deliver it to each local optimizer consumer.
    All task identities and binaries retain their profile compatibility keys.
    ``apply_profiles(result, step_profiles)`` reuses their measured evidence.
    """
    if step.bindings.get("kind") != "step" or not step.bindings.get(
        "optimizer_templates"
    ):
        raise ValueError(
            "optimizer_program requires a step with captured optimizer tasks"
        )
    device = step.devices[0].device_id
    aliases = {obj.object_id: obj.alias_group_id for obj in step.objects}
    sequence = TrainingSequence(
        tasks=[
            TaskSpec(
                "invocation_start",
                ResourceSpec(device, ResourceKind.CONTROL),
                "invocation_start",
                phase="control",
                requires_entrypoint=False,
            )
        ]
    )
    sequence = place_optimizer(
        sequence,
        step.bindings["optimizer_templates"],
        ordering="tail",
        aliases=aliases,
        device_id=device,
    )
    order_collectives(sequence)
    names = canonicalize_tasks(sequence)
    profiles = {task.profile_id for task in sequence.tasks}
    bindings = plain(step.bindings)
    for key in (
        "training_templates",
        "ordering",
        "ordering_constraints",
        "accumulate_gradients",
        "input_signature",
        "input_signatures",
        "example_root_sources",
    ):
        bindings.pop(key, None)
    geometry = tensor_geometry(step)
    gradients = [
        {
            **item,
            "shape": (prod(geometry[item["gradient_object_id"]]["shape"]),),
            "dtype": geometry[item["gradient_object_id"]]["dtype"],
        }
        for item in bindings["gradients"]
    ]
    bindings.update(
        kind="optim_step",
        num_examples=0,
        root_inputs=[],
        root_sources=[],
        fixed_inputs=[],
        public_outputs=[],
        persistent_gradients=gradients,
        entrypoints=sequence.entrypoints,
        task_names=names,
        transient_outputs=sequence.transient_outputs,
        collective_tasks={
            key: value
            for key, value in bindings.get("collective_tasks", {}).items()
            if key in profiles
        },
    )
    state = (
        {item["object_id"] for item in bindings["registrations"]}
        | {
            item["object_id"]
            for item in bindings["optimizer_bindings"]
            if item["object_id"] is not None and item["role"] != "gradient"
        }
        | {item["gradient_object_id"] for item in gradients}
        | {
            item["object_id"]
            for item in bindings.get("registered_state", ())
            if item.get("object_id") is not None
        }
    )
    used = state | {
        value
        for task in sequence.tasks
        for value in (
            *task.inputs,
            *task.outputs,
            *(m.object_id for m in task.mutations),
        )
    }
    kept_aliases = {aliases[value] for value in used}
    produced = {aliases[value] for task in sequence.tasks for value in task.outputs}
    initial = {aliases[value] for value in state} - produced
    final = {aliases[value] for value in state}
    # Reduction scratch belongs to the invocation. Source gradient destinations
    # survive in spill so another operation can inspect/export the same inventory.
    gradient_ids = {item["gradient_object_id"] for item in gradients}
    bindings["weight_shards"] = {
        key: value
        for key, value in bindings.get("weight_shards", {}).items()
        if key in kept_aliases
    }
    bindings["gradient_shards"] = {
        key: value
        for key, value in bindings.get("gradient_shards", {}).items()
        if key in kept_aliases
    }
    return replace(
        step,
        name=step.name + " optimizer",
        objects=tuple(
            replace(obj, persistence=Persistence.RUN)
            if obj.object_id in gradient_ids
            else obj
            for obj in step.objects
            if obj.object_id in used
        ),
        alias_groups=tuple(
            a for a in step.alias_groups if a.alias_group_id in kept_aliases
        ),
        tasks=tuple(sequence.tasks),
        task_alternative_groups=(),
        task_definitions=tuple(
            d for d in step.task_definitions if d.profile_key in profiles
        ),
        executables={
            key: value for key, value in step.executables.items() if key in profiles
        },
        sources={
            key: value for key, value in step.sources.items() if key in kept_aliases
        },
        initial_residency=tuple(
            ResidencySpec(alias, MemoryLocation.SPILL) for alias in sorted(initial)
        ),
        final_residency=tuple(
            ResidencySpec(alias, MemoryLocation.SPILL) for alias in sorted(final)
        ),
        bindings=bindings,
    )
