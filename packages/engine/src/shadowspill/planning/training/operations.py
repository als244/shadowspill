"""Derive an optimizer invocation from saved complete-step task components."""

from dataclasses import replace
from math import prod
from typing import Any

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


def _persistent_gradients(program: Program) -> list[dict[str, Any]]:
    """Keep the producer's gradient geometry for both derived operation paths."""
    geometry = tensor_geometry(program)
    return [
        {
            **item,
            "shape": (prod(geometry[item["gradient_object_id"]]["shape"]),),
            "dtype": geometry[item["gradient_object_id"]]["dtype"],
        }
        for item in program.bindings["gradients"]
    ]


def optimizer_program(step: Program) -> Program:
    """Derive an optimizer-only Program from a complete-step Program.

    The result runs the step's captured updates, gradient reductions and
    weight publications on the gradients already stored, in the step's
    gradient layout, and takes no examples. Nothing is captured or compiled
    and every task keeps its profile key, so
    ``apply_profiles(result, step_profiles)`` reuses the step's measurements.
    The update consumes the gradients: they are not kept after it. Admit its
    Plan in one `Executor` with the backward Plans that produce the gradients;
    the Executor runs it only while their gradients are held.

    Parameters
    ----------
    step : Program
        A complete-step Program, from `build_step`.

    Returns
    -------
    Program
        The optimizer-only Program.

    Raises
    ------
    ValueError
        If ``step`` is not a complete-step Program with captured optimizer
        tasks, or its update reads none of some gradient's value.
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
    gradients = _persistent_gradients(step)
    geometry = tensor_geometry(step)
    bindings.update(
        kind="optim_step",
        num_examples=0,
        root_inputs=[],
        root_sources=[],
        fixed_inputs=[],
        public_outputs=[],
        persistent_gradients=gradients,
        gradient_geometry={
            item["gradient_object_id"]: geometry[item["gradient_object_id"]]
            for item in gradients
        },
        entrypoints=sequence.entrypoints,
        task_names=names,
        transient_outputs=sequence.transient_outputs,
        operation_derivation={"kind": "optim_step", "source_program": step.digest},
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
    # The update consumes the gradients: they end with it, so the runtime frees
    # their storage at their last read instead of writing them back.
    gradient_ids = {item["gradient_object_id"] for item in gradients}
    final = {aliases[value] for value in state - gradient_ids}
    read = {
        aliases[value]
        for task in sequence.tasks
        for value in (*task.inputs, *(m.object_id for m in task.mutations))
    }
    unread = {aliases[value] for value in gradient_ids} - read - final
    if unread:
        raise ValueError(
            f"the update reads no task's value of gradients {sorted(unread)}"
        )
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
