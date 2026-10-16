"""Retain frontend names before lowering bindings leave the build process."""

from __future__ import annotations

from typing import Any

from shadowspill.pytorch.lowering.training.artifacts import LoweredTrainingProgram


def training_semantic_labels(lowered: LoweredTrainingProgram) -> dict[str, Any]:
    """Names are annotations keyed by stable IDs, never planning inputs."""
    tasks = {}
    for entry in lowered.entrypoints:
        options = entry.options
        tasks[entry.task_id] = {
            "phase": options.phase,
            "stage_index": options.stage_index,
            "microbatch": options.repetition,
            "variant": options.variant,
            "target": options.target,
        }
    objects: dict[str, dict[str, Any]] = {}
    for registration in lowered.registrations:
        record = objects.setdefault(registration.object_id, {"names": []})
        path = ".".join((registration.name, *registration.component_path))
        record["names"].append(path)
        role = "parameter" if registration.parameter else "buffer"
        record.setdefault("label", f"{role} {path}")
    for gradient in lowered.gradients:
        objects[gradient.gradient_object_id] = {
            "label": f"gradient {gradient.parameter_name}",
            "parameter_name": gradient.parameter_name,
        }
    for state in lowered.optimizer_objects:
        objects.setdefault(
            state.object_id, {"label": f"optimizer {state.name}", "name": state.name}
        )
    for microbatch, slots in enumerate(lowered.root_input_slots):
        for slot in slots:
            objects.setdefault(
                slot.object_id,
                {
                    "label": f"input {slot.leaf_index}, microbatch {microbatch}",
                    "microbatch": microbatch,
                    "input_index": slot.leaf_index,
                },
            )
    roles = {obj.object_id: obj.role.value for obj in lowered.program.objects}
    for entry in lowered.entrypoints:
        for slot in entry.output_slots:
            record = objects.setdefault(slot.object_id, {})
            record.setdefault("origins", []).append(
                {
                    "task_id": entry.task_id,
                    "output_index": slot.leaf_index,
                }
            )
            # Some objects are shared by save/recompute task alternatives.
            # The analyzer picks the active producer, rather than naming one here.
            record.setdefault("role", roles[slot.object_id])
    return {
        "program_digest": lowered.program.digest,
        "tasks": tasks,
        "objects": objects,
    }
