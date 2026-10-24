"""Static owned-gradient buffers and reusable completion templates."""

from typing import Any

import torch

from shadowspill.artifacts._values import digest
from shadowspill.artifacts.program import TaskDefinition
from shadowspill.ir import ObjectRole, Persistence
from shadowspill.pytorch.lowering.training.artifacts import TrainingObjects

from ..gradient_ops import GradientOperations
from .catalog import CompiledTasks


def gradient_templates(
    operations: dict[str, GradientOperations],
    objects: TrainingObjects,
    compiled: CompiledTasks,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    templates, collectives, persistent = [], {}, []
    for binding in objects.gradients:
        name = binding.parameter_name
        item = operations[name]
        reduction = item.reduction
        example = item.accumulation.example_arguments[0]
        assert isinstance(example, torch.Tensor)

        stored = _buffer(objects, example, reduction.result_elements, Persistence.RUN)
        new = _buffer(objects, example, reduction.result_elements, Persistence.STEP)
        padding = None
        if item.padding is not None:
            padding = {
                "profile_key": compiled.key(item.padding, (f"{name}.grad padding",)),
                "output": _buffer(
                    objects, example, reduction.padded_elements, Persistence.STEP
                ),
            }
        phases = []
        for phase in reduction.phases:
            description = {
                "operation": phase.operation,
                "groups": phase.groups,
                "tensor_bytes": phase.input_elements * item.dtype.itemsize,
                "dtype": str(item.dtype),
            }
            key = "collective_" + digest(description)
            compiled.definitions[key] = TaskDefinition(
                key,
                key,
                None,
                f"{phase.operation} gradients",
            )
            collectives[key] = description
            phases.append({"profile_key": key, "operation": phase.operation})
        templates.append(
            {
                "name": name,
                "source": binding.gradient_object_id,
                "stored": stored,
                "new": new,
                "padding": padding,
                "phases": phases,
                "accumulation_key": compiled.key(
                    item.accumulation, (f"{name}.grad accumulation",)
                ),
            }
        )
        persistent.append(
            {
                "parameter_name": name,
                "gradient_object_id": stored,
                "shape": (reduction.result_elements,),
                "dtype": str(item.dtype),
            }
        )
    return templates, {
        "collective_tasks": collectives,
        "persistent_gradients": persistent,
    }


def _buffer(
    objects: TrainingObjects,
    example: torch.Tensor,
    elements: int,
    persistence: Persistence,
) -> str:
    return objects.catalog.add(
        torch.empty(elements, dtype=example.dtype, device=example.device),
        role=ObjectRole.GRADIENT,
        persistence=persistence,
    )
