"""Static owned-gradient buffers and reusable completion templates."""

from typing import Any

import torch

from shadowspill.ir import ObjectRole, Persistence
from shadowspill.pytorch.lowering.training.artifacts import TrainingObjects

from ..gradient_ops import GradientOperations
from .catalog import CompiledTasks
from .collectives import declare_collective


def gradient_templates(
    operations: dict[str, GradientOperations],
    objects: TrainingObjects,
    compiled: CompiledTasks,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    templates, persistent = [], []
    collectives: dict[str, Any] = {}
    for binding in objects.gradients:
        name = binding.parameter_name
        item = operations[name]
        reduction = item.reduction
        example = item.owned
        persistent_result = item.accumulation is not None

        stored = _buffer(
            objects,
            example,
            reduction.result_elements,
            Persistence.RUN if persistent_result else Persistence.STEP,
        )
        new = (
            _buffer(objects, example, reduction.result_elements, Persistence.STEP)
            if persistent_result
            else None
        )
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
                "tensor_bytes": phase.input_elements * example.dtype.itemsize,
                "dtype": str(example.dtype),
            }
            key = declare_collective(description, compiled, collectives)
            phases.append({"profile_key": key, "operation": phase.operation})
        templates.append(
            {
                "name": name,
                "source": binding.gradient_object_id,
                "stored": stored,
                "new": new,
                "padding": padding,
                "phases": phases,
                "accumulation_key": (
                    compiled.key(item.accumulation, (f"{name}.grad accumulation",))
                    if item.accumulation is not None
                    else None
                ),
            }
        )
        persistent.append(
            {
                "parameter_name": name,
                "gradient_object_id": stored,
                "shape": (reduction.result_elements,),
                "dtype": str(example.dtype),
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
