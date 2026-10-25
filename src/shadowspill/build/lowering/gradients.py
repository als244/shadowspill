"""Sharded accumulator transfers and local optimizer reduction buffers."""

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
    *,
    for_optimizer: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    templates, persistent = [], []
    collectives: dict[str, Any] = {}
    shards = {}
    for binding in objects.gradients:
        name = binding.parameter_name
        item = operations[name]
        reduction = item.reduction
        example = item.owned
        alias = objects.catalog.alias_id(binding.gradient_object_id)
        width = example.dtype.itemsize
        objects.catalog.pad_alias(alias, reduction.padded_elements * width)
        shards[alias] = {
            "groups": reduction.phases[0].groups,
            "replica_groups": (
                reduction.phases[1].groups if len(reduction.phases) > 1 else ()
            ),
            "shard_bytes": reduction.result_elements * width,
            "logical_bytes": reduction.source_elements * width,
            "padded_bytes": reduction.padded_elements * width,
            "dtype": str(example.dtype),
        }
        if not for_optimizer:
            persistent.append(
                {
                    "parameter_name": name,
                    "gradient_object_id": binding.gradient_object_id,
                    "shape": (reduction.source_elements,),
                    "dtype": str(example.dtype),
                }
            )
            continue
        stored = _buffer(objects, example, reduction.result_elements, Persistence.STEP)
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
                "padding": padding,
                "phases": phases,
            }
        )
    return templates, {
        "collective_tasks": collectives,
        "persistent_gradients": persistent,
        "gradient_shards": shards,
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
