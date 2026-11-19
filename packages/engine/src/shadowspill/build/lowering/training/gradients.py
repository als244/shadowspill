"""Sharded accumulator transfers and local optimizer reduction buffers."""

from typing import Any

import torch

from shadowspill.build.gradient_ops import GradientOperations
from shadowspill.ir import ObjectRole, Persistence

from ..catalog import CompiledTasks
from ..collectives import declare_collective
from .records import TrainingObjects


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
        spill = item.spill_reduction
        # The producer owns the gradient-storage layout. Optimizer reduction
        # padding belongs to its separate local buffer below; extending the
        # producer would disagree with its compiled output when only optimizer
        # state is sharded.
        objects.catalog.pad_alias(alias, spill.padded_elements * width)
        if spill.phases and spill.phases[0].operation == "reduce_scatter":
            shards[alias] = {
                "groups": spill.phases[0].groups,
                "replica_groups": (
                    spill.phases[1].groups if len(spill.phases) > 1 else ()
                ),
                "shard_bytes": spill.result_elements * width,
                "logical_bytes": spill.source_elements * width,
                "padded_bytes": spill.padded_elements * width,
                "dtype": str(example.dtype),
            }
        elif spill.phases and not for_optimizer:
            # A gradient every copy keeps whole, in a call with no optimizer to
            # reduce it: summed across its copies as it leaves the device, and
            # reloaded on one copy only, so later contributions add to the sum
            # once. With an optimizer it stays local and the update reduces it.
            assert [phase.operation for phase in spill.phases] == ["all_reduce"]
            shards[alias] = {
                "groups": (),
                "replica_groups": spill.phases[0].groups,
                "shard_bytes": spill.padded_elements * width,
                "logical_bytes": spill.source_elements * width,
                "padded_bytes": spill.padded_elements * width,
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
        padding = None
        if item.padding is not None:
            padding = {
                "profile_key": compiled.key(item.padding, (f"{name}.grad padding",)),
                "output": _buffer(
                    objects, example, reduction.padded_elements, Persistence.STEP
                ),
            }
        if not reduction.phases:
            stored = binding.gradient_object_id
        elif reduction.phases[0].operation == "all_reduce":
            # Clone the local contribution before its in-place SUM. The
            # original accumulator may still be the source of an eviction.
            assert padding is not None
            stored = padding["output"]
        else:
            stored = _buffer(
                objects, example, reduction.result_elements, Persistence.STEP
            )
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
