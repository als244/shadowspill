"""Bind shard-local optimizer tensors to their model and reduction storage."""

from dataclasses import replace
from typing import Any

from shadowspill.build.optimizer import BuiltOptimizer
from shadowspill.ir import ObjectRole, Persistence
from shadowspill.parallelism.ownership import ParameterOwnership
from shadowspill.pytorch.optimizer.artifacts import OptimizerTensorRole

from ..catalog import CompiledTasks
from ..collectives import declare_collective
from .records import OptimizerObjectBinding, TrainingObjects


def owned_optimizer(
    update: BuiltOptimizer,
    objects: TrainingObjects,
    parameters: tuple[ParameterOwnership, ...],
    completions: list[dict[str, Any]],
    compiled: CompiledTasks,
    collectives: dict[str, Any],
) -> tuple[TrainingObjects, dict[str, Any], dict[str, Any]]:
    """Full weights and owned compute views share one allocation, not two copies.

    The representative view starts at zero. Saved group metadata gives each
    worker its actual offset; local compiled math does not specialize that offset.
    Masters/moments have genuinely shard-sized independent allocations.
    """
    by_name = {item.name: item for item in parameters}
    mastered = {
        item.name.removeprefix("compute.")
        for item in update.captured.bindings
        if item.role is OptimizerTensorRole.COMPUTE_COPY
    }
    gradients = {item["name"]: item["stored"] for item in completions}
    names = {f"gradient.{name}": value for name, value in gradients.items()}
    views, publications, added = {}, {}, []
    for binding in update.captured.bindings:
        role = binding.role
        if role is OptimizerTensorRole.GRADIENT or not binding.spillable:
            continue
        if role is OptimizerTensorRole.COMPUTE_COPY or (
            role is OptimizerTensorRole.PARAMETER and binding.name not in mastered
        ):
            name = binding.name.removeprefix("compute.")
            parameter = by_name[name]
            model_id = objects.parameter_objects[name]
            if len(parameter.optimizer.groups[0]) == 1:
                # Local experts and replicated optimizers update the original
                # shape; no flattened shard view or publication is required.
                names[binding.name] = model_id
                continue
            alias = objects.catalog.alias_id(model_id)
            width = binding.tensor.element_size()
            size = parameter.optimizer.capacity * width
            padded = size * len(parameter.optimizer.groups[0])
            objects.catalog.pad_alias(alias, padded)
            view = objects.catalog.add_output_view(
                alias_id=alias,
                offset_bytes=0,
                span_bytes=size,
                role=ObjectRole.PARAMETER,
                persistence=Persistence.CHECKPOINT,
            )
            full = objects.catalog.add_output_view(
                alias_id=alias,
                offset_bytes=0,
                span_bytes=padded,
                role=ObjectRole.PARAMETER,
                persistence=Persistence.CHECKPOINT,
            )
            names[binding.name] = view
            views[view] = {
                "groups": parameter.optimizer.groups,
                "elements": parameter.optimizer.capacity,
                "dtype": str(binding.tensor.dtype),
                "parameter_name": name,
            }
            key = declare_collective(
                {
                    "operation": "all_gather",
                    "groups": parameter.optimizer.groups,
                    "tensor_bytes": padded,
                    "dtype": str(binding.tensor.dtype),
                    "inplace": True,
                },
                compiled,
                collectives,
            )
            publications[binding.name] = {
                "profile_key": key,
                "source": view,
                "destination": full,
            }
        else:
            names[binding.name] = objects.catalog.add(
                binding.tensor,
                role=(
                    ObjectRole.PARAMETER
                    if role is OptimizerTensorRole.PARAMETER
                    else ObjectRole.OPTIMIZER_STATE
                ),
                persistence=Persistence.CHECKPOINT,
                retain_spill_copy=True,
            )
            added.append(
                OptimizerObjectBinding(
                    binding.name, names[binding.name], binding.role, binding.mutable
                )
            )
    return (
        replace(objects, optimizer_objects=(*objects.optimizer_objects, *added)),
        {"names": names, "owned_parameter_views": views},
        publications,
    )
