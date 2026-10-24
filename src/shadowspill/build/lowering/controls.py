"""Device control tensors occupy planned storage like every other device input."""

from dataclasses import replace

from shadowspill.ir import ObjectRole, Persistence
from shadowspill.pytorch.lowering.training.artifacts import (
    OptimizerObjectBinding,
    TrainingObjects,
)

from ..optimizer import BuiltOptimizer


def register_device_controls(
    objects: TrainingObjects, update: BuiltOptimizer | None
) -> TrainingObjects:
    if update is None:
        return objects
    known = {item.name for item in objects.optimizer_objects}
    added = []
    for binding in update.captured.bindings:
        if (
            binding.role.value == "hyperparameter"
            and binding.tensor.device.type != "cpu"
            and binding.name not in known
        ):
            object_id = objects.catalog.add(
                binding.tensor,
                role=ObjectRole.OPTIMIZER_STATE,
                persistence=Persistence.RUN,
                retain_spill_copy=True,
            )
            added.append(
                OptimizerObjectBinding(
                    binding.name, object_id, binding.role, binding.mutable
                )
            )
    return replace(objects, optimizer_objects=(*objects.optimizer_objects, *added))
