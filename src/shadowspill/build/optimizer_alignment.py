"""Compile owned weight views for every rank's possible pointer alignment."""

from torch._inductor.utils import GPU_ALIGN_BYTES

from shadowspill.parallelism.ownership import ParameterOwnership
from shadowspill.pytorch.optimizer.artifacts import OptimizerTensorRole

from .optimizer import BuiltOptimizer


def optimizer_input_offsets(
    update: BuiltOptimizer | None, parameters: tuple[ParameterOwnership, ...]
) -> dict[str, tuple[tuple[int, int], ...]]:
    if update is None or not parameters:
        return {}
    ownership = {item.name: item for item in parameters}
    bindings = update.captured.bindings
    mastered = {
        item.parameter_name
        for item in bindings
        if item.role is OptimizerTensorRole.COMPUTE_COPY
    }
    offsets = {}
    for binding in bindings:
        name = binding.parameter_name
        if binding.role is not OptimizerTensorRole.COMPUTE_COPY and not (
            binding.role is OptimizerTensorRole.PARAMETER and name not in mastered
        ):
            continue
        assert name is not None
        layout = ownership[name].optimizer
        # Inductor's input alignment requirement is 16 bytes. A complete shard
        # stride preserves it on every rank, so leave those inputs unchanged.
        if (
            len(layout.groups[0]) > 1
            and layout.capacity * binding.tensor.element_size() % GPU_ALIGN_BYTES
        ):
            offsets[binding.name] = layout.capacity
    return {
        task.artifact.compatibility_digest: tuple(
            (index, offsets[name])
            for index, name in enumerate(task.binding_names)
            if name in offsets
        )
        for task in update.captured.update_tasks
    }
