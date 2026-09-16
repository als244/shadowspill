"""The tensors an optimizer update reads and writes, named: parameters, gradients,
state and hyperparameters, with the roles and provenance a captured task carries."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import torch
from torch._subclasses.fake_tensor import FakeTensor

from shadowspill.errors import CaptureError
from shadowspill.pytorch.capture.artifacts import TaskInputProvenance
from shadowspill.task.inputs import TaskInputRole

from .artifacts import (
    OptimizerTensorBinding,
    OptimizerTensorRole,
)
from .sandbox import (
    optimizer_parameters,
)


def _tensor_leaves(
    value: object, prefix: str = ""
) -> Iterable[tuple[str, torch.Tensor]]:
    if isinstance(value, torch.Tensor):
        yield prefix or "value", value
    elif isinstance(value, Mapping):
        for key in sorted(value, key=str):
            child = f"{prefix}.{key}" if prefix else str(key)
            yield from _tensor_leaves(value[key], child)
    elif isinstance(value, tuple | list):
        for index, item in enumerate(value):
            child = f"{prefix}.{index}" if prefix else str(index)
            yield from _tensor_leaves(item, child)


def state_tensor_names(
    optimizer: torch.optim.Optimizer, name_by_id: Mapping[int, str]
) -> tuple[str, ...]:
    names: list[str] = []
    for parameter in optimizer_parameters(optimizer):
        parameter_name = name_by_id.get(id(parameter))
        if parameter_name is None:
            continue
        for path, _tensor in _tensor_leaves(optimizer.state.get(parameter, {})):
            names.append(f"optimizer.{parameter_name}.{path}")
    return tuple(names)


def state_structure(
    optimizer: torch.optim.Optimizer, name_by_id: Mapping[int, str]
) -> tuple[tuple[str, str, tuple[int, ...], str], ...]:
    result: list[tuple[str, str, tuple[int, ...], str]] = []
    for parameter in optimizer_parameters(optimizer):
        parameter_name = name_by_id.get(id(parameter), "unknown")
        for path, tensor in _tensor_leaves(optimizer.state.get(parameter, {})):
            result.append(
                (parameter_name, path, tuple(tensor.shape), str(tensor.dtype))
            )
    return tuple(result)


def tensor_bindings(
    optimizer: torch.optim.Optimizer,
    name_by_id: Mapping[int, str],
    *,
    require_gradients: bool = True,
) -> tuple[OptimizerTensorBinding, ...]:
    bindings: list[OptimizerTensorBinding] = []
    seen: set[int] = set()

    def add(
        name: str, role: OptimizerTensorRole, tensor: torch.Tensor, mutable: bool
    ) -> None:
        if id(tensor) in seen:
            return
        seen.add(id(tensor))
        # Device-side control tensors belong in the physical plan even when
        # scalar. Ordinary PyTorch optimizers intentionally keep some scalar
        # counters on the CPU; those remain bounded host-side task inputs.
        # A hyperparameter is neither: it is a handful of bytes the update
        # reads and the caller writes between steps, so placing it would cost
        # more than it could ever free, and moving it would put the value
        # somewhere the caller's next write would not reach.
        spillable = role is not OptimizerTensorRole.HYPERPARAMETER and (
            role
            in {
                OptimizerTensorRole.PARAMETER,
                OptimizerTensorRole.GRADIENT,
            }
            or tensor.ndim != 0
            or tensor.device.type != "cpu"
        )
        bindings.append(OptimizerTensorBinding(name, role, tensor, mutable, spillable))

    for parameter in optimizer_parameters(optimizer):
        name = name_by_id.get(id(parameter))
        if name is None or not parameter.requires_grad:
            continue
        add(name, OptimizerTensorRole.PARAMETER, parameter, True)
        if parameter.grad is None:
            if require_gradients:
                raise CaptureError(f"optimizer capture has no gradient for {name!r}")
        else:
            add(
                f"gradient.{name}",
                OptimizerTensorRole.GRADIENT,
                parameter.grad,
                False,
            )
        for path, tensor in _tensor_leaves(optimizer.state.get(parameter, {})):
            add(f"optimizer.{name}.{path}", OptimizerTensorRole.STATE, tensor, True)
    for group_index, group in enumerate(optimizer.param_groups):
        for path, tensor in _tensor_leaves(
            {key: value for key, value in group.items() if key != "params"}
        ):
            # Read by the update, never written by it. Saying so matters:
            # a mutable binding joins every task that touches it, and a
            # hyperparameter is touched by every parameter's update, so
            # calling it mutable would fuse updates that share nothing else
            # into a single task.
            add(
                f"optimizer_group.{group_index}.{path}",
                OptimizerTensorRole.HYPERPARAMETER,
                tensor,
                False,
            )
    return tuple(bindings)


def representative_optimizer_values(
    optimizer: torch.optim.Optimizer,
    name_by_id: Mapping[int, str],
) -> dict[str, torch.Tensor]:
    """Retain occurrence-local initialized values before symbolic conversion."""

    return {
        binding.name: binding.tensor.detach()
        for binding in tensor_bindings(
            optimizer,
            name_by_id,
            require_gradients=False,
        )
        if binding.role is not OptimizerTensorRole.GRADIENT
        and not isinstance(binding.tensor, FakeTensor)
        and binding.tensor.device.type != "meta"
    }


def optimizer_input_provenance(
    bindings: tuple[OptimizerTensorBinding, ...],
    representative_values: Mapping[str, torch.Tensor],
) -> tuple[TaskInputProvenance, ...]:
    role_map = {
        OptimizerTensorRole.PARAMETER: TaskInputRole.PARAMETER,
        OptimizerTensorRole.GRADIENT: TaskInputRole.GRADIENT,
        OptimizerTensorRole.STATE: TaskInputRole.OPTIMIZER_STATE,
        OptimizerTensorRole.HYPERPARAMETER: (TaskInputRole.OPTIMIZER_HYPERPARAMETER),
    }
    return tuple(
        TaskInputProvenance(
            role_map[binding.role],
            binding.name,
            representative_value=representative_values.get(binding.name),
        )
        for binding in bindings
    )


def completion_stage(
    bindings: tuple[OptimizerTensorBinding, ...],
    parameter_stage_owners: Mapping[str, tuple[int, ...]] | None,
) -> int | None:
    """Return the backward frontier after which all bound gradients are final."""

    if parameter_stage_owners is None:
        return None
    stages = {
        stage
        for binding in bindings
        if binding.role is OptimizerTensorRole.PARAMETER
        for stage in parameter_stage_owners.get(binding.name, ())
    }
    return min(stages) if stages else None


def restore_binding_values(
    bindings: tuple[OptimizerTensorBinding, ...], snapshots: Mapping[int, torch.Tensor]
) -> None:
    with torch.no_grad():
        for binding in bindings:
            binding.tensor.copy_(snapshots[id(binding.tensor)])
