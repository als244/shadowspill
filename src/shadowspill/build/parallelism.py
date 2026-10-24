"""Resolve the model inventory before capture or any live-resource setup."""

from dataclasses import replace

from torch import nn

from shadowspill.artifacts.program import Program
from shadowspill.parallelism import Parallelism
from shadowspill.parallelism.ownership import ParameterOwnership
from shadowspill.parallelism.resolve import ParameterGeometry, resolve_parameters


def model_parallelism(
    model: nn.Module, parallelism: Parallelism | None
) -> tuple[Parallelism, tuple[ParameterOwnership, ...]]:
    spec = parallelism or Parallelism.dp()
    if not isinstance(spec, Parallelism):
        raise TypeError("parallelism must be Parallelism")
    tensors = (
        *model.named_parameters(remove_duplicate=False),
        *model.named_buffers(remove_duplicate=False),
    )
    names: dict[int, list[str]] = {}
    values = {}
    for name, value in tensors:
        names.setdefault(id(value), []).append(name)
        values[id(value)] = value
    geometry = tuple(
        ParameterGeometry(tuple(names[key]), tuple(value.shape), value.requires_grad)
        for key, value in values.items()
    )
    return spec, resolve_parameters(spec, geometry)


def attach_parallelism(
    program: Program, spec: Parallelism, parameters: tuple[ParameterOwnership, ...]
) -> Program:
    program = replace(
        program,
        parallelism=spec,
        bindings={
            **program.bindings,
            "parameter_ownership": [item.to_dict() for item in parameters],
        },
    )
    if spec.world_size > 1:
        from .lowering.shards import weight_shards

        return weight_shards(program, parameters)
    return program


def validate_forward(
    spec: Parallelism, parameters: tuple[ParameterOwnership, ...]
) -> None:
    if spec.world_size > 1 and any(len(item.copy_groups) != 1 for item in parameters):
        raise NotImplementedError(
            "initial distributed forward lowering supports replicated local "
            "computation; "
            "model-sharded operators need their provider resource bindings"
        )


def validate_step(
    spec: Parallelism, parameters: tuple[ParameterOwnership, ...]
) -> None:
    validate_forward(spec, parameters)
    if spec.world_size == 1:
        return
    if any(
        item.trainable
        and (
            len(item.optimizer.groups[0]) == 1
            or item.weights.groups != item.optimizer.groups
            or item.gradients.groups != item.optimizer.groups
        )
        for item in parameters
    ):
        raise NotImplementedError(
            "initial distributed steps require weights, gradients and optimizer "
            "data sharded over the same groups; differing layouts need redistribution"
        )
