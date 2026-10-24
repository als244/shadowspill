"""Geometry-only local optimizer parameters, separate from model computation."""

import copy
from collections.abc import Collection, Mapping

import torch
from torch import nn

from shadowspill.parallelism.ownership import ParameterOwnership
from shadowspill.pytorch.optimizer.sandbox import optimizer_parameters
from shadowspill.pytorch.optimizer.sharding import validate_flat_shards


def local_optimizer_parameters(
    optimizer: torch.optim.Optimizer,
    names: dict[int, str],
    model_parameters: Mapping[str, nn.Parameter],
    trained: Collection[str],
    dtype: torch.dtype | None,
    ownership: Mapping[str, ParameterOwnership] | None = None,
) -> tuple[torch.optim.Optimizer, dict[int, str], dict[str, torch.Tensor]]:
    if dtype is None and not ownership:
        return optimizer, names, {}
    if dtype is not None and not dtype.is_floating_point:
        raise ValueError("master_dtype must be a floating-point dtype")
    sharded = {
        name: record.optimizer.capacity
        for name, record in (ownership or {}).items()
        if name in trained and len(record.optimizer.groups[0]) > 1
    }
    if sharded:
        validate_flat_shards(optimizer)
    replacements = {}
    copies: dict[str, torch.Tensor] = {}
    for parameter in optimizer_parameters(optimizer):
        name = names[id(parameter)]
        target_dtype = dtype or parameter.dtype
        if name not in trained or (
            parameter.dtype == target_dtype and name not in sharded
        ):
            continue
        if name in sharded and optimizer.state.get(parameter):
            raise ValueError(
                "build sharded updates with an uninitialized optimizer; "
                "restore persistent optimizer tensors during execution"
            )
        if name in sharded:
            from shadowspill.pytorch.representations import is_wrapper

            if is_wrapper(model_parameters[name]):
                raise NotImplementedError(
                    "owned updates of tensor-wrapper parameters need their "
                    "representation's shard/copy contract"
                )
        shape = (sharded[name],) if name in sharded else tuple(parameter.shape)
        stride = (1,) if name in sharded else tuple(parameter.stride())
        owned = nn.Parameter(
            torch.empty_strided(
                shape,
                stride,
                dtype=target_dtype,
                device="meta",
            ),
            requires_grad=parameter.requires_grad,
        )
        replacements[id(parameter)] = owned
        if target_dtype != parameter.dtype:
            copies[name] = (
                torch.empty(shape, dtype=model_parameters[name].dtype, device="meta")
                if name in sharded
                else model_parameters[name]
            )
    if not replacements:
        return optimizer, names, copies
    # Replace every reference, including state keys or subclass-held references.
    result = object.__new__(type(optimizer))
    result.__dict__ = copy.deepcopy(optimizer.__dict__, replacements)
    # deepcopy also copies unchanged parameters; recover names by stable group order.
    result_names = {
        id(new): names[id(old)]
        for old, new in zip(
            optimizer_parameters(optimizer), optimizer_parameters(result), strict=True
        )
    }
    return result, result_names, copies
