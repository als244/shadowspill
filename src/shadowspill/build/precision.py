"""Geometry-only optimizer masters, separate from model compute precision."""

import copy
from collections.abc import Collection, Mapping

import torch
from torch import nn

from shadowspill.pytorch.optimizer.sandbox import optimizer_parameters


def with_master_parameters(
    optimizer: torch.optim.Optimizer,
    names: dict[int, str],
    model_parameters: Mapping[str, nn.Parameter],
    trained: Collection[str],
    dtype: torch.dtype | None,
) -> tuple[torch.optim.Optimizer, dict[int, str], dict[str, torch.Tensor]]:
    if dtype is None:
        return optimizer, names, {}
    if not dtype.is_floating_point:
        raise ValueError("master_dtype must be a floating-point dtype")
    replacements = {}
    copies: dict[str, torch.Tensor] = {}
    for parameter in optimizer_parameters(optimizer):
        name = names[id(parameter)]
        if name not in trained or parameter.dtype == dtype:
            continue
        master = nn.Parameter(
            torch.empty_strided(
                parameter.shape,
                parameter.stride(),
                dtype=dtype,
                device="meta",
            ),
            requires_grad=parameter.requires_grad,
        )
        replacements[id(parameter)] = master
        copies[name] = model_parameters[name]
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
