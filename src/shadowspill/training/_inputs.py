"""User data stays a pytree; only loss scaling is added by the runner."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from typing import Any

import torch
from torch import nn
from torch.utils._pytree import tree_map

from shadowspill.pytorch.contracts import normalize_objective_result

from ._types import Microbatches, Objective


def single_microbatch(data: Any) -> Iterator[tuple[Any, float]]:
    yield data, 1.0


def candidate_functions(
    value: Microbatches | Mapping[str, Microbatches] | None,
) -> dict[str, Microbatches]:
    if value is None:
        return {"default": single_microbatch}
    if callable(value):
        return {"default": value}
    if not isinstance(value, Mapping) or not value:
        raise ValueError("microbatches must be a function or a nonempty named mapping")
    if any(
        not isinstance(name, str) or not name or not callable(fn)
        for name, fn in value.items()
    ):
        raise TypeError("each microbatch candidate needs a nonempty name and function")
    return dict(value)


def make_microbatches(fn: Microbatches, data: Any) -> list[tuple[Any, torch.Tensor]]:
    result = []
    for position, item in enumerate(fn(data)):
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise TypeError(f"microbatch {position} must be (data, loss_scale)")
        batch, scale = item
        if isinstance(scale, torch.Tensor):
            if scale.ndim != 0 or scale.requires_grad or scale.device.type != "cpu":
                raise ValueError(
                    "loss_scale must be a scalar CPU tensor without gradients"
                )
            if not torch.isfinite(scale).item():
                raise ValueError("loss_scale must be finite")
            scale = scale.detach().clone()
        else:
            if isinstance(scale, bool) or not isinstance(scale, (int, float)):
                raise TypeError("loss_scale must be a number or scalar CPU tensor")
            if not math.isfinite(scale):
                raise ValueError("loss_scale must be finite")
            scale = torch.tensor(scale, dtype=torch.float32)
        result.append((batch, scale))
    if not result:
        raise ValueError("an update must contain at least one microbatch")
    return result


def on_device(value: Any, device: torch.device) -> Any:
    # Preserve repeated tensor identities in arbitrary nested inputs.
    tensors: dict[int, torch.Tensor] = {}

    def move(leaf: Any) -> Any:
        if not isinstance(leaf, torch.Tensor):
            return leaf
        if id(leaf) not in tensors:
            tensors[id(leaf)] = leaf.to(device=device)
        return tensors[id(leaf)]

    return tree_map(move, value)


def unpack_objective(value: Any) -> tuple[torch.Tensor, Any]:
    return normalize_objective_result(value, require_grad=False)


class ScaledObjective:
    """Both backends capture exactly this tensor-only objective."""

    def __init__(self, objective: Objective) -> None:
        self.objective = objective

    def __call__(
        self, model: nn.Module, data: Any, scale: torch.Tensor
    ) -> tuple[torch.Tensor, Any]:
        loss, metrics = unpack_objective(self.objective(model, data))
        return loss * scale, metrics
