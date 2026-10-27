"""User data stays a pytree; only loss scaling is added by the runner."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import nn
from torch.utils._pytree import tree_map

from shadowspill.pytorch.contracts import normalize_objective_result


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
    """Apply the text recipe's explicit per-example loss scale."""

    def __init__(self, objective: Callable[[nn.Module, Any], Any]) -> None:
        self.objective = objective

    def __call__(
        self, model: nn.Module, data: Any, scale: torch.Tensor
    ) -> tuple[torch.Tensor, Any]:
        loss, metrics = unpack_objective(self.objective(model, data))
        return loss * scale, metrics
