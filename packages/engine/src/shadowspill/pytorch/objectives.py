"""Explicit objective scaling for captured calls and independent references."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import nn

from shadowspill.pytorch.contracts import normalize_objective_result


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
