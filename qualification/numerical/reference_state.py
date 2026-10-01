"""PyTorch reference for independent master and accumulation dtypes.

This implements the public training contract without using ShadowSpill's
capture, state import, or execution code.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import nn


class ReferenceState:
    def __init__(
        self,
        model: nn.Module,
        optimizer_factory: Callable[..., Any],
        *,
        master_dtype: torch.dtype | None,
        grad_dtype: torch.dtype | None,
    ) -> None:
        self.model = model
        self.weights = [weight for weight in model.parameters() if weight.requires_grad]
        self.stepped = (
            self.weights
            if master_dtype is None
            else [
                weight.detach().to(master_dtype).clone().requires_grad_()
                for weight in self.weights
            ]
        )
        self.grad_dtype = grad_dtype
        self.optimizer = optimizer_factory(self.stepped)
        self.sums: list[torch.Tensor | None] = [None] * len(self.weights)

    def begin_step(self) -> None:
        self.model.zero_grad(set_to_none=True)
        self.optimizer.zero_grad(set_to_none=True)
        self.sums = [None] * len(self.weights)

    def accumulate(self) -> None:
        """Add each microbatch gradient at the requested accumulation dtype."""
        for index, weight in enumerate(self.weights):
            gradient = weight.grad
            if gradient is not None:
                value = gradient.detach().to(self.grad_dtype or weight.dtype)
                total = self.sums[index]
                if total is None:
                    self.sums[index] = value.clone()
                else:
                    total.add_(value)
            weight.grad = None

    def step(self) -> None:
        for parameter, total in zip(self.stepped, self.sums, strict=True):
            parameter.grad = None if total is None else total.to(parameter.dtype)
        self.optimizer.step()
        if self.stepped is not self.weights:
            with torch.no_grad():
                for weight, master in zip(self.weights, self.stepped, strict=True):
                    weight.copy_(master)
        self.sums = [None] * len(self.weights)

    def model_state(self) -> dict[str, torch.Tensor]:
        """A training checkpoint saves masters in place of derived weights."""
        result = dict(self.model.state_dict())
        if self.stepped is not self.weights:
            masters = {
                id(weight): master.detach()
                for weight, master in zip(self.weights, self.stepped, strict=True)
            }
            for name, weight in self.model.named_parameters(remove_duplicate=False):
                if id(weight) in masters:
                    result[name] = masters[id(weight)]
        return result
