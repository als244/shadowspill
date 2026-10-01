"""Compute weights, optional optimizer masters and accumulated gradients."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .._types import ParameterObserver


class Weights:
    def __init__(
        self,
        model: nn.Module,
        master_dtype: torch.dtype | None,
        grad_dtype: torch.dtype | None,
    ) -> None:
        self.named = dict(model.named_parameters())
        self.trainable = {n: p for n, p in self.named.items() if p.requires_grad}
        self.masters = {
            name: nn.Parameter(weight.detach().to(master_dtype))
            for name, weight in self.trainable.items()
            if master_dtype is not None and weight.dtype != master_dtype
        }
        self.grad_dtype = grad_dtype
        self.sums: dict[str, torch.Tensor] = {}

    def parameters(self) -> list[nn.Parameter]:
        return [self.masters.get(n, p) for n, p in self.named.items()]

    def accumulate(self) -> None:
        if self.grad_dtype is None:
            return
        for name, weight in self.trainable.items():
            if weight.grad is None or weight.grad.dtype == self.grad_dtype:
                continue
            gradient = weight.grad.to(self.grad_dtype)
            weight.grad = None
            if name in self.sums:
                self.sums[name].add_(gradient)
            else:
                self.sums[name] = gradient

    @torch.no_grad()
    def observe(self, observer: ParameterObserver | None) -> dict[str, Any]:
        if observer is None:
            return {}
        return {
            name: observer(weight, gradient)
            for name, weight in self.trainable.items()
            if (gradient := self.sums.get(name, weight.grad)) is not None
        }

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        for name, weight in self.trainable.items():
            gradient = self.sums.pop(name, weight.grad)
            parameter = self.masters.get(name, weight)
            parameter.grad = None if gradient is None else gradient.to(parameter.dtype)
            if parameter is not weight:
                weight.grad = None
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        # A parameter-group callback can deliberately omit trainable parameters.
        # Do not let their previous update's gradients leak into the next one.
        for weight in self.trainable.values():
            weight.grad = None
        with torch.no_grad():
            for name, master in self.masters.items():
                self.trainable[name].copy_(master)
