"""The reference backend: model, gradients and optimizer state on the device."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from training import checkpoints, models
from training.backends import GIB, Microbatch, Setup


class PyTorch:
    """An ordinary PyTorch loop, compiled with ``torch.compile`` unless told not."""

    plan = None  # nothing is planned
    planning = None

    def __init__(self, compile: bool = True, device: str | None = None) -> None:
        self.compile = compile
        accelerator = torch.accelerator.current_accelerator() or torch.device("cpu")
        self.device = torch.device(device) if device else accelerator
        # Checkpoints load onto the device, where an optimizer that keeps its
        # step counts on the device left them.
        self.checkpoint_device = str(self.device)

    def setup(self, setup: Setup) -> None:
        if setup.max_tokens_per_microbatch is None:
            raise ValueError("the PyTorch backend needs max_tokens_per_microbatch")
        per_microbatch = setup.max_tokens_per_microbatch
        self.geometry = (per_microbatch, setup.max_tokens_per_step // per_microbatch)
        self.module = setup.module
        self.module.to_empty(device="cpu")
        torch.manual_seed(setup.seed)
        models.initialize(self.module)
        self.module.to(self.device)
        self.masters = _Masters(self.module, setup.master_dtype, setup.grad_dtype)
        self.optimizer = setup.optimizer(
            self.masters.parameters(), **setup.optimizer_args
        )
        self.steps = 0
        self.loss = (
            torch.compile(self.module, fullgraph=True, dynamic=False)
            if self.compile
            else self.module
        )

    def step(self, microbatches: list[Microbatch], lr: float | None) -> list[float]:
        if lr is not None:
            _set_learning_rate(self.optimizer, lr)
        losses = []
        for microbatch in microbatches:
            loss = self.loss(*(value.to(self.device) for value in microbatch))
            loss.backward()  # gradients add up across microbatches, as ShadowSpill's do
            self.masters.accumulate()
            losses.append(loss.detach())
        self.masters.step(self.optimizer)
        self.steps += 1
        return [loss.item() for loss in losses]

    def synchronize(self) -> None:
        if self.device.type != "cpu":
            torch.accelerator.synchronize(self.device)

    @torch.no_grad()
    def evaluate(self, microbatches: list[Microbatch]) -> list[float]:
        return [
            self.loss(*(value.to(self.device) for value in microbatch)).item()
            for microbatch in microbatches
        ]

    def device_peak_gib(self) -> float:
        """The most the device has held at once; 0 for a run on the CPU."""

        if self.device.type == "cpu":
            return 0.0
        return torch.accelerator.max_memory_allocated(self.device) / GIB

    def save(self, path: Path) -> None:
        checkpoints.save(
            path, self.module, self.optimizer, self.steps, self.masters.masters
        )

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.steps = checkpoints.load(
            state, self.module, self.optimizer, self.masters.masters
        )

    def close(self) -> None:
        pass


class _Masters:
    """What ShadowSpill's ``master_dtype`` and ``grad_dtype`` do, eagerly.

    Every weight trained at another dtype than ``master_dtype`` has a master
    copy at it, which the optimizer steps in the weight's place; after each
    step the weight is written from its master. Gradients are summed over a
    step's microbatches at ``grad_dtype`` -- each microbatch's cast as it
    arrives, autograd having computed it at the weight's dtype, where
    ShadowSpill's backward gives it at ``grad_dtype`` directly -- or at the
    weights' own dtype, in ``.grad``, when that is not given; the optimizer
    gets each at its parameter's dtype.
    """

    def __init__(
        self,
        module: nn.Module,
        master_dtype: torch.dtype | None,
        grad_dtype: torch.dtype | None,
    ) -> None:
        self.named = dict(module.named_parameters())
        self.weights = {
            name: weight for name, weight in self.named.items() if weight.requires_grad
        }
        self.masters = {
            name: nn.Parameter(weight.detach().to(master_dtype))
            for name, weight in self.weights.items()
            if master_dtype is not None
            and weight.dtype.is_floating_point
            and weight.dtype != master_dtype
        }
        self.grad_dtype = grad_dtype
        self.sums: dict[str, torch.Tensor] = {}

    def parameters(self) -> list[nn.Parameter]:
        """The optimizer's parameters, in the model's order."""

        return [self.masters.get(name, weight) for name, weight in self.named.items()]

    def accumulate(self) -> None:
        """Add this microbatch's gradients to the ones kept at ``grad_dtype``."""

        if self.grad_dtype is None:
            return
        for name, weight in self.weights.items():
            if weight.grad is None or weight.grad.dtype == self.grad_dtype:
                continue
            gradient = weight.grad.to(self.grad_dtype)
            weight.grad = None
            kept = self.sums.get(name)
            self.sums[name] = gradient if kept is None else kept.add_(gradient)

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        """Step the optimizer on the summed gradients, and write each weight
        from its master."""

        for name, weight in self.weights.items():
            gradient = self.sums.pop(name, weight.grad)
            parameter = self.masters.get(name, weight)
            parameter.grad = None if gradient is None else gradient.to(parameter.dtype)
            if parameter is not weight:
                weight.grad = None
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            for name, master in self.masters.items():
                self.weights[name].copy_(master)


def _set_learning_rate(optimizer: torch.optim.Optimizer, lr: float) -> None:
    """Give every group ``lr``, in place when the optimizer holds it in a tensor
    (so that anything reading that tensor sees the new rate)."""

    for group in optimizer.param_groups:
        if isinstance(group["lr"], torch.Tensor):
            group["lr"].fill_(lr)
        else:
            group["lr"] = lr
