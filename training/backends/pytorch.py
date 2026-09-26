"""The reference backend: model, gradients and optimizer state on the device."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

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
        self.optimizer = setup.optimizer(
            self.module.parameters(), **setup.optimizer_args
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
            losses.append(loss.detach())
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
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
        checkpoints.save(path, self.module, self.optimizer, self.steps)

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.steps = checkpoints.load(state, self.module, self.optimizer)

    def close(self) -> None:
        pass


def _set_learning_rate(optimizer: torch.optim.Optimizer, lr: float) -> None:
    """Give every group ``lr``, in place when the optimizer holds it in a tensor
    (so that anything reading that tensor sees the new rate)."""

    for group in optimizer.param_groups:
        if isinstance(group["lr"], torch.Tensor):
            group["lr"].fill_(lr)
        else:
            group["lr"] = lr
