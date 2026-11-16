"""Stateless, zero-based schedules; any callable(step) also works."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Constant:
    value: float

    def __call__(self, step: int) -> float:
        return self.value


@dataclass(frozen=True)
class WarmupCosine:
    lr: float
    min_lr: float
    warmup_steps: int
    total_steps: int

    def __post_init__(self) -> None:
        if self.total_steps < 1 or not 0 <= self.warmup_steps < self.total_steps:
            raise ValueError("require total_steps > warmup_steps >= 0")
        if not 0 <= self.min_lr <= self.lr:
            raise ValueError("require 0 <= min_lr <= lr")

    def __call__(self, step: int) -> float:
        if step < 0:
            raise ValueError("schedule step must be nonnegative")
        if step < self.warmup_steps:
            return self.lr * (step + 1) / self.warmup_steps
        width = max(1, self.total_steps - self.warmup_steps - 1)
        progress = min(1.0, (step - self.warmup_steps) / width)
        return self.min_lr + 0.5 * (self.lr - self.min_lr) * (
            1 + math.cos(math.pi * progress)
        )
