"""Learning-rate schedules: each gives the rate at a step of a run of ``steps``."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Constant:
    """The same rate at every step."""

    lr: float

    def rate(self, step: int, steps: int) -> float:
        return self.lr


@dataclass(frozen=True)
class WarmupCosine:
    """Linear warmup to ``lr``, then cosine decay to ``min_lr`` at the last step."""

    lr: float
    min_lr: float
    warmup_steps: int

    def rate(self, step: int, steps: int) -> float:
        if step < self.warmup_steps:
            return self.lr * (step + 1) / self.warmup_steps
        progress = (step - self.warmup_steps) / max(1, steps - self.warmup_steps)
        cosine = 1 + math.cos(math.pi * progress)
        return self.min_lr + 0.5 * (self.lr - self.min_lr) * cosine
