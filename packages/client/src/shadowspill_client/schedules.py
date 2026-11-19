"""Stateless, zero-based schedules; any callable(step) also works."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Constant:
    """A schedule that gives the same value at every step.

    Calling it with a step number returns ``value``, for example
    ``optim={"lr": schedule(step)}``.

    Parameters
    ----------
    value : float
        The value at every step.
    """

    value: float

    def __call__(self, step: int) -> float:
        return self.value


@dataclass(frozen=True)
class WarmupCosine:
    """A learning rate schedule with linear warmup and cosine decay.

    Calling it with a zero-based step ``s`` returns
    ``lr * (s + 1) / warmup_steps`` while ``s < warmup_steps``, and then
    ``min_lr + (lr - min_lr) * (1 + cos(pi * p)) / 2`` with
    ``p = min(1, (s - warmup_steps) / max(1, total_steps - warmup_steps - 1))``:
    a rise to ``lr``, a half cosine down to ``min_lr``, then ``min_lr``.

    Parameters
    ----------
    lr : float
        Peak rate, reached at the end of warmup.
    min_lr : float
        Final rate, between 0 and ``lr`` inclusive.
    warmup_steps : int
        Number of warmup steps; zero starts the decay at ``lr``.
    total_steps : int
        Number of steps the warmup and decay span; more than
        ``warmup_steps``.

    Raises
    ------
    ValueError
        If ``warmup_steps`` is negative or not less than ``total_steps``, or
        ``min_lr`` is not between 0 and ``lr``; and from a call, if the step
        is negative.
    """

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
