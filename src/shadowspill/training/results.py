"""Completed training observations; these contain no Executor-owned GPU storage."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .observations import MetricSummary, StepObservations


@dataclass(frozen=True)
class StepResult(StepObservations):
    step: int = 0
    seconds: float = 0.0
    hyperparams: Mapping[str, Any] = field(default_factory=dict)
    summary: MetricSummary = field(default_factory=MetricSummary)

    @property
    def loss(self) -> float:
        return sum(self.losses)
