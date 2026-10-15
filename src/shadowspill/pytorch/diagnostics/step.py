"""Deferred, immutable diagnostics returned by one planned training step."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch

from shadowspill.diagnostics.step import (
    PhaseTimingComparison,
    StepDiagnostics,
    StepTimingSummary,
)


@dataclass(frozen=True, slots=True)
class StepResult:
    """Raw detached tensors: per-microbatch results and per-step observations."""

    objectives: tuple[torch.Tensor, ...]
    metrics: tuple[Any, ...]
    step_number: int
    diagnostics: DiagnosticsHandle | None = None
    #: Pre-update compute weights/final gradients observed once per parameter.
    #: Values keep the callback's pytree; tensor leaves remain on the device.
    parameter_metrics: dict[str, Any] = field(default_factory=dict)


class DiagnosticsHandle:
    """Deferred collection of an explicitly enabled detailed execution trace.

    Collection runs once. The measurement is consumed by that attempt whether
    it succeeds or raises, so a second attempt could only fail with a message
    about the measurement rather than the cause; a failed collection keeps its
    error and raises it again instead.
    """

    def __init__(self, collector: Callable[[], StepDiagnostics]) -> None:
        self._collector = collector
        self._result: StepDiagnostics | None = None
        self._error: BaseException | None = None

    @property
    def resolved(self) -> bool:
        """Whether collection has run, whether it produced evidence or raised."""

        return self._result is not None or self._error is not None

    def result(self) -> StepDiagnostics:
        """Synchronize trace completion and return immutable step evidence."""

        if self._error is not None:
            raise self._error
        if self._result is None:
            try:
                self._result = self._collector()
            except BaseException as error:
                self._error = error
                raise
        return self._result

    wait = result


__all__ = [
    "DiagnosticsHandle",
    "PhaseTimingComparison",
    "StepDiagnostics",
    "StepResult",
    "StepTimingSummary",
]
