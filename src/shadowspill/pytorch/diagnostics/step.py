"""Deferred, immutable diagnostics returned by one planned training step."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from .execution import ExecutionTiming, StepDiagnostics


@dataclass(frozen=True, slots=True)
class StepResult:
    """Detached device results for every microbatch in one logical step."""

    objectives: tuple[torch.Tensor, ...]
    metrics: tuple[Any, ...]
    step_number: int
    diagnostics: DiagnosticsHandle | None = None


class DiagnosticsHandle:
    """Deferred collection of an explicitly enabled detailed execution trace."""

    def __init__(self, collector: Callable[[], StepDiagnostics]) -> None:
        self._collector = collector
        self._result: StepDiagnostics | None = None

    @property
    def resolved(self) -> bool:
        return self._result is not None

    def result(self) -> StepDiagnostics:
        """Synchronize trace completion and return immutable step evidence."""

        if self._result is None:
            self._result = self._collector()
        return self._result

    wait = result


__all__ = [
    "DiagnosticsHandle",
    "ExecutionTiming",
    "StepDiagnostics",
    "StepResult",
]
