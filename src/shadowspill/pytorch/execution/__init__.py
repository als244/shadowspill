"""Forward and accumulated-training task execution."""

from shadowspill.pytorch.diagnostics.execution import ExecutionTiming, StepDiagnostics

from .forward import ForwardExecutor
from .training import TrainingExecutor

__all__ = [
    "ExecutionTiming",
    "ForwardExecutor",
    "StepDiagnostics",
    "TrainingExecutor",
]
