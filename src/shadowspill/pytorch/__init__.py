"""Public PyTorch values and planning entrypoints for ShadowSpill."""

from .contracts import (
    CaptureError,
    InputGuardError,
    ObjectiveError,
    ObjectiveResult,
    PlanningError,
    TensorSpec,
)
from .public import (
    PlannedForward,
    PlannedTrainStep,
    PlanReport,
    StepResult,
    forward_pass,
    plan,
)

__all__ = [
    "CaptureError",
    "InputGuardError",
    "ObjectiveError",
    "ObjectiveResult",
    "PlanReport",
    "PlannedForward",
    "PlannedTrainStep",
    "PlanningError",
    "StepResult",
    "TensorSpec",
    "forward_pass",
    "plan",
]
