"""What planning raises when it cannot answer.

One hierarchy, because callers catch `PlanningError` to mean "planning did
not produce a plan" without caring which phase gave up. It lives here rather
than beside any one phase: the errors carry no framework types, and the
planner has to raise and catch them without importing a framework.
"""

from __future__ import annotations


class PlanningError(RuntimeError):
    """Raised before execution when a requested plan cannot be constructed."""


class CaptureError(PlanningError):
    """Raised when the frontend cannot represent the requested fixed graph."""


class CompilationError(PlanningError):
    """Raised when a captured structural task cannot be compiled."""

    def __init__(
        self,
        message: str,
        *,
        structural_contract: str | None = None,
        task_kind: str | None = None,
        operators: tuple[str, ...] = (),
    ) -> None:
        super().__init__(message)
        self.structural_contract = structural_contract
        self.task_kind = task_kind
        self.operators = operators


class ProfilingError(PlanningError):
    """Raised when an isolated task cannot be measured or audited."""

    def __init__(
        self,
        message: str,
        *,
        structural_contract: str | None = None,
        task_kind: str | None = None,
        operators: tuple[str, ...] = (),
    ) -> None:
        super().__init__(message)
        self.structural_contract = structural_contract
        self.task_kind = task_kind
        self.operators = operators


class AdmissionError(PlanningError):
    """Raised when requested memory resources cannot be physically admitted."""


class PlanInfeasibleError(AdmissionError):
    """Raised when no schedule satisfies the declared planning constraints."""

    def __init__(
        self,
        message: str,
        *,
        kind: str,
        device_id: str | None = None,
        boundary_task_id: str | None = None,
        required_bytes: int | None = None,
        capacity_bytes: int | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.device_id = device_id
        self.boundary_task_id = boundary_task_id
        self.required_bytes = required_bytes
        self.capacity_bytes = capacity_bytes


class PlanSearchExhaustedError(PlanningError):
    """Raised when a bounded planner search stops without a proof either way."""


class InputGuardError(ValueError):
    """Raised before mutation when runtime inputs differ from the template."""


class ObjectiveError(PlanningError):
    """Raised when an objective does not satisfy the training contract."""


__all__ = [
    "AdmissionError",
    "CaptureError",
    "CompilationError",
    "InputGuardError",
    "ObjectiveError",
    "PlanInfeasibleError",
    "PlanSearchExhaustedError",
    "PlanningError",
    "ProfilingError",
]
