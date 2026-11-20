"""What planning raises when it cannot answer.

One hierarchy, because callers catch `PlanningError` to mean "planning did
not produce a plan" without caring which phase gave up. It lives here rather
than beside any one phase: the errors carry no framework types, and the
planner has to raise and catch them without importing a framework.
"""

from __future__ import annotations

from typing import Any


class PlanningError(RuntimeError):
    """Raised when a requested plan cannot be produced.

    The base of the planning errors.
    """


class CaptureError(PlanningError):
    """Raised when the requested computation cannot be represented as a fixed program.

    Capturing it, dividing it into tasks, chunking them or binding a built
    task to its storage contract can each refuse.
    """


class TaskPhaseError(PlanningError):
    """Raised when compiling or profiling fails on one task.

    It names the task when it is known. Its subclasses say which phase failed.

    Parameters
    ----------
    message : str
        The error message.
    structural_contract : str, optional
        The failing task's compatibility digest, which identifies its
        structural contract. None when the task is not known.
    task_kind : str, optional
        The failing task's kind: ``"forward"``, ``"backward"``,
        ``"inference"`` or ``"optimizer"``. None when not known.
    operators : tuple of str, default ()
        Names of the operators the failing task calls, sorted.
    """

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


class CompilationError(TaskPhaseError):
    """Raised when a captured task cannot be compiled.

    Parameters
    ----------
    message : str
        The error message.
    structural_contract : str, optional
        The failing task's compatibility digest, as in `TaskPhaseError`.
    task_kind : str, optional
        The failing task's kind, as in `TaskPhaseError`.
    operators : tuple of str, default ()
        Names of the operators the failing task calls, sorted.
    """


class ProfilingError(TaskPhaseError):
    """Raised when a task cannot be measured, or its measurement fails its audit.

    Parameters
    ----------
    message : str
        The error message.
    structural_contract : str, optional
        The failing task's compatibility digest, as in `TaskPhaseError`.
    task_kind : str, optional
        The failing task's kind, as in `TaskPhaseError`.
    operators : tuple of str, default ()
        Names of the operators the failing task calls, sorted.
    """


class AdmissionError(PlanningError):
    """Raised when the requested memory cannot be physically admitted.

    Planning raises its subclass `PlanInfeasibleError` when no schedule fits.
    A worker raises it when a physical budget check fails, or when the memory
    its process holds outside the pool exceeds the headroom reserved for it.
    """


class PlanInfeasibleError(AdmissionError):
    """Raised when no schedule satisfies the declared planning constraints.

    It is a proof that nothing fits, unlike `PlanSearchExhaustedError`.

    Parameters
    ----------
    message : str
        The error message.
    kind : str
        Name of the constraint that failed, such as ``"physical_layout"``,
        ``"analytic_capacity"`` or ``"required_capacity"``.
    device_id : str, optional
        The program device the shortfall is on, when one is to blame.
    boundary_task_id : str, optional
        The task at whose boundary the shortfall occurs. None at
        initialization or when not known.
    required_bytes : int, optional
        Bytes the failing constraint needs, when known.
    capacity_bytes : int, optional
        Bytes available to it, when known.
    diagnostics : tuple, default ()
        The search's records of the candidates it rejected and why.
    """

    def __init__(
        self,
        message: str,
        *,
        kind: str,
        device_id: str | None = None,
        boundary_task_id: str | None = None,
        required_bytes: int | None = None,
        capacity_bytes: int | None = None,
        diagnostics: tuple[Any, ...] = (),
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.device_id = device_id
        self.boundary_task_id = boundary_task_id
        self.required_bytes = required_bytes
        self.capacity_bytes = capacity_bytes
        self.diagnostics = diagnostics


class PlanSearchExhaustedError(PlanningError):
    """Raised when a bounded planner search stops without a proof either way.

    Parameters
    ----------
    message : str
        The error message.
    diagnostics : tuple, default ()
        The search's records of the candidates it evaluated.
    """

    def __init__(
        self,
        message: str,
        *,
        diagnostics: tuple[Any, ...] = (),
    ) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics


class InputGuardError(ValueError):
    """Raised when a call's inputs differ from those its program was captured with.

    It is raised before anything changes, when an input's shape, strides,
    element type, layout or gradient flag, or a static argument's value,
    differs from the capture.
    """


class ObjectiveError(PlanningError):
    """Raised when an objective does not satisfy the training contract.

    The objective returns its loss alone, as ``(loss, metrics)`` or as an
    `ObjectiveResult`. The loss is one floating-point or complex value, and
    requires gradients when the step trains; the metrics keep one structure
    across calls.
    """


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
    "TaskPhaseError",
]
