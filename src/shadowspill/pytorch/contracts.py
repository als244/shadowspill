"""Public value contracts for the PyTorch frontend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


class PlanningError(RuntimeError):
    """Raised before execution when a requested plan cannot be constructed."""


class CaptureError(PlanningError):
    """Raised when PyTorch cannot represent the requested fixed graph."""


class CompilationError(PlanningError):
    """Raised when a captured structural task cannot be compiled."""

    def __init__(
        self,
        message: str,
        *,
        structural_abi: str | None = None,
        task_kind: str | None = None,
        operators: tuple[str, ...] = (),
    ) -> None:
        super().__init__(message)
        self.structural_abi = structural_abi
        self.task_kind = task_kind
        self.operators = operators


class ProfilingError(PlanningError):
    """Raised when an isolated task cannot be measured or audited."""

    def __init__(
        self,
        message: str,
        *,
        structural_abi: str | None = None,
        task_kind: str | None = None,
        operators: tuple[str, ...] = (),
    ) -> None:
        super().__init__(message)
        self.structural_abi = structural_abi
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


def _contiguous_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride = 1
    result: list[int] = []
    for dimension in reversed(shape):
        result.append(stride)
        stride *= max(dimension, 1)
    result.reverse()
    return tuple(result)


@dataclass(frozen=True, slots=True)
class TensorSpec:
    """Storage-free representative tensor geometry used during planning."""

    shape: tuple[int, ...]
    dtype: torch.dtype
    stride: tuple[int, ...] | None = None
    requires_grad: bool = False
    layout: torch.layout = torch.strided

    def __post_init__(self) -> None:
        if any(not isinstance(value, int) or value < 0 for value in self.shape):
            raise ValueError("TensorSpec shape must contain non-negative integers")
        if not isinstance(self.dtype, torch.dtype):
            raise TypeError("TensorSpec dtype must be a torch.dtype")
        if self.layout is not torch.strided:
            raise ValueError("fixed-shape v1 supports strided TensorSpec values")
        if self.stride is not None and (
            len(self.stride) != len(self.shape)
            or any(not isinstance(value, int) or value < 0 for value in self.stride)
        ):
            raise ValueError("TensorSpec stride must match rank and be non-negative")

    @property
    def resolved_stride(self) -> tuple[int, ...]:
        """Return the authored stride or the standard contiguous geometry."""

        return self.stride or _contiguous_stride(self.shape)

    @property
    def storage_nbytes(self) -> int:
        """Smallest storage extent that can represent the fixed strided tensor."""

        if not self.shape or any(dimension == 0 for dimension in self.shape):
            elements = 0 if self.shape else 1
        else:
            elements = 1 + sum(
                (dimension - 1) * stride
                for dimension, stride in zip(
                    self.shape, self.resolved_stride, strict=True
                )
            )
        return elements * self.dtype.itemsize


@dataclass(frozen=True, slots=True)
class ObjectiveResult:
    """Training loss plus metrics that are not differentiated."""

    loss: torch.Tensor
    metrics: Any = None
