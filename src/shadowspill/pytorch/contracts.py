"""Public value contracts for the PyTorch frontend.

The planning error hierarchy lives in `shadowspill.errors`, not here: it
carries no torch types, so the planner can raise and catch it without
importing torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from shadowspill.errors import ObjectiveError


def contiguous_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    """The stride PyTorch gives a contiguous tensor of this shape.

    A zero-length dimension contributes nothing to the running product,
    which is why the product takes ``max(dimension, 1)``: the stride of a
    dimension outside an empty one is still the size of a full row.
    """

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

        return self.stride or contiguous_stride(self.shape)

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


def normalize_objective_result(
    value: torch.Tensor | ObjectiveResult | tuple[torch.Tensor, Any],
    *,
    require_grad: bool,
) -> tuple[torch.Tensor, Any]:
    """Validate the scalar differentiable objective contract."""

    if isinstance(value, ObjectiveResult):
        loss, metrics = value.loss, value.metrics
    elif isinstance(value, tuple) and len(value) == 2:
        loss, metrics = value
    else:
        loss, metrics = value, None
    if not isinstance(loss, torch.Tensor):
        raise ObjectiveError(
            "objective must return a tensor, (loss, metrics), or ObjectiveResult"
        )
    if loss.numel() != 1:
        raise ObjectiveError(
            f"objective loss must be scalar, got shape {tuple(loss.shape)}"
        )
    if not (loss.is_floating_point() or loss.is_complex()):
        raise ObjectiveError("objective loss must be floating point or complex")
    if require_grad and not loss.requires_grad:
        raise ObjectiveError("objective loss must require gradients")
    return loss, metrics
