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
    """A tensor's geometry, standing in for the tensor in a build's examples.

    `build_forward`, `build_forward_backward` and `build_step` accept one in
    place of a tensor in their example arguments, so an example needs no
    storage. A run's tensor at that position must have this shape, strides,
    dtype and gradient flag, a storage offset of zero, and a storage of its
    own of exactly `TensorSpec.storage_nbytes` bytes.

    Parameters
    ----------
    shape : tuple of int
        The tensor's shape. Each size at least zero.
    dtype : torch.dtype
        The element type.
    stride : tuple of int, optional
        Strides, in elements, one per dimension, each at least zero. None, the
        default, means contiguous strides.
    requires_grad : bool, default False
        Whether the tensor requires gradients.
    layout : torch.layout, default torch.strided
        The layout. Only ``torch.strided`` is accepted.

    Raises
    ------
    ValueError
        If a size or stride is negative or not an int, ``stride`` has another
        length than ``shape``, or ``layout`` is not ``torch.strided``.
    TypeError
        If ``dtype`` is not a ``torch.dtype``.
    """

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
        """The strides: ``stride``, or the contiguous strides of ``shape``."""

        return self.stride or contiguous_stride(self.shape)

    @property
    def storage_nbytes(self) -> int:
        """Bytes of the smallest storage that holds every element at these strides."""

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
    """An objective's loss, with metrics that are not differentiated.

    An objective may return one in place of its loss alone or a
    ``(loss, metrics)`` pair. A run's `TrainingResult` holds each example's
    loss and metrics.

    Parameters
    ----------
    loss : torch.Tensor
        The value the backward differentiates: one floating-point or complex
        element that requires gradients. Capture fails with `ObjectiveError`
        otherwise.
    metrics : object, optional
        Any structure of values returned beside the loss, keeping one
        structure across calls and examples. Its tensors are returned
        detached; its other values are copied once for each example, during
        build, and every run returns that example's copies. None by default.
    """

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
