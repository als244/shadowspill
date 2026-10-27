"""Tensor roles and physical view extents."""

from __future__ import annotations

import torch

from shadowspill.errors import CaptureError
from shadowspill.ir import (
    ObjectRole,
)
from shadowspill.pytorch.accelerator import DEVICE_TYPE


def tensor_value_role(
    tensor: torch.Tensor,
    *,
    continuous_role: ObjectRole,
) -> ObjectRole:
    """Classify integer/boolean values as control objects."""

    return (
        continuous_role
        if tensor.is_floating_point() or tensor.is_complex()
        else ObjectRole.CONTROL
    )


def serialized_dtype_role(
    dtype_name: str,
    *,
    continuous_role: ObjectRole,
) -> ObjectRole:
    """Classify one storage-contract dtype without materializing a tensor."""

    name = dtype_name.removeprefix("torch.")
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise CaptureError(f"storage contract has unknown dtype {dtype_name!r}")
    return (
        continuous_role
        if dtype.is_floating_point or dtype.is_complex
        else ObjectRole.CONTROL
    )


def _view_extent_bytes(tensor: torch.Tensor) -> int:
    """Return the device storage span one strided tensor view occupies.

    This is what an object is worth in bytes: the reach of its locations
    from where it starts, which is what its alias group must hold and what
    moving it costs. It is not how many elements it has. A broadcast reads
    one location many times and so has more elements than bytes, and a view
    that steps over its storage has more bytes than elements; neither is
    the storage it occupies.

    A task may also return a value it allocated somewhere the plan does not
    manage. Such a value is still one of the task's outputs and is still
    passed on to whatever reads it, but no device storage holds it, which
    is what its contract says and what this has to agree with.
    """

    if tensor.numel() == 0 or tensor.device.type != DEVICE_TYPE:
        return 0
    if any(stride < 0 for stride in tensor.stride()):
        raise CaptureError("compiled task output has a negative stride")
    last_element = sum(
        (extent - 1) * stride
        for extent, stride in zip(tensor.shape, tensor.stride(), strict=True)
    )
    return int((last_element + 1) * tensor.element_size())
