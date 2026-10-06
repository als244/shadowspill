"""Explicit allocation for parameter-gradient slices at a task boundary."""

from __future__ import annotations

import torch


@torch.library.custom_op("shadowspill::materialize_gradient", mutates_args=())
def materialize_gradient(value: torch.Tensor) -> torch.Tensor:
    """Give one gradient independent storage with its own optimizer lifetime.

    Inductor can remove ordinary clones of output slices and return the shared
    allocation again. This operation makes the required allocation explicit;
    its copy and temporary lifetime are measured as part of the backward task.
    """
    return value.clone(memory_format=torch.contiguous_format)


@materialize_gradient.register_fake
def _fake_materialize(value: torch.Tensor) -> torch.Tensor:
    return torch.empty(value.shape, dtype=value.dtype, device=value.device)


MATERIALIZE_GRADIENT = torch.ops.shadowspill.materialize_gradient.default
