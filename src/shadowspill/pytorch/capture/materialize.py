"""Explicit allocation for shared gradients at a task boundary."""

from __future__ import annotations

import torch


@torch.library.custom_op("shadowspill::materialize_gradient", mutates_args=())
def materialize_gradient(
    value: torch.Tensor, memory_format: str = "contiguous_format"
) -> torch.Tensor:
    """Give one gradient independent storage with its own planned lifetime.

    Inductor can remove ordinary clones of output slices and return the shared
    allocation again. This operation makes the required allocation explicit;
    its copy and temporary lifetime are measured as part of the backward task.
    """
    return value.clone(memory_format=_memory_format(memory_format))


@materialize_gradient.register_fake
def _fake_materialize(
    value: torch.Tensor, memory_format: str = "contiguous_format"
) -> torch.Tensor:
    return torch.empty_like(value, memory_format=_memory_format(memory_format))


def _memory_format(name: str) -> torch.memory_format:
    formats = {
        "contiguous_format": torch.contiguous_format,
        "channels_last": torch.channels_last,
        "channels_last_3d": torch.channels_last_3d,
    }
    if name not in formats:
        raise ValueError(f"Unsupported gradient memory format: {name!r}")
    return formats[name]


MATERIALIZE_GRADIENT = torch.ops.shadowspill.materialize_gradient.default


def _padded_view(value: torch.Tensor, elements: int) -> torch.Tensor:
    owner = value.new_empty((elements,))
    return owner.as_strided(value.shape, value.stride())


@torch.library.custom_op("shadowspill::pad_gradient_storage", mutates_args=())
def pad_gradient_storage(value: torch.Tensor, elements: int) -> torch.Tensor:
    """Keep the logical gradient shape while allocating equal-shard padding."""
    result = _padded_view(value, elements)
    result.copy_(value)
    return result


@pad_gradient_storage.register_fake
def _fake_pad(value: torch.Tensor, elements: int) -> torch.Tensor:
    return _padded_view(value, elements)


PAD_GRADIENT_STORAGE = torch.ops.shadowspill.pad_gradient_storage.default
