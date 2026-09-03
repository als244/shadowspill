"""The accelerator PyTorch drives for ShadowSpill, named once.

PyTorch exposes CUDA and ROCm devices under the same ``"cuda"`` device type
and the same ``torch.cuda`` frontend, so this is the only place the string
appears in the PyTorch layer.  Everything else says *device* for tensors,
placements, and ordinals, and *backend* for streams, events, allocators, and
the provider library behind them.
"""

from __future__ import annotations

from typing import Final

import torch

DEVICE_TYPE: Final = "cuda"


def accelerator_device(ordinal: int) -> torch.device:
    return torch.device(DEVICE_TYPE, ordinal)


def is_accelerator(device: torch.device) -> bool:
    return device.type == DEVICE_TYPE


def provider_version() -> str | None:
    """The CUDA or ROCm version PyTorch was built against, if any."""

    return torch.version.cuda or torch.version.hip


__all__ = ["DEVICE_TYPE", "accelerator_device", "is_accelerator", "provider_version"]
