"""The accelerator PyTorch drives for ShadowSpill, named once.

PyTorch exposes CUDA and ROCm devices under the same ``"cuda"`` device type
and the same ``torch.cuda`` frontend, so this is the only place the string
appears in the PyTorch layer.  Everything else says *device* for tensors,
placements, and ordinals, and *backend* for streams, events, allocators, and
the provider library behind them.
"""

from __future__ import annotations

import os
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


__all__ = [
    "DEVICE_TYPE",
    "accelerator_device",
    "is_accelerator",
    "provider_version",
    "resolve_device",
]


def resolve_device(
    value: int | str | torch.device | None = "auto", *, allow_cpu: bool = False
) -> torch.device:
    """Resolve a process-local device without creating an accelerator context.

    A single visible device always has ordinal zero. With several visible
    devices, a launched process uses LOCAL_RANK; an ordinary unlaunched process
    defaults to zero. Global ranks are never interpreted as device ordinals.
    """
    if isinstance(value, bool):
        raise TypeError("device must be an ordinal or a device name")
    selected = (
        None
        if value in (None, "auto")
        else accelerator_device(value)
        if isinstance(value, int)
        else torch.device(value)
    )
    if selected is not None and selected.type == "cpu" and allow_cpu:
        return selected
    if selected is not None and not is_accelerator(selected):
        raise ValueError(f"unsupported execution device: {selected}")
    count = torch.cuda.device_count()
    if count == 0:
        if selected is None and allow_cpu:
            return torch.device("cpu")
        raise RuntimeError("no accelerator device is visible to this process")
    index = None if selected is None else selected.index
    if index is None:
        if count == 1:
            index = 0
        elif "LOCAL_RANK" in os.environ:
            index = int(os.environ["LOCAL_RANK"])
        elif int(os.environ.get("WORLD_SIZE", "1")) > 1:
            raise ValueError(
                "set LOCAL_RANK or select an explicit device for this process"
            )
        else:
            index = 0
    if not 0 <= index < count:
        raise ValueError(f"device ordinal {index} is outside {count} visible devices")
    return accelerator_device(index)
