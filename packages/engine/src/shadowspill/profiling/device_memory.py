"""Observe per-process device overhead outside PyTorch's reserved cache."""

import os
from contextlib import suppress

import torch


def external_device_bytes(device: torch.device) -> int | None:
    """NVML is optional; an unavailable measurement remains explicitly missing."""
    try:
        import pynvml  # type: ignore[import-untyped]
    except ImportError:
        return None
    try:
        pynvml.nvmlInit()
        identifier = str(torch.cuda.get_device_properties(device).uuid)
        if not identifier.startswith("GPU-"):
            identifier = f"GPU-{identifier}"
        handle = pynvml.nvmlDeviceGetHandleByUUID(identifier)
        processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        used = next(
            (item.usedGpuMemory for item in processes if item.pid == os.getpid()), None
        )
        if used is None:
            return None
        return max(0, int(used) - torch.cuda.memory_reserved(device))
    except pynvml.NVMLError:
        return None
    finally:
        with suppress(pynvml.NVMLError):
            pynvml.nvmlShutdown()
