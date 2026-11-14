"""Resolve W&B metric indices from device identity, not CUDA/local-rank order."""

from __future__ import annotations

import ctypes
import socket
from contextlib import suppress
from typing import Any

import torch

from shadowspill.pytorch.accelerator import resolve_device


def monitor_index(uuid: str) -> int:
    """Return the system monitor's index for a physical GPU UUID."""
    if torch.version.hip:
        identities = torch.cuda._raw_device_uuid_amdsmi()
        if identities is None:
            raise RuntimeError("cannot read AMD GPU identities")
        return identities.index(uuid)

    # NVML ordinals can differ from CUDA ordinals even without visibility masks.
    # Looking up the selected UUID also avoids enumerating unrelated faulty GPUs.
    library = ctypes.CDLL("libnvidia-ml.so.1")

    def check(code: int) -> None:
        if code != 0:
            raise RuntimeError(f"NVML device identification failed (status {code})")

    check(library.nvmlInit_v2())
    try:
        handle = ctypes.c_void_p()
        check(library.nvmlDeviceGetHandleByUUID(uuid.encode(), ctypes.byref(handle)))
        index = ctypes.c_uint()
        check(library.nvmlDeviceGetIndex(handle, ctypes.byref(index)))
        return index.value
    finally:
        check(library.nvmlShutdown())


def device_record(device: str | int | torch.device | None) -> dict[str, Any]:
    """Describe the process device once, outside training and compiled tasks."""
    if device is None and torch.cuda.is_initialized():  # type: ignore[no-untyped-call]
        selected = torch.device("cuda", torch.cuda.current_device())
    else:
        selected = resolve_device(device, allow_cpu=True)
    record: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "device": str(selected),
        "name": None,
        "uuid": None,
        "wandb_gpu_index": None,
        "metric_prefix": None,
    }
    if selected.type == "cpu":
        return record
    properties = torch.cuda.get_device_properties(selected)
    uuid = str(properties.uuid)
    if not torch.version.hip and not uuid.startswith(("GPU-", "MIG-")):
        uuid = "GPU-" + uuid
    index = monitor_index(uuid)
    record.update(
        name=properties.name,
        uuid=uuid,
        wandb_gpu_index=index,
        metric_prefix=f"gpu.{index}.",
    )
    return record


def aggregate_devices(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Map member devices to indices visible from the aggregate run's process.

    Remote hosts and devices hidden from this process retain their rank mapping,
    but have no automatic system series in the aggregate run.
    """
    host = socket.gethostname()
    result = []
    for value in records:
        record = dict(value)
        index = None
        if record["hostname"] == host and record["uuid"] is not None:
            # Its rank run still owns telemetry if this process cannot see it.
            with suppress(OSError, RuntimeError, ValueError):
                index = monitor_index(record["uuid"])
        record["aggregate_gpu_index"] = index
        record["aggregate_metric_prefix"] = None if index is None else f"gpu.{index}."
        result.append(record)
    return result


def monitored_options(options: dict[str, Any], indices: list[int]) -> dict[str, Any]:
    """Copy caller options while restricting only GPU system telemetry."""
    import wandb

    result = dict(options)
    supplied = result.get("settings")
    settings = (
        supplied.model_copy(deep=True)
        if isinstance(supplied, wandb.Settings)
        else wandb.Settings(**(supplied or {}))
    )
    # W&B serializes [] as an unset filter (all GPUs). -1 matches no system
    # GPU index, while keeping CPU/network/disk monitoring for CPU-only ranks.
    settings.x_stats_gpu_device_ids = indices or [-1]
    result["settings"] = settings
    return result
