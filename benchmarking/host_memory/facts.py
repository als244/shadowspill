"""Collect process-local resource facts without initializing a device context."""

from __future__ import annotations

import socket
from pathlib import Path

import torch

from .limits import Limit, Resources


def _cgroup_available_memory(directory: Path, maximum: int, used: int) -> int:
    """Estimate headroom without charging clean, unmapped file cache as live state.

    A prior checkpoint read can leave tens of GiB charged to memory.current
    after its process exits. The kernel can reclaim those pages for a new
    pool. Keep mapped files, shared memory, dirty/writeback pages and all
    unevictable memory charged; overlapping exclusions are conservative.
    """

    try:
        statistics = {
            name: int(value)
            for name, value in (
                line.split()
                for line in (directory / "memory.stat").read_text().splitlines()
            )
        }
    except (OSError, ValueError):
        statistics = {}
    excluded = sum(
        statistics.get(name, 0)
        for name in (
            "shmem",
            "file_mapped",
            "file_dirty",
            "file_writeback",
            "unevictable",
        )
    )
    reclaimable = min(used, max(0, statistics.get("file", 0) - excluded))
    return min(maximum, max(0, maximum - used + reclaimable))


def host_limits() -> tuple[Limit, ...]:
    meminfo = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        name, value = line.split(":", 1)
        meminfo[name] = int(value.strip().split()[0]) * 1024
    limits = [Limit("physical", meminfo["MemAvailable"])]
    # Every visible cgroup ancestor is a distinct capacity domain. Inode identity
    # distinguishes separate container cgroups even if both expose path '/'.
    for line in Path("/proc/self/cgroup").read_text().splitlines():
        hierarchy, controllers, name = line.split(":", 2)
        if hierarchy != "0" or controllers:
            continue
        root = Path("/sys/fs/cgroup")
        parts = [part for part in Path(name).parts if part not in ("/", ".", "..")]
        current = root.joinpath(*parts)
        if not current.exists():
            current = root
        while current.is_relative_to(root):
            maximum, used = current / "memory.max", current / "memory.current"
            if maximum.exists() and used.exists():
                value = maximum.read_text().strip()
                if value != "max":
                    available = _cgroup_available_memory(
                        current, int(value), int(used.read_text())
                    )
                    limits.append(Limit(f"cgroup:{current.stat().st_ino}", available))
            if current == root:
                break
            current = current.parent
    return tuple(limits)


def physical_device_id(device: torch.device) -> str | None:
    if device.type == "cpu":
        return None
    if device.index is None:
        raise ValueError(
            "resource preflight needs an explicitly resolved device ordinal"
        )
    if torch.version.hip:
        raw = torch.cuda._raw_device_uuid_amdsmi()
        index = torch.cuda._get_amdsmi_device_index(device)
    else:
        raw = torch.cuda._raw_device_uuid_nvml()
        index = torch.cuda._get_nvml_device_index(device)
    if raw is None or not 0 <= index < len(raw):
        raise RuntimeError(
            "cannot identify the physical device before runtime allocation"
        )
    return raw[index]


def local_resources(
    device: torch.device, spill_bytes: int, staging_bytes: int
) -> Resources:
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    host = socket.gethostname() + "/" + boot
    return Resources(
        host, physical_device_id(device), spill_bytes, staging_bytes, host_limits()
    )
