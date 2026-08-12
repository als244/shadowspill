"""Installation of ShadowSpill through PyTorch's supported allocator API."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ._abi import (
    ADAPTER_ABI_VERSION,
    RUNTIME_ABI_VERSION,
    AdapterCapabilities,
    AdapterConfig,
    PhysicalAdmission,
    PhysicalMemory,
    configure_adapter_library,
)


class AllocatorInstallError(RuntimeError):
    """Raised when the process-global PyTorch allocator cannot be installed."""


@dataclass(frozen=True)
class InstalledAllocator:
    """Process-lifetime owners for PyTorch's selected allocator callbacks."""

    library: Any
    allocator: Any
    path: Path
    admission: PhysicalAdmission


_installed: InstalledAllocator | None = None


def installed_allocator() -> InstalledAllocator | None:
    """Return the process-lifetime allocator owner, if already selected."""

    return _installed


def resize_host_arena(
    installed: InstalledAllocator, *, host_arena_bytes: int, host_budget_bytes: int
) -> None:
    """Grow the pinned spill pool during planning without exceeding its cap."""

    current = int(installed.admission.host_arena_bytes)
    if host_arena_bytes < current:
        raise AllocatorInstallError("pinned host arena cannot shrink")
    if host_arena_bytes == current:
        return
    if host_arena_bytes > host_budget_bytes or current + host_arena_bytes > (
        host_budget_bytes
    ):
        raise AllocatorInstallError(
            "planning-time pinned arena replacement exceeds host_budget"
        )
    status = int(
        installed.library.shadowspill_pytorch_resize_host_arena(host_arena_bytes)
    )
    if status != 0:
        raise AllocatorInstallError(
            f"pinned host arena growth failed with status {status}"
        )
    admission = PhysicalAdmission()
    status = int(
        installed.library.shadowspill_pytorch_physical_admission(
            ctypes.byref(admission)
        )
    )
    if status != 0 or int(admission.host_arena_bytes) != host_arena_bytes:
        raise AllocatorInstallError("pinned host arena admission was not updated")
    installed.admission.host_arena_bytes = admission.host_arena_bytes


def _function_pointer(library: Any, name: str) -> int:
    try:
        symbol = getattr(library, name)
    except AttributeError as exc:
        raise AllocatorInstallError(f"adapter has no {name!r} export") from exc
    pointer = ctypes.cast(symbol, ctypes.c_void_p).value
    if pointer is None:
        raise AllocatorInstallError(f"adapter export {name!r} is null")
    return pointer


def install_allocator(
    library_path: str | Path,
    *,
    device_ordinal: int,
    device_budget_bytes: int,
    provider_headroom_bytes: int,
    host_arena_bytes: int,
    worker_poll_nanoseconds: int = 100_000,
) -> InstalledAllocator:
    """Install the process-global CUDA allocator before PyTorch CUDA init.

    This is an internal frontend primitive. Public planning computes physical
    admission before invoking it. Installation is intentionally irreversible
    for the process lifetime.
    """

    global _installed
    if device_ordinal < 0:
        raise AllocatorInstallError("device ordinal must be non-negative")
    if device_budget_bytes <= 0:
        raise AllocatorInstallError("device budget must be positive")
    if provider_headroom_bytes < 0 or provider_headroom_bytes >= device_budget_bytes:
        raise AllocatorInstallError(
            "provider headroom must be non-negative and smaller than device budget"
        )
    if host_arena_bytes < 0:
        raise AllocatorInstallError("host arena bytes must be non-negative")
    if worker_poll_nanoseconds < 0:
        raise AllocatorInstallError("progress poll interval must be non-negative")
    path = Path(library_path).expanduser().resolve()
    if not path.is_file():
        raise AllocatorInstallError(f"PyTorch adapter does not exist: {path}")
    if _installed is not None:
        raise AllocatorInstallError("ShadowSpill's allocator is already installed")
    if torch.version.cuda is None:
        raise AllocatorInstallError("a CUDA-enabled PyTorch build is required")
    cuda: Any = torch.cuda
    if cuda.is_initialized():
        raise AllocatorInstallError(
            "PyTorch CUDA was initialized before ShadowSpill allocator installation"
        )
    library = ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
    configure_adapter_library(library)
    capabilities = AdapterCapabilities()
    status = int(
        library.shadowspill_pytorch_adapter_capabilities(ctypes.byref(capabilities))
    )
    if (
        status != 0
        or capabilities.abi_version != ADAPTER_ABI_VERSION
        or capabilities.runtime_abi_version != RUNTIME_ABI_VERSION
        or capabilities.debug_task_host_timing != 1
        or capabilities.runtime_trace != 1
    ):
        raise AllocatorInstallError("PyTorch adapter capability/ABI validation failed")
    allocator: Any = cuda.memory.CUDAPluggableAllocator(
        str(path),
        "shadowspill_pytorch_cuda_malloc",
        "shadowspill_pytorch_cuda_free",
    )
    record_stream_pointer = _function_pointer(
        library, "shadowspill_pytorch_cuda_record_stream"
    )
    native_allocator = allocator.allocator()
    set_record_stream = getattr(native_allocator, "set_record_stream_fn", None)
    if set_record_stream is None or not callable(set_record_stream):
        raise AllocatorInstallError(
            "this PyTorch build lacks the required record-stream callback"
        )
    set_record_stream(record_stream_pointer)
    config = AdapterConfig(
        abi_version=ADAPTER_ABI_VERSION,
        device_ordinal=device_ordinal,
        device_budget_bytes=device_budget_bytes,
        provider_headroom_bytes=provider_headroom_bytes,
        host_arena_bytes=host_arena_bytes,
        worker_poll_nanoseconds=worker_poll_nanoseconds,
    )
    status = int(library.shadowspill_pytorch_allocator_bootstrap(ctypes.byref(config)))
    if status != 0:
        raise AllocatorInstallError(
            f"ShadowSpill runtime bootstrap failed with status {status}"
        )
    admission = PhysicalAdmission()
    status = int(
        library.shadowspill_pytorch_physical_admission(ctypes.byref(admission))
    )
    if (
        status != 0
        or admission.abi_version != ADAPTER_ABI_VERSION
        or admission.device_budget_bytes != device_budget_bytes
        or admission.provider_headroom_bytes != provider_headroom_bytes
        or admission.slab_bytes == 0
    ):
        raise AllocatorInstallError("physical admission handshake failed")
    physical = PhysicalMemory()
    status = int(library.shadowspill_pytorch_physical_memory(ctypes.byref(physical)))
    if status != 0 or physical.process_bytes > device_budget_bytes:
        raise AllocatorInstallError("bootstrap exceeds the physical device budget")
    cuda.memory.change_current_allocator(allocator)
    _installed = InstalledAllocator(
        library=library,
        allocator=allocator,
        path=path,
        admission=admission,
    )
    return _installed
