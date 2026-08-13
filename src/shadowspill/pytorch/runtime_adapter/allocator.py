"""Installation of ShadowSpill through PyTorch's supported allocator API."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from shadowspill.pytorch.runtime_adapter.abi import (
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


def resize_spill_pool(
    installed: InstalledAllocator, *, spill_pool_bytes: int, spill_budget_bytes: int
) -> None:
    """Grow the pinned spill pool during planning without exceeding its cap."""

    current = int(installed.admission.spill_pool_bytes)
    if spill_pool_bytes < current:
        raise AllocatorInstallError("spill pool cannot shrink")
    if spill_pool_bytes == current:
        return
    if spill_pool_bytes > spill_budget_bytes or current + spill_pool_bytes > (
        spill_budget_bytes
    ):
        raise AllocatorInstallError(
            "planning-time spill pool replacement exceeds spill_budget"
        )
    status = int(
        installed.library.shadowspill_pytorch_resize_spill_pool(spill_pool_bytes)
    )
    if status != 0:
        raise AllocatorInstallError(f"spill pool growth failed with status {status}")
    admission = PhysicalAdmission()
    status = int(
        installed.library.shadowspill_pytorch_physical_admission(
            ctypes.byref(admission)
        )
    )
    if status != 0 or int(admission.spill_pool_bytes) != spill_pool_bytes:
        raise AllocatorInstallError("spill pool admission was not updated")
    installed.admission.spill_pool_bytes = admission.spill_pool_bytes


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
    spill_pool_bytes: int,
    worker_poll_nanoseconds: int = 1_000,
) -> InstalledAllocator:
    """Install the process-global CUDA allocator before PyTorch CUDA init.

    This is an internal frontend primitive. Public planning computes physical
    admission before invoking it. Installation is intentionally irreversible
    for the process lifetime.
    """

    global _installed
    _validate_install_request(
        device_ordinal,
        device_budget_bytes,
        provider_headroom_bytes,
        spill_pool_bytes,
        worker_poll_nanoseconds,
    )
    path = _adapter_path(library_path)
    cuda = _available_cuda_frontend()
    library = _load_adapter(path)
    allocator = _create_allocator(cuda, path)
    _configure_record_stream(library, allocator)
    config = AdapterConfig(
        abi_version=ADAPTER_ABI_VERSION,
        device_ordinal=device_ordinal,
        device_budget_bytes=device_budget_bytes,
        provider_headroom_bytes=provider_headroom_bytes,
        spill_pool_bytes=spill_pool_bytes,
        worker_poll_nanoseconds=worker_poll_nanoseconds,
    )
    _bootstrap_allocator(library, config)
    admission = _read_physical_admission(
        library,
        device_budget_bytes=device_budget_bytes,
        provider_headroom_bytes=provider_headroom_bytes,
    )
    _validate_physical_usage(library, device_budget_bytes)
    cuda.memory.change_current_allocator(allocator)
    _installed = InstalledAllocator(library, allocator, path, admission)
    return _installed


def _validate_install_request(
    device_ordinal: int,
    device_budget_bytes: int,
    provider_headroom_bytes: int,
    spill_pool_bytes: int,
    worker_poll_nanoseconds: int,
) -> None:
    if device_ordinal < 0:
        raise AllocatorInstallError("device ordinal must be non-negative")
    if device_budget_bytes <= 0:
        raise AllocatorInstallError("device budget must be positive")
    if provider_headroom_bytes < 0 or provider_headroom_bytes >= device_budget_bytes:
        raise AllocatorInstallError(
            "provider headroom must be non-negative and smaller than device budget"
        )
    if spill_pool_bytes < 0:
        raise AllocatorInstallError("host arena bytes must be non-negative")
    if worker_poll_nanoseconds < 0:
        raise AllocatorInstallError("worker poll interval must be non-negative")
    if _installed is not None:
        raise AllocatorInstallError("ShadowSpill's allocator is already installed")


def _adapter_path(library_path: str | Path) -> Path:
    path = Path(library_path).expanduser().resolve()
    if not path.is_file():
        raise AllocatorInstallError(f"PyTorch adapter does not exist: {path}")
    return path


def _available_cuda_frontend() -> Any:
    if torch.version.cuda is None:
        raise AllocatorInstallError("a CUDA-enabled PyTorch build is required")
    cuda: Any = torch.cuda
    if cuda.is_initialized():
        raise AllocatorInstallError(
            "PyTorch CUDA was initialized before ShadowSpill allocator installation"
        )
    return cuda


def _load_adapter(path: Path) -> Any:
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
    return library


def _create_allocator(cuda: Any, path: Path) -> Any:
    return cuda.memory.CUDAPluggableAllocator(
        str(path),
        "shadowspill_pytorch_cuda_malloc",
        "shadowspill_pytorch_cuda_free",
    )


def _configure_record_stream(library: Any, allocator: Any) -> None:
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


def _bootstrap_allocator(library: Any, config: AdapterConfig) -> None:
    status = int(library.shadowspill_pytorch_allocator_bootstrap(ctypes.byref(config)))
    if status != 0:
        raise AllocatorInstallError(
            f"ShadowSpill runtime bootstrap failed with status {status}"
        )


def _read_physical_admission(
    library: Any,
    *,
    device_budget_bytes: int,
    provider_headroom_bytes: int,
) -> PhysicalAdmission:
    admission = PhysicalAdmission()
    status = int(
        library.shadowspill_pytorch_physical_admission(ctypes.byref(admission))
    )
    if (
        status != 0
        or admission.abi_version != ADAPTER_ABI_VERSION
        or admission.device_budget_bytes != device_budget_bytes
        or admission.provider_headroom_bytes != provider_headroom_bytes
        or admission.execution_pool_bytes == 0
    ):
        raise AllocatorInstallError("physical admission handshake failed")
    return admission


def _validate_physical_usage(library: Any, device_budget_bytes: int) -> None:
    physical = PhysicalMemory()
    status = int(library.shadowspill_pytorch_physical_memory(ctypes.byref(physical)))
    if status != 0 or physical.process_bytes > device_budget_bytes:
        raise AllocatorInstallError("bootstrap exceeds the physical device budget")
