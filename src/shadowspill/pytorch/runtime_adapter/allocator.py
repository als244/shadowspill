"""Installation of ShadowSpill through PyTorch's supported allocator API."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from shadowspill.libraries import library_candidates, resolve_library
from shadowspill.pytorch.accelerator import accelerator_device
from shadowspill.pytorch.runtime_adapter.abi import (
    ADAPTER_ABI_VERSION,
    AdapterCapabilities,
    AdapterConfig,
    AdapterStatistics,
    PhysicalAdmission,
    PhysicalMemory,
    PoolConfig,
    RouteConfig,
    configure_adapter_library,
)
from shadowspill.pytorch.runtime_adapter.failures import wait_allocator_idle
from shadowspill.status import ABI_VERSION

_REQUIRED_STORAGE_OPERATIONS = (
    "_import_cpu_storages",
    "_export_cpu_storages",
    "_make_runtime_cpu_storage",
    "_acquire_storages",
    "_before_task_storages",
    "_dematerialize_storages",
    "_after_task_storages",
    "_transfer_acquired_storage_to_caller",
)


class AllocatorInstallError(RuntimeError):
    """Raised when the process-global PyTorch allocator cannot be installed."""


_MIB = 1 << 20
_PROVIDER_GROWTH_MARGIN = 64 * _MIB
_PROVIDER_RESERVATION_GRANULARITY = 64 * _MIB


@dataclass(frozen=True)
class InstalledAllocator:
    """Process-lifetime owners for PyTorch's selected allocator callbacks."""

    library: Any
    allocator: Any
    path: Path
    admission: PhysicalAdmission
    #: The neutral runtime this process bound. Callers that only need a
    #: runtime call the neutral library with this, rather than going through
    #: an entry point here that would fetch the same pointer and forward.
    runtime_handle: int = 0
    fixed_execution_bytes: int = 0


_installed: InstalledAllocator | None = None


@dataclass(frozen=True, slots=True)
class PoolBootstrap:
    """One pool entry passed to the runtime constructor."""

    pool_id: int
    kind: int
    capacity_bytes: int


@dataclass(frozen=True, slots=True)
class RouteBootstrap:
    """One directed route entry passed to the runtime constructor."""

    route_id: int
    name: str
    source_pool_id: int
    destination_pool_id: int


def installed_allocator() -> InstalledAllocator | None:
    """Return the process-lifetime allocator owner, if already selected."""

    return _installed


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
    allocator_pool_id: int,
    pools: tuple[PoolBootstrap, ...],
    routes: tuple[RouteBootstrap, ...],
    worker_poll_nanoseconds: int = 1_000,
    backend: str | None = None,
) -> InstalledAllocator:
    """Install the process-global allocator before PyTorch initializes the accelerator.

    ``backend`` selects the backend shared object the adapter loads: ``None``
    is the one accelerator backend installed beside the libraries, a name
    resolves to ``libshadowspill_backend_<name>.so`` there, and a path is used
    as given.

    This is an internal frontend primitive. Public planning computes physical
    admission before invoking it. Installation is intentionally irreversible
    for the process lifetime.
    """

    global _installed
    _validate_install_request(
        device_ordinal,
        device_budget_bytes,
        provider_headroom_bytes,
        allocator_pool_id,
        pools,
        routes,
        worker_poll_nanoseconds,
    )
    path = _adapter_path(library_path)
    backend_library = _backend_path(backend)
    frontend = _accelerator_frontend()
    library = _load_adapter(path)
    allocator = _create_allocator(frontend, path)
    _configure_record_stream(library, allocator)
    pool_values = (PoolConfig * len(pools))(
        *(
            PoolConfig(
                pool_id=item.pool_id,
                kind=item.kind,
                capacity_bytes=item.capacity_bytes,
            )
            for item in pools
        )
    )
    route_names = tuple(item.name.encode("utf-8") for item in routes)
    route_values = (RouteConfig * len(routes))(
        *(
            RouteConfig(
                route_id=item.route_id,
                source_pool_id=item.source_pool_id,
                destination_pool_id=item.destination_pool_id,
                name=name,
            )
            for item, name in zip(routes, route_names, strict=True)
        )
    )
    config = AdapterConfig(
        abi_version=ADAPTER_ABI_VERSION,
        device_ordinal=device_ordinal,
        device_budget_bytes=device_budget_bytes,
        provider_headroom_bytes=provider_headroom_bytes,
        allocator_pool_id=allocator_pool_id,
        pools=pool_values,
        pool_count=len(pools),
        routes=route_values,
        route_count=len(routes),
        worker_poll_nanoseconds=worker_poll_nanoseconds,
        backend_library=str(backend_library).encode("utf-8"),
    )
    _bootstrap_allocator(library, config)
    admission = _read_physical_admission(
        library,
        device_budget_bytes=device_budget_bytes,
        provider_headroom_bytes=provider_headroom_bytes,
    )
    _validate_physical_usage(library, device_budget_bytes)
    frontend.memory.change_current_allocator(allocator)
    fixed_execution_bytes = _initialize_provider_state(
        library,
        admission,
        device_ordinal=device_ordinal,
    )
    _installed = InstalledAllocator(
        library,
        allocator,
        path,
        admission,
        _published_runtime_handle(library),
        fixed_execution_bytes,
    )
    return _installed


def _published_runtime_handle(library: Any) -> int:
    """Read the neutral runtime the bootstrap just published."""

    handle = ctypes.c_size_t(0)
    status = int(library.shadowspill_pytorch_runtime_handle(ctypes.byref(handle)))
    if status != 0:
        raise RuntimeError(
            f"the runtime was not published after bootstrap (status {status})"
        )
    return int(handle.value)


def _initialize_provider_state(
    library: Any,
    admission: PhysicalAdmission,
    *,
    device_ordinal: int,
) -> int:
    """Create persistent PyTorch CUDA provider state before plan admission.

    PyTorch lazily obtains its cuBLAS handle on the first matrix operation.
    That handle's workspace is allocated through the selected allocator and
    remains live. Creating it while the slab is otherwise empty makes its
    physical cost explicit before planning. Dynamic allocation does not assign
    a special address to this state; admission excludes its charged bytes and
    verifies that the remaining capacity is physically usable.
    """

    # Tiny allocator/failure canaries and genuinely small non-BLAS workloads
    # need not reserve a 32 MiB library workspace merely to initialize the
    # runtime. A real matrix task in such a pool will still receive the normal
    # allocator failure if its provider state cannot fit.
    if int(admission.allocator_pool_bytes) < 64 << 20:
        return 0

    get_handle = getattr(torch._C, "_cuda_getCurrentBlasHandle", None)
    if get_handle is None or not callable(get_handle):
        raise AllocatorInstallError(
            "this PyTorch build lacks the required CUDA provider initializer"
        )
    torch.cuda.set_device(device_ordinal)
    get_handle()

    # Merely obtaining the handle does not force cuBLAS to create its retained
    # workspace.  If its first GEMM runs later while a large profiling input is
    # live, that small persistent allocation can split the otherwise empty
    # slab and prevent a large fixed-layout arena from being reserved despite
    # ample aggregate capacity.  Exercise the provider now, while the pool is
    # empty, so its retained state has deterministic low-address placement.
    device = accelerator_device(device_ordinal)
    shape = (2048, 2048)
    left = torch.empty(shape, dtype=torch.bfloat16, device=device)
    right = torch.empty(shape, dtype=torch.bfloat16, device=device)
    output = torch.empty(shape, dtype=torch.bfloat16, device=device)
    try:
        torch.mm(left, right, out=output)
        torch.cuda.current_stream(device_ordinal).synchronize()
    finally:
        del output
        del right
        del left
        torch.cuda.current_stream(device_ordinal).synchronize()

    message = wait_allocator_idle(
        library,
        _published_runtime_handle(library),
        problem="provider initialization",
    )
    if message is not None:
        raise AllocatorInstallError(message)
    statistics = AdapterStatistics()
    status = int(
        library.shadowspill_pytorch_allocator_statistics(ctypes.byref(statistics))
    )
    if status != 0:
        raise AllocatorInstallError(
            f"CUDA provider allocation accounting failed (status {status})"
        )
    runtime = statistics.runtime
    fixed = int(runtime.allocated_bytes)
    free = int(runtime.free_bytes)
    capacity = int(admission.allocator_pool_bytes)
    largest = int(runtime.largest_free_range_bytes)
    if fixed + free != capacity or largest != free:
        raise AllocatorInstallError(
            "CUDA provider initialization fragmented the otherwise empty slab: "
            f"fixed={fixed}, free={free}, largest={largest}, capacity={capacity}"
        )
    required = fixed + _PROVIDER_GROWTH_MARGIN
    return (
        (required + _PROVIDER_RESERVATION_GRANULARITY - 1)
        // _PROVIDER_RESERVATION_GRANULARITY
        * _PROVIDER_RESERVATION_GRANULARITY
    )


def validate_dynamic_execution_reservation(
    installed: InstalledAllocator,
    *,
    reserved_bytes: int,
) -> int:
    """Verify persistent allocations fit inside the capacity excluded by planning.

    Dynamic admission consumes every compatible free range in the execution
    pool.  It therefore requires sufficient aggregate unreserved capacity, not
    one contiguous range as large as the complete planning capacity.
    """

    if reserved_bytes < installed.fixed_execution_bytes:
        raise ValueError("fixed execution reservation is smaller than bootstrap")
    message = wait_allocator_idle(
        installed.library,
        installed.runtime_handle,
        problem="fixed execution reservation",
    )
    if message is not None:
        raise AllocatorInstallError(message)
    statistics = AdapterStatistics()
    status = int(
        installed.library.shadowspill_pytorch_allocator_statistics(
            ctypes.byref(statistics)
        )
    )
    if status != 0:
        raise AllocatorInstallError(
            f"fixed execution reservation accounting failed (status {status})"
        )
    runtime = statistics.runtime
    allocated = int(runtime.allocated_bytes)
    free = int(runtime.free_bytes)
    capacity = int(installed.admission.allocator_pool_bytes)
    if allocated > reserved_bytes:
        raise AllocatorInstallError(
            "persistent provider allocations exceed the admitted slab reserve: "
            f"observed={allocated}, reserved={reserved_bytes}"
        )
    largest = int(runtime.largest_free_range_bytes)
    usable_capacity = capacity - reserved_bytes
    if allocated + free != capacity or free < usable_capacity:
        raise AllocatorInstallError(
            "live execution allocation accounting is incompatible with the "
            "admitted dynamic capacity: "
            f"observed={allocated}, reserved={reserved_bytes}, free={free}, "
            f"required_free={usable_capacity}, largest={largest}, capacity={capacity}"
        )
    return allocated


def _validate_install_request(
    device_ordinal: int,
    device_budget_bytes: int,
    provider_headroom_bytes: int,
    allocator_pool_id: int,
    pools: tuple[PoolBootstrap, ...],
    routes: tuple[RouteBootstrap, ...],
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
    if not pools:
        raise AllocatorInstallError("pool registry must not be empty")
    if allocator_pool_id < 0 or allocator_pool_id >= len(pools):
        raise AllocatorInstallError("allocator pool ID is outside the pool registry")
    if tuple(item.pool_id for item in pools) != tuple(range(len(pools))):
        raise AllocatorInstallError("pool IDs must match their registry positions")
    if any(item.capacity_bytes < 0 for item in pools):
        raise AllocatorInstallError("pool capacities must be non-negative")
    if tuple(item.route_id for item in routes) != tuple(range(len(routes))):
        raise AllocatorInstallError("route IDs must match their registry positions")
    if any(
        item.source_pool_id < 0
        or item.source_pool_id >= len(pools)
        or item.destination_pool_id < 0
        or item.destination_pool_id >= len(pools)
        or item.source_pool_id == item.destination_pool_id
        for item in routes
    ):
        raise AllocatorInstallError("route endpoints must name distinct known pools")
    if worker_poll_nanoseconds < 0:
        raise AllocatorInstallError("worker poll interval must be non-negative")
    if _installed is not None:
        raise AllocatorInstallError("ShadowSpill's allocator is already installed")


def _backend_path(backend: str | None) -> Path:
    """Resolve the backend shared object the adapter will load.

    ``None`` selects the one accelerator backend installed beside the
    ShadowSpill libraries; a name selects ``libshadowspill_backend_<name>.so``
    there; a path is used as given.
    """

    if backend is None:
        found = {
            candidate.name.removeprefix("libshadowspill_backend_").removesuffix(
                ".so"
            ): candidate
            for directory in {
                item.parent for item in library_candidates("libshadowspill.so")
            }
            if directory.is_dir()
            for candidate in sorted(directory.glob("libshadowspill_backend_*.so"))
        }
        found.pop("mock", None)
        if len(found) != 1:
            names = ", ".join(sorted(found)) or "none"
            raise AllocatorInstallError(
                "backend=None needs exactly one accelerator backend beside the"
                f" ShadowSpill libraries; installed: {names}"
            )
        return next(iter(found.values())).resolve()
    if not isinstance(backend, str) or not backend:
        raise AllocatorInstallError(
            "backend must be a backend name, a library path, or None"
        )
    if "/" in backend or backend.endswith(".so"):
        path = Path(backend).expanduser().resolve()
    else:
        resolved = resolve_library(f"libshadowspill_backend_{backend}.so")
        if resolved is None:
            raise AllocatorInstallError(
                f"backend {backend!r} is not installed: no"
                f" libshadowspill_backend_{backend}.so beside the ShadowSpill libraries"
            )
        path = resolved
    if not path.is_file():
        raise AllocatorInstallError(f"backend library does not exist: {path}")
    return path


def _adapter_path(library_path: str | Path) -> Path:
    path = Path(library_path).expanduser().resolve()
    if not path.is_file():
        raise AllocatorInstallError(f"PyTorch adapter does not exist: {path}")
    return path


def _accelerator_frontend() -> Any:
    if torch.version.cuda is None:
        raise AllocatorInstallError("a CUDA-enabled PyTorch build is required")
    frontend: Any = torch.cuda
    if frontend.is_initialized():
        raise AllocatorInstallError(
            "PyTorch CUDA was initialized before ShadowSpill allocator installation"
        )
    return frontend


def _load_adapter(path: Path) -> Any:
    library = ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
    missing_operations = [
        name
        for name in _REQUIRED_STORAGE_OPERATIONS
        if not hasattr(torch.ops.shadowspill, name)
    ]
    if missing_operations:
        raise AllocatorInstallError(
            "PyTorch adapter is missing canonical storage operations: "
            + ", ".join(missing_operations)
        )
    configure_adapter_library(library)
    capabilities = AdapterCapabilities()
    status = int(
        library.shadowspill_pytorch_adapter_capabilities(ctypes.byref(capabilities))
    )
    if (
        status != 0
        or capabilities.abi_version != ADAPTER_ABI_VERSION
        or capabilities.runtime_abi_version != ABI_VERSION
    ):
        raise AllocatorInstallError("PyTorch adapter capability/ABI validation failed")
    return library


def _create_allocator(frontend: Any, path: Path) -> Any:
    return frontend.memory.CUDAPluggableAllocator(
        str(path),
        "shadowspill_pytorch_backend_malloc",
        "shadowspill_pytorch_backend_free",
    )


def _configure_record_stream(library: Any, allocator: Any) -> None:
    record_stream_pointer = _function_pointer(
        library, "shadowspill_pytorch_backend_record_stream"
    )
    torch_allocator = allocator.allocator()
    set_record_stream = getattr(torch_allocator, "set_record_stream_fn", None)
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
        or admission.allocator_pool_bytes == 0
    ):
        raise AllocatorInstallError("physical admission handshake failed")
    return admission


def _validate_physical_usage(library: Any, device_budget_bytes: int) -> None:
    physical = PhysicalMemory()
    status = int(library.shadowspill_pytorch_physical_memory(ctypes.byref(physical)))
    if status != 0 or physical.process_bytes > device_budget_bytes:
        raise AllocatorInstallError("bootstrap exceeds the physical device budget")
