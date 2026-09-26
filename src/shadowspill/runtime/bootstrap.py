"""Bootstrapping one runtime in this process, over a frontend's allocator.

A ShadowSpill runtime is installed once per process and never uninstalled,
because the framework allocator it is installed through is process-global and
irreversible. This module owns that sequence -- validate the request, load the
adapter library, bootstrap the runtime, read back what it physically admitted,
and record what the process now holds -- and asks a
:class:`~shadowspill.frontend.RuntimeFrontend` for the four steps only the
framework can take.

Nothing here knows which framework that is.
"""

from __future__ import annotations

import ctypes
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from shadowspill.frontend import RuntimeFrontend
from shadowspill.libraries import (
    LIBRARY_DIRECTORY_ENVIRONMENT,
    library_candidates,
    load_shadowspill_library,
    resolve_library,
    shadowspill_library_path,
)
from shadowspill.status import ABI_VERSION

from .abi import (
    ADAPTER_ABI_VERSION,
    AdapterCapabilities,
    AdapterConfig,
    AdapterFailure,
    AdapterStatistics,
    PhysicalAdmission,
    PhysicalMemory,
    PoolConfig,
    RouteConfig,
    configure_adapter_library,
)
from .failures import format_bytes, wait_allocator_idle


class RuntimeInstallError(RuntimeError):
    """Raised when the process-global framework allocator cannot be installed."""


_MIB = 1 << 20
_PROVIDER_GROWTH_MARGIN = 64 * _MIB
_PROVIDER_RESERVATION_GRANULARITY = 64 * _MIB


@dataclass(frozen=True)
class InstalledRuntime:
    """Process-lifetime owners for the runtime this process bootstrapped."""

    library: Any
    path: Path
    admission: PhysicalAdmission
    #: The neutral runtime this process bound. Callers that only need a
    #: runtime call the neutral library with this directly.
    runtime_handle: int = 0
    fixed_execution_bytes: int = 0
    #: The fixed layout each admitted plan holds in the allocator pool, in
    #: bytes, by plan handle. A plan holds its layout for as long as it is
    #: admitted, so a plan being made beside it finds those bytes taken.
    admitted_layout_bytes: dict[int, int] = field(default_factory=dict)


_installed: InstalledRuntime | None = None


@dataclass(frozen=True, slots=True)
class PoolBootstrap:
    """One pool entry passed to the runtime constructor."""

    pool_id: int
    kind: int
    capacity_bytes: int
    #: What this pool's kind is told about this pool, or ``None``. Held by the
    #: caller for the whole bootstrap call, since only a pointer is passed.
    configuration: Any = None
    #: The shared object supplying this pool's kind, or ``None`` for a built-in
    #: one. Loaded once per distinct path, before the runtime is created.
    library: Path | None = None


@dataclass(frozen=True, slots=True)
class RouteBootstrap:
    """One directed route entry passed to the runtime constructor."""

    route_id: int
    name: str
    source_pool_id: int
    destination_pool_id: int


def installed_runtime() -> InstalledRuntime | None:
    """Return the process-lifetime allocator owner, if already selected."""

    return _installed


def _function_pointer(library: Any, name: str) -> int:
    try:
        symbol = getattr(library, name)
    except AttributeError as exc:
        raise RuntimeInstallError(f"adapter has no {name!r} export") from exc
    pointer = ctypes.cast(symbol, ctypes.c_void_p).value
    if pointer is None:
        raise RuntimeInstallError(f"adapter export {name!r} is null")
    return pointer


#: How far a lane may run ahead with transfers the plan did not schedule (the
#: opening restore, a reconciliation): a plan transfer never waits behind more
#: than this many background bytes. Zero removes the bound.
DEFAULT_BACKGROUND_WINDOW_BYTES: Final = 64 << 20


def install_runtime(
    library_path: str | Path,
    *,
    frontend: RuntimeFrontend,
    device_ordinal: int,
    device_budget_bytes: int,
    provider_headroom_bytes: int,
    allocator_pool_id: int,
    pools: tuple[PoolBootstrap, ...],
    routes: tuple[RouteBootstrap, ...],
    worker_poll_nanoseconds: int = 1_000,
    background_transfer_window_bytes: int = DEFAULT_BACKGROUND_WINDOW_BYTES,
    backend: str | None = None,
) -> InstalledRuntime:
    """Install the process-global allocator before the device is initialized.

    ``backend`` selects the backend shared object the adapter loads: ``None``
    is the one accelerator backend installed beside the libraries, a name
    resolves to ``libshadowspill_backend_<name>.so`` there, and a path is used
    as given.

    ``frontend`` is the framework whose process allocator this runtime is
    installed through; it is asked to refuse an unusable build, to prepare
    and activate its allocator, and to initialize its provider's retained
    workspaces. Public planning computes physical admission before invoking
    this. Installation is intentionally irreversible for the process
    lifetime.
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
        background_transfer_window_bytes,
    )
    path = _validated_adapter_path(library_path)
    backend_library = _backend_path(backend)
    frontend.refuse_unusable_build()
    library = _load_adapter(path)
    missing_operations = tuple(frontend.missing_operations())
    if missing_operations:
        raise RuntimeInstallError(
            "framework adapter is missing canonical storage operations: "
            + ", ".join(missing_operations)
        )
    frontend.prepare_allocator(
        path, _function_pointer(library, "shadowspill_pytorch_backend_record_stream")
    )
    # The configurations are held in a local until bootstrap returns: the
    # adapter is handed pointers into them and reads them during create.
    pool_configurations = tuple(item.configuration for item in pools)
    pool_values = (PoolConfig * len(pools))(
        *(
            PoolConfig(
                pool_id=item.pool_id,
                kind=item.kind,
                capacity_bytes=item.capacity_bytes,
                configuration=(
                    None if configuration is None
                    else ctypes.cast(
                        ctypes.byref(configuration), ctypes.c_void_p
                    )
                ),
            )
            for item, configuration in zip(pools, pool_configurations, strict=True)
        )
    )
    # One entry per distinct library, in the order the pools that need them
    # appear. A kind whose library is missing raises where it is asked for,
    # which names the pool rather than a path.
    library_paths: list[bytes] = []
    for item in pools:
        if item.library is None:
            continue
        encoded = str(item.library).encode("utf-8")
        if encoded not in library_paths:
            library_paths.append(encoded)
    library_values = (ctypes.c_char_p * len(library_paths))(*library_paths)
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
        background_transfer_window_bytes=background_transfer_window_bytes,
        backend_library=str(backend_library).encode("utf-8"),
        libraries=library_values,
        library_count=len(library_paths),
    )
    _bootstrap_allocator(library, config)
    admission = _read_physical_admission(
        library,
        device_budget_bytes=device_budget_bytes,
        provider_headroom_bytes=provider_headroom_bytes,
    )
    _validate_physical_usage(library, device_budget_bytes, provider_headroom_bytes)
    frontend.activate_allocator()
    fixed_execution_bytes = _initialize_provider_state(
        library,
        admission,
        frontend=frontend,
        device_ordinal=device_ordinal,
    )
    _installed = InstalledRuntime(
        library,
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
    frontend: RuntimeFrontend,
    device_ordinal: int,
) -> int:
    """Charge the frontend provider's persistent state before plan admission.

    A provider that creates a retained workspace lazily creates it in the
    middle of a plan, splitting a slab admission had already certified.
    Creating it here, while the slab is otherwise empty, makes its physical
    cost explicit before planning. Dynamic allocation does not assign a special
    address to this state; admission excludes its charged bytes and verifies
    that the remaining capacity is physically usable.
    """

    # Tiny allocator/failure canaries and genuinely small non-BLAS workloads
    # need not reserve a library workspace merely to initialize the runtime.
    # A real matrix task in such a pool will still receive the normal
    # allocator failure if its provider state cannot fit.
    if int(admission.allocator_pool_bytes) < 64 << 20:
        return 0

    frontend.initialize_provider_workspaces(device_ordinal)

    message = wait_allocator_idle(
        library,
        _published_runtime_handle(library),
        problem="provider initialization",
    )
    if message is not None:
        raise RuntimeInstallError(message)
    statistics = AdapterStatistics()
    status = int(
        library.shadowspill_pytorch_allocator_statistics(ctypes.byref(statistics))
    )
    if status != 0:
        raise RuntimeInstallError(
            f"provider allocation accounting failed (status {status})"
        )
    pool = statistics.allocator_pool
    fixed = int(pool.allocated_bytes)
    free = int(pool.free_bytes)
    capacity = int(admission.allocator_pool_bytes)
    largest = int(pool.largest_free_range_bytes)
    if fixed + free != capacity or largest != free:
        raise RuntimeInstallError(
            "provider initialization fragmented the otherwise empty slab: "
            f"fixed={fixed}, free={free}, largest={largest}, capacity={capacity}"
        )
    required = fixed + _PROVIDER_GROWTH_MARGIN
    return (
        (required + _PROVIDER_RESERVATION_GRANULARITY - 1)
        // _PROVIDER_RESERVATION_GRANULARITY
        * _PROVIDER_RESERVATION_GRANULARITY
    )


def validate_dynamic_execution_reservation(
    installed: InstalledRuntime,
    *,
    reserved_bytes: int,
) -> int:
    """Verify persistent allocations fit inside the capacity excluded by planning.

    Dynamic admission consumes every compatible free range in the execution
    pool.  It therefore requires sufficient aggregate unreserved capacity, not
    one contiguous range as large as the complete planning capacity.

    The plans already admitted hold their fixed layouts in the same pool, and
    a plan is checked here before it admits its own, so every layout admitted
    is another plan's and is excluded as well.
    """

    if reserved_bytes < installed.fixed_execution_bytes:
        raise ValueError("fixed execution reservation is smaller than bootstrap")
    message = wait_allocator_idle(
        installed.library,
        installed.runtime_handle,
        problem="fixed execution reservation",
    )
    if message is not None:
        raise RuntimeInstallError(message)
    statistics = AdapterStatistics()
    status = int(
        installed.library.shadowspill_pytorch_allocator_statistics(
            ctypes.byref(statistics)
        )
    )
    if status != 0:
        raise RuntimeInstallError(
            f"fixed execution reservation accounting failed (status {status})"
        )
    pool = statistics.allocator_pool
    allocated = int(pool.allocated_bytes)
    free = int(pool.free_bytes)
    capacity = int(installed.admission.allocator_pool_bytes)
    admitted = sum(installed.admitted_layout_bytes.values())
    if allocated > reserved_bytes + admitted:
        raise RuntimeInstallError(
            "persistent provider allocations exceed the admitted slab reserve: "
            f"observed={allocated}, reserved={reserved_bytes}, "
            f"admitted layouts={admitted}"
        )
    largest = int(pool.largest_free_range_bytes)
    usable_capacity = capacity - reserved_bytes - admitted
    if allocated + free != capacity or free < usable_capacity:
        raise RuntimeInstallError(
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
    background_transfer_window_bytes: int,
) -> None:
    if device_ordinal < 0:
        raise RuntimeInstallError("device ordinal must be non-negative")
    if device_budget_bytes <= 0:
        raise RuntimeInstallError("device budget must be positive")
    if provider_headroom_bytes < 0 or provider_headroom_bytes >= device_budget_bytes:
        raise RuntimeInstallError(
            "provider headroom must be non-negative and smaller than device budget"
        )
    if not pools:
        raise RuntimeInstallError("pool registry must not be empty")
    if allocator_pool_id < 0 or allocator_pool_id >= len(pools):
        raise RuntimeInstallError("allocator pool ID is outside the pool registry")
    if tuple(item.pool_id for item in pools) != tuple(range(len(pools))):
        raise RuntimeInstallError("pool IDs must match their registry positions")
    if any(item.capacity_bytes < 0 for item in pools):
        raise RuntimeInstallError("pool capacities must be non-negative")
    if tuple(item.route_id for item in routes) != tuple(range(len(routes))):
        raise RuntimeInstallError("route IDs must match their registry positions")
    if any(
        item.source_pool_id < 0
        or item.source_pool_id >= len(pools)
        or item.destination_pool_id < 0
        or item.destination_pool_id >= len(pools)
        or item.source_pool_id == item.destination_pool_id
        for item in routes
    ):
        raise RuntimeInstallError("route endpoints must name distinct known pools")
    if worker_poll_nanoseconds < 0:
        raise RuntimeInstallError("worker poll interval must be non-negative")
    if background_transfer_window_bytes < 0:
        raise RuntimeInstallError("background transfer window must be non-negative")
    if _installed is not None:
        raise RuntimeInstallError("ShadowSpill's allocator is already installed")


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
            raise RuntimeInstallError(
                "backend=None needs exactly one accelerator backend beside the"
                f" ShadowSpill libraries; installed: {names}"
            )
        return next(iter(found.values())).resolve()
    if not isinstance(backend, str) or not backend:
        raise RuntimeInstallError(
            "backend must be a backend name, a library path, or None"
        )
    if "/" in backend or backend.endswith(".so"):
        path = Path(backend).expanduser().resolve()
    else:
        resolved = resolve_library(f"libshadowspill_backend_{backend}.so")
        if resolved is None:
            raise RuntimeInstallError(
                f"backend {backend!r} is not installed: no"
                f" libshadowspill_backend_{backend}.so beside the ShadowSpill libraries"
            )
        path = resolved
    if not path.is_file():
        raise RuntimeInstallError(f"backend library does not exist: {path}")
    return path


def _validated_adapter_path(library_path: str | Path) -> Path:
    path = Path(library_path).expanduser().resolve()
    if not path.is_file():
        raise RuntimeInstallError(f"framework adapter does not exist: {path}")
    return path


def _load_adapter(path: Path) -> Any:
    """Load the adapter library and confirm it speaks this ABI."""

    library = ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
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
        raise RuntimeInstallError("framework adapter capability/ABI validation failed")
    _refuse_a_second_runtime(library, path)
    return library


def _refuse_a_second_runtime(adapter: Any, path: Path) -> None:
    """Refuse an adapter linked against a different build than Python loaded.

    The adapter pulls in ``libshadowspill.so`` through its own RPATH, while
    Python resolves one by :func:`resolve_library`. When those two find
    different files the loader maps **both** -- they are different inodes, so
    nothing dedupes them -- and the process then has two runtimes: Python's
    calls land in one, and the pools, routes and lanes the adapter built live
    in the other. Every structure they exchange is a pointer into the wrong
    copy, so it survives exactly as long as the two builds happen to agree
    about a layout, and segfaults in whichever function stopped agreeing.

    The ABI version cannot see it: both copies report the same number, because
    the number changes when the *contract* changes and not when a build does.
    What distinguishes them is the address of a symbol they both export --
    resolved through the adapter's dependency chain against resolved through
    Python's handle. One copy, one address.
    """

    neutral = load_shadowspill_library()
    through_adapter = ctypes.cast(
        adapter.shadowspill_abi_version, ctypes.c_void_p
    ).value
    through_python = ctypes.cast(
        neutral.shadowspill_abi_version, ctypes.c_void_p
    ).value
    if through_adapter == through_python:
        return
    raise RuntimeInstallError(
        "two ShadowSpill runtimes are loaded in this process: the adapter "
        f"{path} is linked against a different libshadowspill.so than Python "
        f"loaded from {shadowspill_library_path()}. Point both at one build -- "
        f"pass an adapter from beside that library, or set "
        f"{LIBRARY_DIRECTORY_ENVIRONMENT} to the adapter's own directory."
    )


def _bootstrap_allocator(library: Any, config: AdapterConfig) -> None:
    status = int(library.shadowspill_pytorch_allocator_bootstrap(ctypes.byref(config)))
    if status == 0:
        return
    raise RuntimeInstallError(
        f"ShadowSpill runtime bootstrap failed with status {status}"
        f"{_bootstrap_refusal(library, config)}"
    )


def _bootstrap_refusal(library: Any, config: AdapterConfig) -> str:
    """What the bootstrap latched about why it refused, if it latched anything.

    A budget refusal latches the bytes the process held against the budget it was
    given. Saying only the status reads as "the device is full", which is the one
    thing it does not mean: the cap is the caller's own, and the device is usually
    far from full when it is exceeded.
    """

    failure = AdapterFailure()
    if int(library.shadowspill_pytorch_allocator_failure(ctypes.byref(failure))) == 0:
        return ""
    held = int(failure.runtime.requested_bytes)
    budget = int(failure.runtime.free_bytes)
    if held <= budget or budget == 0:
        return ""
    return (
        f": the process held {format_bytes(held)} on the device against a declared"
        f" execution budget of {format_bytes(budget)}, so it passed that budget by"
        f" {format_bytes(held - budget)}. The budget has to cover what the process"
        f" already holds, the provider headroom"
        f" ({format_bytes(int(config.provider_headroom_bytes))} here) and the"
        " suballocatable slab, and the slab is sized from the first two -- so a"
        " headroom too small to cover what the process acquires while the pools are"
        " created leaves the slab filling the rest and the total over the cap. This"
        " is the caller's cap, not the device's capacity."
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
        raise RuntimeInstallError("physical admission handshake failed")
    return admission


def _validate_physical_usage(
    library: Any, device_budget_bytes: int, provider_headroom_bytes: int
) -> None:
    """Confirm the bootstrapped process fits the cap it declared.

    A zero provider headroom is the caller asking to be told rather than
    stopped, so the overshoot is reported on stderr and the bootstrap stands.
    The adapter has already latched and printed the same numbers; this repeats
    the decision on the Python side so both gates agree.
    """

    physical = PhysicalMemory()
    status = int(library.shadowspill_pytorch_physical_memory(ctypes.byref(physical)))
    if status != 0:
        raise RuntimeInstallError("bootstrap exceeds the physical device budget")
    if physical.process_bytes <= device_budget_bytes:
        return
    if provider_headroom_bytes != 0:
        raise RuntimeInstallError("bootstrap exceeds the physical device budget")
    excess = int(physical.process_bytes) - device_budget_bytes
    print(
        f"ShadowSpill: the bootstrapped process holds {physical.process_bytes:,} "
        f"bytes against a declared budget of {device_budget_bytes:,}, over by "
        f"{excess:,}. provider_headroom is zero, so this is reported and the "
        f"bootstrap continues.",
        file=sys.stderr,
    )
