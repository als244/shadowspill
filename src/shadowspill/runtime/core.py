"""The runtime object: construction, pools and routes, plan resolution, close."""

from __future__ import annotations

import ctypes
import threading
from collections.abc import Mapping, Sequence
from enum import IntEnum
from pathlib import Path
from types import MappingProxyType

from shadowspill.frontend import RuntimeFrontend
from shadowspill.memory import MemoryPoolConfig
from shadowspill.memory import (
    TransferRoute as TransferRouteConfig,
)
from shadowspill.status import Status

from .abi import (
    MemoryPoolStatistics,
    runtime_library,
)
from .bootstrap import (
    DEFAULT_BACKGROUND_WINDOW_BYTES,
    install_runtime,
)
from .calibration import (
    INITIALIZATION_PROVENANCE,
    RECALIBRATION_PROVENANCE,
    calibrate,
    read_transfer_capabilities,
)
from .configuration import RuntimeConfigurationError, adapter_path, configure_topology
from .failures import (
    RuntimeExecutionError,
    RuntimeFailureDiagnostics,
)
from .objects import ObjectRef, release_object_reference
from .topology import (
    MemoryPool,
    RuntimeRoute,
    TransferCapabilities,
)

_RUNTIME_INVALID_STATE = Status.INVALID_STATE


_runtime_lock = threading.Lock()
_active_runtime: Runtime | None = None


class PlanState(IntEnum):
    """What became of one plan id, mirroring `ShadowSpillPlanState`.

    An id stays answerable after its plan is gone. A closing plan releases the
    ranges its own scopes made, so a live allocation naming a closed or destroyed
    plan is a defect rather than an expected state, and this is what makes it
    visible. Allocations taken outside any plan's scope carry no plan at all.
    """

    #: No plan was ever created with this id.
    UNKNOWN = 0
    #: Created and open: it may still admit work.
    LIVE = 1
    #: Closed, record still present: it admits no further work.
    CLOSED = 2
    #: Closed and the record freed. The id stays claimed, because a lease may
    #: still carry it.
    DESTROYED = 3


class Runtime:
    """Own ShadowSpill's process-lifetime allocator, pools, routes, and worker.

    Accelerator allocator selection is process-global and irreversible after
    the framework initializes the accelerator. Construct exactly one ``Runtime``
    before any device allocation the framework makes, then pass it to every
    ``plan_step`` or ``plan_forward`` call.

    The current backend supports one device pool plus any number of pinned-host
    pools. Pools and directed routes have explicit identities; each admitted
    callable independently selects its execution/spill pool pair and matching
    routes.
    """

    def __init__(
        self,
        *,
        frontend: RuntimeFrontend,
        pools: Mapping[str, MemoryPoolConfig],
        routes: Mapping[str, TransferRouteConfig],
        library_path: str | Path | None = None,
        calibrate: bool = True,
        worker_poll_nanoseconds: int = 1_000,
        background_transfer_window_bytes: int = DEFAULT_BACKGROUND_WINDOW_BYTES,
        backend: str | None = None,
    ) -> None:
        topology = configure_topology(pools, routes)
        path = adapter_path(library_path)
        global _active_runtime
        with _runtime_lock:
            if _active_runtime is not None:
                raise RuntimeConfigurationError(
                    "a ShadowSpill Runtime is already initialized in this process"
                )
            installed = install_runtime(
                path,
                frontend=frontend,
                device_ordinal=topology.device.device,
                device_budget_bytes=topology.device.physical_capacity,
                provider_headroom_bytes=topology.device.provider_headroom,
                allocator_pool_id=topology.allocator_pool_id,
                pools=topology.pool_bootstrap,
                routes=topology.route_bootstrap,
                worker_poll_nanoseconds=worker_poll_nanoseconds,
                background_transfer_window_bytes=background_transfer_window_bytes,
                backend=backend,
            )
            self._frontend = frontend
            self._installed = installed
            # The neutral runtime this process bound. Holding it here lets the
            # calls that need nothing else go straight to the neutral library.
            # It is dropped on close, so a call after close fails on the
            # closed guard rather than on a stale pointer.
            self._runtime_handle: int = installed.runtime_handle
            self._lock = threading.RLock()
            self._closed = False
            self._unusable_reason: str | None = None
            self._last_failure: RuntimeFailureDiagnostics | None = None
            self._active_plan_handles: set[int] = set()
            self._planning_plan_handle: int | None = None
            self._active_object_references = 0
            self._persistent_state_count = 0
            self._next_persistent_object_id = 1 << 62
            self._pools = topology.pools(int(installed.admission.allocator_pool_bytes))
            self._routes = topology.routes()
            self._route_by_pair = MappingProxyType(
                {
                    (route.source, route.destination): route
                    for route in self._routes.values()
                }
            )
            self._pool_names = topology.pool_names
            self._route_names = topology.route_names
            _active_runtime = self
        if calibrate:
            self._calibrate(routes=None, provenance=INITIALIZATION_PROVENANCE)

    @property
    def frontend(self) -> RuntimeFrontend:
        """The framework this runtime was opened with.

        Every framework call the runtime makes goes through this, and through
        nothing else: device selection and synchronization, the process
        allocator, and finding or detaching the objects that hold a lease.
        """

        return self._frontend

    @property
    def pools(self) -> Mapping[str, MemoryPool]:
        """Read-only initialized pool registry keyed by user names."""

        return self._pools

    def pool_statistics(self, pool: str = "execution") -> MemoryPoolStatistics:
        """What one pool holds, asked of that pool by name.

        A runtime may own any number of pools and a pool carries no role of its
        own, so its numbers are read per pool. The allocator's own pool also
        arrives with the adapter's statistics, which is the one a caller on the
        allocation path usually wants.
        """

        registered = self._pools.get(pool)
        if registered is None:
            raise RuntimeConfigurationError(f"unknown pool {pool!r}")
        statistics = MemoryPoolStatistics()
        status = int(
            runtime_library().shadowspill_memory_pool_statistics(
                self._runtime_handle,
                ctypes.c_uint32(registered.pool_id),
                ctypes.byref(statistics),
            )
        )
        if status != 0:
            raise RuntimeExecutionError(
                f"failed to read statistics for pool {pool!r}: status={status}"
            )
        return statistics

    def next_plan_id(self) -> int:
        """Take an id no other plan on this runtime will be given.

        Taken before the plan is created, so the same id can name the allocation
        scopes opened for it -- profiling runs under the plan but outside any of
        its tasks, so the runtime cannot infer the plan there.
        """

        plan_id = ctypes.c_uint64()
        status = int(
            runtime_library().shadowspill_runtime_next_plan_id(
                self._runtime_handle, ctypes.byref(plan_id)
            )
        )
        if status != 0 or plan_id.value == 0:
            raise RuntimeExecutionError(f"failed to take a plan id: status={status}")
        return int(plan_id.value)

    def plan_state(self, plan_id: int) -> PlanState:
        """What became of one plan id, after the plan itself may be gone."""

        state = ctypes.c_uint32()
        status = int(
            runtime_library().shadowspill_runtime_plan_state(
                self._runtime_handle, ctypes.c_uint64(plan_id), ctypes.byref(state)
            )
        )
        if status != 0:
            raise RuntimeExecutionError(
                f"failed to read the state of plan {plan_id}: status={status}"
            )
        return PlanState(int(state.value))

    @property
    def routes(self) -> Mapping[str, RuntimeRoute]:
        """Read-only directed-route registry keyed by user names."""

        return self._routes

    @property
    def transfer_capabilities(self) -> TransferCapabilities:
        """Return a lock-consistent immutable transfer-matrix snapshot."""

        with self._lock:
            self._require_open()
            return read_transfer_capabilities(self._runtime_handle, self._pool_names)

    @property
    def last_failure(self) -> RuntimeFailureDiagnostics | None:
        """Return the latest structured frontend failure, if any."""

        with self._lock:
            return self._last_failure

    def calibrate_transfer_capabilities(
        self,
        *,
        routes: Sequence[tuple[str, str]] | None = None,
        small_copy_bytes: int = 4096,
        large_copy_bytes: int = 256 << 20,
        warmup_copies: int = 4,
        measured_copies: int = 16,
    ) -> TransferCapabilities:
        """Measure all or selected routes and atomically publish a new matrix.

        This runtime must be locally idle. ShadowSpill deliberately performs no
        cross-process barrier: callers may coordinate independent runtimes and
        invoke this method concurrently to measure contended link behavior.
        """

        return self._calibrate(
            routes=routes,
            provenance=RECALIBRATION_PROVENANCE,
            small_copy_bytes=small_copy_bytes,
            large_copy_bytes=large_copy_bytes,
            warmup_copies=warmup_copies,
            measured_copies=measured_copies,
        )

    def close(self) -> None:
        """Close every runtime resource after verifying external ownership.

        the framework's selected allocator shim remains installed because allocator
        selection is process-global. Its runtime is permanently closed: future
        nonzero device allocations raise a typed closed-runtime error.
        """

        with self._lock:
            if self._closed:
                return
            if self._active_plan_handles or self._planning_plan_handle is not None:
                raise RuntimeConfigurationError(
                    "cannot close Runtime while a callable or in-progress plan owns it"
                )
            if self._persistent_state_count != 0:
                raise RuntimeConfigurationError(
                    "cannot close Runtime while persistent frontend state remains; "
                    "export it with release_runtime=True first"
                )
            if self._active_object_references != 0:
                raise RuntimeConfigurationError(
                    "cannot close Runtime while public object references remain; "
                    "release every retained object reference first"
                )
            status = int(self._installed.library.shadowspill_pytorch_allocator_close())
            if status == _RUNTIME_INVALID_STATE:
                raise RuntimeConfigurationError(
                    "cannot close Runtime while caller-owned device outputs "
                    "still reference its memory pools; release those references first"
                )
            self._closed = True
            if status != 0:
                raise RuntimeConfigurationError(
                    "runtime close released its resources after observing "
                    f"status {status}"
                )

    def _release_object_reference(self, reference: ObjectRef) -> None:
        """Release exactly one public runtime-object owner: the `ObjectRef`
        protocol's half of `objects.acquire_object_reference`."""

        release_object_reference(self, reference)

    def __enter__(self) -> Runtime:
        self._require_open()
        return self

    def __exit__(self, *exception: object) -> None:
        del exception
        self.close()

    def _calibrate(
        self,
        *,
        routes: Sequence[tuple[str, str]] | None,
        provenance: int,
        small_copy_bytes: int = 4096,
        large_copy_bytes: int = 256 << 20,
        warmup_copies: int = 4,
        measured_copies: int = 16,
    ) -> TransferCapabilities:
        with self._lock:
            self._require_open()
            if self._active_plan_handles or self._planning_plan_handle is not None:
                raise RuntimeConfigurationError(
                    "transfer calibration requires no callable or in-progress plan"
                )
            calibrate(
                self._runtime_handle,
                self._pool_names,
                routes=routes,
                provenance=provenance,
                small_copy_bytes=small_copy_bytes,
                large_copy_bytes=large_copy_bytes,
                warmup_copies=warmup_copies,
                measured_copies=measured_copies,
            )
            return read_transfer_capabilities(self._runtime_handle, self._pool_names)

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeConfigurationError("ShadowSpill Runtime is closed")
        if self._unusable_reason is not None:
            raise RuntimeConfigurationError(
                f"ShadowSpill Runtime is unusable: {self._unusable_reason}"
            )


__all__ = ["PlanState", "Runtime"]
