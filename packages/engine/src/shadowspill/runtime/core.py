"""The runtime object: construction, pools and routes, plan resolution, close."""

from __future__ import annotations

import ctypes
import os
import sys
import threading
import time
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
from .failures import RuntimeExecutionError
from .objects import ObjectRef, release_object_reference, reserve_pool_range
from .topology import (
    MemoryPool,
    RuntimeRoute,
    TransferCapabilities,
)


def _announce(message: str) -> None:
    """Opt-in progress for pool registration and optional route calibration."""

    if os.environ.get("SHADOWSPILL_RUNTIME_PROGRESS"):
        print(f"shadowspill runtime: {message}", file=sys.stderr, flush=True)


_RUNTIME_INVALID_STATE = Status.INVALID_STATE


_runtime_lock = threading.Lock()
_active_runtime: Runtime | None = None


class PlanState(IntEnum):
    """What became of one plan id, as `Runtime.plan_state` reports it.

    Mirrors ``ShadowSpillPlanState``. A closing plan releases the ranges its
    own scopes made, so a live allocation naming a closed or destroyed plan is
    a defect. Allocations made outside any plan's scope name no plan.

    Attributes
    ----------
    UNKNOWN : int
        No plan was ever created with this id.
    LIVE : int
        Created and open: the plan may still admit work.
    CLOSED : int
        Closed, with its record still present: it admits no further work.
    DESTROYED : int
        Closed and its record freed. The id stays claimed, because a lease may
        still carry it.
    """

    UNKNOWN = 0
    LIVE = 1
    CLOSED = 2
    DESTROYED = 3


class Runtime:
    """The process's runtime: its device allocator, pools, routes and worker.

    A process opens one runtime, before the framework makes any device
    allocation. Opening it installs the process's device allocator through
    ``frontend``, which cannot be undone, opens the pools and routes, and
    measures the routes unless told not to; the transfer worker then runs
    until the runtime closes. `Executor` opens one in a fresh worker and
    shares it between its Plans, each of which names its execution and spill
    pools and the routes between them.

    By default, opening finds the device's NUMA node, narrows the CPU masks of
    the process's existing threads to that node's permitted CPUs, and prefers
    the node's pages before pinning spill pools; where a step fails it warns
    and keeps the operating system's placement. The CPU masks and the opening
    thread's memory policy stay in effect for the process, and existing
    allocations are not moved. As a context manager, it closes on exit.

    Parameters
    ----------
    frontend : RuntimeFrontend
        The framework integration the runtime works through: it selects and
        synchronizes the device, provides the process allocator, and finds or
        detaches the objects that hold a lease.
    pools : mapping of str to DevicePool or SpillPool
        The pools, by name. Names are identifiers; exactly one pool is a
        device pool, and at least one other is a spill pool. A pool's id is
        its position in the mapping.
    routes : mapping of str to TransferRoute
        The directed routes bytes may move along, by name; at least one. Each
        joins the device pool and a spill pool, in either direction, and no
        two join the same pools in the same direction. A route's id is its
        position in the mapping.
    library_path : str or path-like, optional
        The framework adapter library to load. By default, the one installed
        beside the package.
    calibrate : bool, default True
        Whether to measure every route before returning. Without it, the
        routes carry no measurements until `calibrate_transfer_capabilities`
        runs.
    numa_binding : bool, default True
        Whether to place the process on the device's NUMA node, as described
        above. False leaves the caller's placement alone.
    worker_poll_nanoseconds : int, default 1000
        How long the transfer worker waits before asking again whether a
        pending completion has finished, in nanoseconds. At least zero.
    background_transfer_window_bytes : int, default DEFAULT_BACKGROUND_WINDOW_BYTES
        Bytes of transfers the plan did not schedule, such as materializing
        or reconciling state, that a route may have in flight at once (64 MiB
        by default): a scheduled transfer never waits behind more. Zero
        removes the bound. At least zero.
    backend : str, optional
        The backend library the adapter loads. By default, the one
        accelerator backend installed beside the libraries; a name selects
        ``libshadowspill_backend_<name>.so`` there, and a path (one that
        contains ``/`` or ends in ``.so``) is used as given.

    Raises
    ------
    RuntimeConfigurationError
        If the pools or routes are invalid, the adapter library is not found, a
        runtime has already been opened in this process (even one since
        closed), a pool's image cannot be reserved, or calibration fails.
    TypeError
        If ``pools`` or ``routes`` is not a mapping, or holds something other
        than a pool or route configuration.
    RuntimeError
        If the allocator cannot be installed: for example, the device budget
        leaves no room for a pool, the backend library is missing, or the
        adapter and runtime libraries come from different builds.
    """

    def __init__(
        self,
        *,
        frontend: RuntimeFrontend,
        pools: Mapping[str, MemoryPoolConfig],
        routes: Mapping[str, TransferRouteConfig],
        library_path: str | Path | None = None,
        calibrate: bool = True,
        numa_binding: bool = True,
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
            _announce("installing the runtime and its pools")
            installed = install_runtime(
                path,
                frontend=frontend,
                numa_binding=numa_binding,
                device_ordinal=topology.device.device,
                device_budget_bytes=topology.device.physical_capacity,
                external_headroom_bytes=topology.device.external_headroom,
                external_baseline_bytes=topology.device.external_baseline,
                physical_limit_bytes=topology.device.physical_limit or 0,
                unattributed_reservation_bytes=topology.device.unattributed_reservation,
                reject_overbudget=topology.device.reject_overbudget,
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
            self._active_plan_handles: set[int] = set()
            self._pending_plan_handle: int | None = None
            self._active_object_references = 0
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
        # A pool opened on an image keeps the image's bytes out of the
        # allocator's hands for its whole life; placed objects borrow them.
        for name, pool in self._pools.items():
            if not pool.image_bytes:
                continue
            status = reserve_pool_range(
                self, pool_id=pool.pool_id, offset=0, size_bytes=pool.image_bytes
            )
            if status != 0:
                self.close()
                raise RuntimeConfigurationError(
                    f"pool {name!r} could not reserve the {pool.image_bytes} bytes "
                    f"of its image: status={status}"
                )
        if calibrate:
            # Low-level callers may calibrate here; Executor uses saved evidence.
            _announce("calibrating transfer routes")
            started = time.perf_counter()
            self._calibrate(routes=None, provenance=INITIALIZATION_PROVENANCE)
            _announce(f"calibrated in {time.perf_counter() - started:.1f}s")

    @property
    def frontend(self) -> RuntimeFrontend:
        """The framework integration this runtime was opened with."""

        return self._frontend

    @property
    def pools(self) -> Mapping[str, MemoryPool]:
        """The runtime's pools, by name; read-only."""

        return self._pools

    def pool_statistics(self, pool: str = "execution") -> MemoryPoolStatistics:
        """Read what one pool holds now.

        Parameters
        ----------
        pool : str, default "execution"
            Name of the pool, as given to the runtime.

        Returns
        -------
        MemoryPoolStatistics
            The pool's capacity, occupancy, fragmentation and record reserves.

        Raises
        ------
        RuntimeConfigurationError
            If the runtime is closed, or no pool has that name.
        RuntimeExecutionError
            If the runtime refuses the read.
        """

        statistics = MemoryPoolStatistics()
        with self._lock:
            self._require_unclosed()
            registered = self._pools.get(pool)
            if registered is None:
                raise RuntimeConfigurationError(f"unknown pool {pool!r}")
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
        """Take a new plan id.

        Ids count up from one and are never reissued; each creates one plan.

        Returns
        -------
        int
            The id, at least one.

        Raises
        ------
        RuntimeConfigurationError
            If the runtime is closed.
        RuntimeExecutionError
            If the runtime has latched a failure.
        """

        plan_id = ctypes.c_uint64()
        with self._lock:
            self._require_unclosed()
            status = int(
                runtime_library().shadowspill_runtime_next_plan_id(
                    self._runtime_handle, ctypes.byref(plan_id)
                )
            )
        if status != 0 or plan_id.value == 0:
            raise RuntimeExecutionError(f"failed to take a plan id: status={status}")
        return int(plan_id.value)

    def plan_state(self, plan_id: int) -> PlanState:
        """Read what became of one plan id, even after its plan is gone.

        Parameters
        ----------
        plan_id : int
            A plan id, such as the plan a live allocation names. Positive.

        Returns
        -------
        PlanState
            Whether the id was never used, or its plan is live, closed or
            destroyed.

        Raises
        ------
        RuntimeConfigurationError
            If the runtime is closed.
        RuntimeExecutionError
            If ``plan_id`` is zero, or the runtime refuses the read.
        """

        state = ctypes.c_uint32()
        with self._lock:
            self._require_unclosed()
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
        """The runtime's directed routes, by name; read-only."""

        return self._routes

    @property
    def transfer_capabilities(self) -> TransferCapabilities:
        """The transfer matrix the runtime publishes now, as one snapshot.

        Raises
        ------
        RuntimeConfigurationError
            If the runtime is closed or unusable, or the matrix cannot be read.
        """

        with self._lock:
            self._require_open()
            return read_transfer_capabilities(self._runtime_handle, self._pool_names)

    def calibrate_transfer_capabilities(
        self,
        *,
        routes: Sequence[tuple[str, str]] | None = None,
        small_copy_bytes: int = 4096,
        large_copy_bytes: int = 256 << 20,
        warmup_copies: int = 4,
        measured_copies: int = 16,
    ) -> TransferCapabilities:
        """Measure routes and publish a new transfer matrix.

        Each measured route first makes ``warmup_copies`` large copies. Its
        latency is the median of ``measured_copies`` small copies timed one at
        a time; its bandwidth comes from the median of three batches of
        ``measured_copies`` large copies. When both directions between two
        pools are measured, they are then measured again together, and each
        uses that concurrent rate. A successful call publishes the whole
        matrix at once, under the next generation.

        The call waits for this runtime to go idle and takes no barrier with
        other processes, so callers that coordinate several runtimes can
        measure them at once to see contended links.

        Parameters
        ----------
        routes : sequence of tuple of str, optional
            The ``(source, destination)`` pool names of the routes to measure.
            By default, every route.
        small_copy_bytes : int, default 4096
            Size of the copies that measure latency, in bytes. Positive.
        large_copy_bytes : int, default 268435456
            Size of the copies that measure bandwidth, in bytes (256 MiB by
            default). At least ``small_copy_bytes``.
        warmup_copies : int, default 4
            Large copies made and discarded before measuring.
        measured_copies : int, default 16
            Copies timed per measurement. Positive.

        Returns
        -------
        TransferCapabilities
            The matrix the call published.

        Raises
        ------
        RuntimeConfigurationError
            If the runtime is closed or unusable, a plan is admitted or being
            set up, a pair is not a route of the runtime, a size or count is
            out of range, or a measurement fails.
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
        """Close the runtime and release everything it owns.

        Closing a closed runtime does nothing. Afterwards every other method
        raises `RuntimeConfigurationError`; ``frontend``, ``pools`` and
        ``routes`` still answer. The process's device allocator stays
        installed, since it cannot be removed, but refuses every later nonzero
        device allocation, and no other runtime can be opened in the process.

        Raises
        ------
        RuntimeConfigurationError
            If a plan is admitted or being set up, an `ObjectRef` is still
            open, or device outputs the caller holds still reference a pool,
            which leaves the runtime open; or if closing released everything
            but observed a failure, which leaves it closed.
        """

        with self._lock:
            if self._closed:
                return
            if self._active_plan_handles or self._pending_plan_handle is not None:
                raise RuntimeConfigurationError(
                    "cannot close Runtime while an active plan or plan setup owns it"
                )
            if self._active_object_references != 0:
                raise RuntimeConfigurationError(
                    "cannot close Runtime while public object references remain; "
                    "release every retained object reference first"
                )
            status = int(self._installed.library.shadowspill_pytorch_allocator_close())
            # A refused close leaves the runtime published. One that released
            # everything returns the failure it observed, and a latched
            # INVALID_STATE is the same status a refusal returns.
            if status == _RUNTIME_INVALID_STATE and self._published():
                raise RuntimeConfigurationError(
                    "cannot close Runtime while caller-owned device outputs "
                    "still reference its memory pools; release those references first"
                )
            self._closed = True
            self._runtime_handle = 0
            if status != 0:
                raise RuntimeConfigurationError(
                    "runtime close released its resources after observing "
                    f"status {status}"
                )

    def _published(self) -> bool:
        """Whether the adapter still publishes this runtime to its callers."""

        handle = ctypes.c_size_t()
        status = self._installed.library.shadowspill_pytorch_runtime_handle(
            ctypes.byref(handle)
        )
        return int(status) == 0

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
            if self._active_plan_handles or self._pending_plan_handle is not None:
                raise RuntimeConfigurationError(
                    "transfer calibration requires no active plan or plan setup"
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
        self._require_unclosed()
        if self._unusable_reason is not None:
            raise RuntimeConfigurationError(
                f"ShadowSpill Runtime is unusable: {self._unusable_reason}"
            )

    def _require_unclosed(self) -> None:
        # Closing frees the neutral runtime, so nothing may reach it after. An
        # unusable runtime is still allocated, and its reads stay answerable.
        if self._closed:
            raise RuntimeConfigurationError("ShadowSpill Runtime is closed")


__all__ = ["PlanState", "Runtime"]
