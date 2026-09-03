"""Explicit process-lifetime runtime initialization for the PyTorch frontend."""

from __future__ import annotations

import ctypes
import hashlib
import json
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch

from shadowspill.errors import AdmissionError
from shadowspill.libraries import resolve_library
from shadowspill.memory import (
    DevicePool,
    MemoryPoolConfig,
    PinnedHostPool,
)
from shadowspill.memory import (
    TransferRoute as TransferRouteConfig,
)
from shadowspill.planner.quantization import GIBIBYTE, floored
from shadowspill.pytorch.accelerator import accelerator_device, is_accelerator
from shadowspill.pytorch.runtime_adapter.abi import (
    PlanDescription,
    TransferCalibrationConfig,
    TransferRouteKey,
    runtime_library,
)
from shadowspill.pytorch.runtime_adapter.abi import (
    TransferProfile as RuntimeTransferProfile,
)
from shadowspill.pytorch.runtime_adapter.allocator import (
    InstalledAllocator,
    PoolBootstrap,
    RouteBootstrap,
    install_allocator,
)
from shadowspill.pytorch.runtime_adapter.failures import (
    RuntimeExecutionError,
    RuntimeFailureDiagnostics,
    read_allocator_failure,
)
from shadowspill.runtime import ObjectRef
from shadowspill.runtime.topology import (
    MemoryPool,
    RuntimeRoute,
    TransferCapabilities,
    TransferProfile,
)
from shadowspill.status import ABI_VERSION, Status

_INITIALIZATION_PROVENANCE = 0
_RECALIBRATION_PROVENANCE = 1
_RUNTIME_INVALID_STATE = Status.INVALID_STATE


class RuntimeConfigurationError(RuntimeError):
    """Raised when a runtime or plan asks for incompatible pool resources."""


@dataclass(frozen=True, slots=True)
class PlanMemory:
    """Resolved pool roles and capacities consumed by one planning call."""

    runtime: Runtime
    installed: InstalledAllocator
    execution: MemoryPool
    spill: MemoryPool
    fetch: RuntimeRoute
    evict: RuntimeRoute
    execution_budget: int
    spill_budget: int
    dynamic_scratch_reserve_bytes: int
    execution_device: int
    transfers: TransferCapabilities
    plan_handle: int


_runtime_lock = threading.Lock()
_active_runtime: Runtime | None = None


class Runtime:
    """Own ShadowSpill's process-lifetime allocator, pools, routes, and worker.

    Accelerator allocator selection is process-global and irreversible after
    PyTorch initializes the accelerator. Construct exactly one ``Runtime``
    before any accelerator tensor allocation, then pass it to every
    ``plan_step`` or ``plan_forward`` call.

    The current backend supports one device pool plus any number of pinned-host
    pools. Pools and directed routes have explicit identities; each admitted
    callable independently selects its execution/spill pool pair and matching
    routes.
    """

    def __init__(
        self,
        *,
        pools: Mapping[str, MemoryPoolConfig],
        routes: Mapping[str, TransferRouteConfig],
        library_path: str | Path | None = None,
        calibrate: bool = True,
        worker_poll_nanoseconds: int = 1_000,
        backend: str | None = None,
    ) -> None:
        normalized, normalized_routes = _validate_topology(pools, routes)
        device_name, device_config = next(
            (name, config)
            for name, config in normalized.items()
            if isinstance(config, DevicePool)
        )
        pool_names = tuple(normalized)
        pool_ids = {name: index for index, name in enumerate(pool_names)}
        route_names = tuple(normalized_routes)
        allocator_pool_id = pool_ids[device_name]
        pool_bootstrap = tuple(
            PoolBootstrap(
                pool_id=pool_ids[name],
                kind=0 if isinstance(config, DevicePool) else 1,
                capacity_bytes=(
                    0 if isinstance(config, DevicePool) else config.capacity
                ),
            )
            for name, config in normalized.items()
        )
        route_bootstrap = tuple(
            RouteBootstrap(
                route_id=index,
                name=name,
                source_pool_id=pool_ids[route.source],
                destination_pool_id=pool_ids[route.destination],
            )
            for index, (name, route) in enumerate(normalized_routes.items())
        )
        path = _adapter_path(library_path)
        global _active_runtime
        with _runtime_lock:
            if _active_runtime is not None:
                raise RuntimeConfigurationError(
                    "a ShadowSpill Runtime is already initialized in this process"
                )
            installed = install_allocator(
                path,
                device_ordinal=device_config.device,
                device_budget_bytes=device_config.physical_capacity,
                provider_headroom_bytes=device_config.provider_headroom,
                allocator_pool_id=allocator_pool_id,
                pools=pool_bootstrap,
                routes=route_bootstrap,
                worker_poll_nanoseconds=worker_poll_nanoseconds,
                backend=backend,
            )
            self._installed = installed
            # The neutral runtime this process bound. Holding it here is what
            # lets the calls that need nothing else go straight to the neutral
            # library, instead of through an entry point that would only fetch
            # this pointer and forward. It is dropped on close, so a call after
            # close fails on the closed guard rather than on a stale pointer.
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
            initialized_pools = {
                name: MemoryPool(
                    name=name,
                    pool_id=pool_ids[name],
                    kind=(
                        "device" if isinstance(config, DevicePool) else "pinned_host"
                    ),
                    capacity=(
                        int(installed.admission.allocator_pool_bytes)
                        if isinstance(config, DevicePool)
                        else config.capacity
                    ),
                    physical_capacity=(
                        config.physical_capacity
                        if isinstance(config, DevicePool)
                        else config.capacity
                    ),
                    device_ordinal=(
                        config.device if isinstance(config, DevicePool) else None
                    ),
                )
                for name, config in normalized.items()
            }
            initialized_routes = {
                name: RuntimeRoute(
                    name=name,
                    route_id=index,
                    source=route.source,
                    destination=route.destination,
                    source_pool_id=pool_ids[route.source],
                    destination_pool_id=pool_ids[route.destination],
                )
                for index, (name, route) in enumerate(normalized_routes.items())
            }
            self._pools = MappingProxyType(initialized_pools)
            self._routes = MappingProxyType(initialized_routes)
            self._route_by_pair = MappingProxyType(
                {
                    (route.source, route.destination): route
                    for route in initialized_routes.values()
                }
            )
            self._pool_names = pool_names
            self._route_names = route_names
            _active_runtime = self
        if calibrate:
            self._calibrate(routes=None, provenance=_INITIALIZATION_PROVENANCE)

    @property
    def pools(self) -> Mapping[str, MemoryPool]:
        """Read-only initialized pool registry keyed by user names."""

        return self._pools

    @property
    def routes(self) -> Mapping[str, RuntimeRoute]:
        """Read-only directed-route registry keyed by user names."""

        return self._routes

    @property
    def transfer_capabilities(self) -> TransferCapabilities:
        """Return a lock-consistent immutable transfer-matrix snapshot."""

        with self._lock:
            self._require_open()
            return self._read_transfer_capabilities()

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
            provenance=_RECALIBRATION_PROVENANCE,
            small_copy_bytes=small_copy_bytes,
            large_copy_bytes=large_copy_bytes,
            warmup_copies=warmup_copies,
            measured_copies=measured_copies,
        )

    def close(self) -> None:
        """Close every runtime resource after verifying external ownership.

        PyTorch's selected allocator shim remains installed because allocator
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
                    "cannot close Runtime while persistent PyTorch state remains; "
                    "export it with release_runtime=True first"
                )
            if self._active_object_references != 0:
                raise RuntimeConfigurationError(
                    "cannot close Runtime while public object references remain; "
                    "close every TensorRef or StateRef first"
                )
            status = int(self._installed.library.shadowspill_pytorch_allocator_close())
            if status == _RUNTIME_INVALID_STATE:
                raise RuntimeConfigurationError(
                    "cannot close Runtime while caller-owned device outputs "
                    "still reference its memory pools; release those tensors first"
                )
            self._closed = True
            if status != 0:
                raise RuntimeConfigurationError(
                    "runtime close released its resources after observing "
                    f"status {status}"
                )

    def _reserve_persistent_object_ids(
        self,
        count: int,
        *,
        allow_in_progress_plan: bool = False,
    ) -> tuple[int, ...]:
        """Reserve globally unique runtime object identities."""

        if count < 0:
            raise ValueError("persistent object count must be non-negative")
        with self._lock:
            self._require_state_operation_allowed(
                allow_in_progress_plan=allow_in_progress_plan
            )
            return self._reserve_runtime_object_ids(count)

    def _reserve_runtime_object_ids(self, count: int) -> tuple[int, ...]:
        """Reserve runtime-global identities without changing state ownership."""

        if count < 0:
            raise ValueError("runtime object count must be non-negative")
        with self._lock:
            self._require_open()
            first = self._next_persistent_object_id
            limit = first + count
            if limit >= (1 << 63):
                raise RuntimeConfigurationError(
                    "persistent PyTorch object identity space is exhausted"
                )
            self._next_persistent_object_id = limit
            return tuple(range(first, limit))

    def _acquire_object_reference(
        self,
        *,
        object_id: int,
        size_bytes: int,
    ) -> ObjectRef:
        """Create one public owner for an existing runtime object."""

        with self._lock:
            self._require_open()
            handle = ctypes.c_size_t()
            status = int(
                runtime_library().shadowspill_object_handle_acquire(
                    self._runtime_handle, object_id, ctypes.byref(handle)
                )
            )
            if status != 0 or handle.value == 0:
                raise RuntimeExecutionError(
                    f"failed to retain runtime object {object_id}: status={status}"
                )
            try:
                reference = ObjectRef(
                    self,
                    object_id=object_id,
                    size_bytes=size_bytes,
                    handle=int(handle.value),
                )
            except BaseException:
                runtime_library().shadowspill_object_handle_release(handle.value)
                raise
            self._active_object_references += 1
            return reference

    def _release_object_reference(self, reference: ObjectRef) -> None:
        """Release exactly one public runtime-object owner."""

        with self._lock:
            if not reference._belongs_to(self):
                raise RuntimeError(
                    "runtime object reference belongs to another Runtime"
                )
            if self._active_object_references <= 0:
                raise RuntimeError("runtime object reference ownership underflow")
            status = int(
                runtime_library().shadowspill_object_handle_release(
                    reference._require_handle()
                )
            )
            if status != 0:
                raise RuntimeExecutionError(
                    "failed to release runtime object "
                    f"{reference.object_id}: status={status}"
                )
            self._active_object_references -= 1

    def _release_object_generation(
        self,
        *,
        object_id: int,
        expected_generation: int,
    ) -> None:
        """Release a completed value while retaining its logical identity."""

        with self._lock:
            self._require_open()
            handle = ctypes.c_size_t()
            status = int(
                runtime_library().shadowspill_object_handle_acquire(
                    self._runtime_handle, object_id, ctypes.byref(handle)
                )
            )
            if status != 0 or handle.value == 0:
                raise RuntimeExecutionError(
                    "failed to resolve runtime object generation "
                    f"{object_id}: status={status}"
                )
            operation_status = 0
            try:
                operation_status = int(
                    runtime_library().shadowspill_object_release_generation(
                        handle.value, expected_generation
                    )
                )
            finally:
                release_status = int(
                    runtime_library().shadowspill_object_handle_release(handle.value)
                )
            if operation_status != 0:
                raise RuntimeExecutionError(
                    "failed to release runtime object generation "
                    f"{object_id}/{expected_generation}: "
                    f"status={operation_status}"
                )
            if release_status != 0:
                raise RuntimeExecutionError(
                    "failed to release temporary runtime object handle "
                    f"{object_id}: status={release_status}"
                )

    def _retain_persistent_state(self, *, allow_in_progress_plan: bool = False) -> None:
        with self._lock:
            self._require_state_operation_allowed(
                allow_in_progress_plan=allow_in_progress_plan
            )
            self._persistent_state_count += 1

    def _release_persistent_state(self) -> None:
        with self._lock:
            if self._persistent_state_count <= 0:
                raise RuntimeError("persistent state ownership underflow")
            self._persistent_state_count -= 1

    def _require_state_operation_allowed(
        self, *, allow_in_progress_plan: bool = False
    ) -> None:
        with self._lock:
            self._require_open()
            if self._active_plan_handles or (
                self._planning_plan_handle is not None and not allow_in_progress_plan
            ):
                raise RuntimeConfigurationError(
                    "persistent state import requires an idle Runtime"
                )

    def __enter__(self) -> Runtime:
        self._require_open()
        return self

    def __exit__(self, *exception: object) -> None:
        del exception
        self.close()

    def _resolve_plan(
        self,
        *,
        execution: str,
        spill: str,
        execution_budget: int | None,
        spill_budget: int | None,
        dynamic_scratch_reserve_bytes: int | None,
        execution_device: object | None,
    ) -> PlanMemory:
        with self._lock:
            self._require_open()
            if self._planning_plan_handle is not None:
                raise RuntimeConfigurationError(
                    "this Runtime already has an in-progress planning call"
                )
            if execution == spill:
                raise RuntimeConfigurationError(
                    "execution and spill must select distinct pools"
                )
            try:
                execution_pool = self._pools[execution]
                spill_pool = self._pools[spill]
            except KeyError as exc:
                raise RuntimeConfigurationError(
                    f"unknown runtime pool {exc.args[0]!r}"
                ) from exc
            if execution_pool.kind != "device":
                raise RuntimeConfigurationError(
                    "the current PyTorch frontend requires an accelerator "
                    "execution pool"
                )
            resolved_device = _resolve_execution_device(
                execution_device, execution_pool
            )
            resolved_execution = floored(
                _resolve_execution_budget(execution_budget, execution_pool), GIBIBYTE
            )
            resolved_spill = floored(
                _resolve_budget(spill_budget, spill_pool, "spill_budget"), GIBIBYTE
            )
            resolved_scratch = _resolve_dynamic_scratch_reserve(
                dynamic_scratch_reserve_bytes,
                execution_budget=resolved_execution,
            )
            transfers = self._read_transfer_capabilities()
            try:
                fetch_route = self._route_by_pair[(spill, execution)]
                evict_route = self._route_by_pair[(execution, spill)]
            except KeyError as exc:
                source, destination = exc.args[0]
                raise RuntimeConfigurationError(
                    f"runtime has no directed route {source!r} -> {destination!r}"
                ) from exc
            fetch_profile = transfers.route(spill, execution)
            evict_profile = transfers.route(execution, spill)
            if not fetch_profile.available or not fetch_profile.calibrated:
                raise RuntimeConfigurationError(
                    f"route {spill!r} -> {execution!r} is not calibrated"
                )
            if not evict_profile.available or not evict_profile.calibrated:
                raise RuntimeConfigurationError(
                    f"route {execution!r} -> {spill!r} is not calibrated"
                )
            plan_handle_value = ctypes.c_size_t()
            status = int(
                runtime_library().shadowspill_plan_create(
                    self._runtime_handle,
                    ctypes.byref(
                        PlanDescription(
                            execution_pool_id=execution_pool.pool_id,
                            spill_pool_id=spill_pool.pool_id,
                            fetch_route_id=fetch_route.route_id,
                            evict_route_id=evict_route.route_id,
                        )
                    ),
                    ctypes.byref(plan_handle_value),
                )
            )
            if status != 0 or plan_handle_value.value == 0:
                raise RuntimeConfigurationError(
                    "handle plan creation failed: "
                    f"status={status}, execution={execution!r}, spill={spill!r}"
                )
            plan_handle = int(plan_handle_value.value)
            memory = PlanMemory(
                runtime=self,
                installed=self._installed,
                execution=execution_pool,
                spill=spill_pool,
                fetch=fetch_route,
                evict=evict_route,
                execution_budget=resolved_execution,
                spill_budget=resolved_spill,
                dynamic_scratch_reserve_bytes=resolved_scratch,
                execution_device=resolved_device,
                transfers=transfers,
                plan_handle=plan_handle,
            )
            self._planning_plan_handle = plan_handle
            return memory

    def _adopt_plan(self, plan_handle: int) -> None:
        with self._lock:
            self._require_open()
            if self._planning_plan_handle != plan_handle:
                raise RuntimeConfigurationError(
                    "Runtime does not own this in-progress plan"
                )
            self._planning_plan_handle = None
            self._active_plan_handles.add(plan_handle)

    def _release_plan(self, plan_handle: int) -> None:
        with self._lock:
            if plan_handle not in self._active_plan_handles:
                raise RuntimeError("Runtime plan ownership underflow")
            try:
                self._close_and_destroy_plan(plan_handle)
            except BaseException as error:
                self._unusable_reason = f"execution-plan teardown failed: {error}"
                raise
            finally:
                self._active_plan_handles.discard(plan_handle)

    def _wait_plan_idle(self, plan_handle: int) -> None:
        """Block until the plan has no work in flight."""

        status = int(runtime_library().shadowspill_plan_wait_idle(plan_handle))
        if status != 0:
            raise RuntimeError(
                f"compiled executor did not become idle (status {status})"
            )

    def _abort_plan(self, plan_handle: int | None = None) -> None:
        """Release cold-path task records after a failed planning call."""

        with self._lock:
            target = self._planning_plan_handle if plan_handle is None else plan_handle
            if target is None or self._planning_plan_handle != target:
                raise RuntimeError("Runtime planning ownership underflow")
            try:
                self._close_and_destroy_plan(target)
            except BaseException as error:
                self._unusable_reason = f"execution-plan rollback failed: {error}"
                raise
            finally:
                self._planning_plan_handle = None

    def _close_and_destroy_plan(self, plan_handle: int) -> None:
        status = int(runtime_library().shadowspill_plan_close(plan_handle))
        if status != 0:
            raise RuntimeConfigurationError(
                f"handle plan close failed with status {status}"
            )
        runtime_library().shadowspill_plan_destroy(plan_handle)

    def _prepare_failure_cleanup(
        self,
        error: BaseException,
        *,
        operation: str,
        synchronize_unlatched: bool,
    ) -> None:
        """Record handle failure state and safely prepare runtime teardown."""

        if isinstance(error, RuntimeExecutionError) and not error._begin_cleanup():
            return
        diagnostics = (
            error.diagnostics if isinstance(error, RuntimeExecutionError) else None
        )
        if diagnostics is None:
            diagnostics = read_allocator_failure(self._installed.library, operation)
        if diagnostics is not None:
            self._record_failure(diagnostics)
        elif not synchronize_unlatched:
            return
        try:
            torch.cuda.synchronize(int(self._installed.admission.device_ordinal))
        except BaseException as synchronize_error:
            error.add_note(
                "Failed to synchronize the execution device during fault cleanup: "
                f"{synchronize_error}"
            )
            self._mark_unusable("execution-device synchronization failed")
            return
        if diagnostics is None or not diagnostics.is_recoverable_no_progress:
            return
        status = int(self._installed.library.shadowspill_pytorch_recover_no_progress())
        if status != 0:
            error.add_note(
                f"Failed to recover the no-progress latch for teardown: status {status}"
            )
            self._mark_unusable(
                f"no-progress teardown recovery failed with status {status}"
            )

    def _record_failure(self, diagnostics: RuntimeFailureDiagnostics) -> None:
        with self._lock:
            self._last_failure = diagnostics

    def _mark_unusable(self, reason: str) -> None:
        with self._lock:
            if self._unusable_reason is None:
                self._unusable_reason = reason

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
            keys: Any = None
            count = 0
            if routes is not None:
                encoded: list[TransferRouteKey] = []
                for source, destination in routes:
                    try:
                        source_id = self._pool_names.index(source)
                        destination_id = self._pool_names.index(destination)
                    except ValueError as exc:
                        raise RuntimeConfigurationError(
                            f"unknown transfer route {(source, destination)!r}"
                        ) from exc
                    encoded.append(TransferRouteKey(source_id, destination_id))
                count = len(encoded)
                keys = (TransferRouteKey * count)(*encoded) if count else None
            config = TransferCalibrationConfig(
                abi_version=ABI_VERSION,
                small_copy_bytes=small_copy_bytes,
                large_copy_bytes=large_copy_bytes,
                warmup_copies=warmup_copies,
                measured_copies=measured_copies,
                provenance=provenance,
            )
            status = int(
                self._installed.library.shadowspill_pytorch_calibrate_transfer_capabilities(
                    ctypes.byref(config), keys, count
                )
            )
            if status != 0:
                raise RuntimeConfigurationError(
                    f"transfer calibration failed with status {status}"
                )
            return self._read_transfer_capabilities()

    def _read_transfer_capabilities(self) -> TransferCapabilities:
        count = ctypes.c_uint32()
        generation = ctypes.c_uint64()
        status = int(
            self._installed.library.shadowspill_pytorch_transfer_profiles(
                None, 0, ctypes.byref(count), ctypes.byref(generation)
            )
        )
        if status != 0:
            raise RuntimeConfigurationError(
                f"transfer-profile size query failed with status {status}"
            )
        handle = (RuntimeTransferProfile * count.value)()
        status = int(
            self._installed.library.shadowspill_pytorch_transfer_profiles(
                handle,
                count.value,
                ctypes.byref(count),
                ctypes.byref(generation),
            )
        )
        if status != 0:
            raise RuntimeConfigurationError(
                f"transfer-profile read failed with status {status}"
            )
        profiles = tuple(
            TransferProfile(
                source=self._pool_names[item.source_pool_id],
                destination=self._pool_names[item.destination_pool_id],
                source_pool_id=int(item.source_pool_id),
                destination_pool_id=int(item.destination_pool_id),
                generation=int(item.generation),
                latency_nanoseconds=int(item.latency_nanoseconds),
                bandwidth_bytes_per_second=int(item.bandwidth_bytes_per_second),
                solo_bandwidth_bytes_per_second=int(
                    item.solo_bandwidth_bytes_per_second
                ),
                concurrent_bandwidth_bytes_per_second=int(
                    item.concurrent_bandwidth_bytes_per_second
                ),
                solo_measurement_nanoseconds=int(item.solo_measurement_nanoseconds),
                concurrent_measurement_nanoseconds=int(
                    item.concurrent_measurement_nanoseconds
                ),
                calibrated_timestamp_nanoseconds=int(
                    item.calibrated_timestamp_nanoseconds
                ),
                small_copy_bytes=int(item.small_copy_bytes),
                large_copy_bytes=int(item.large_copy_bytes),
                measured_copies=int(item.measured_copies),
                available=bool(item.available),
                calibrated=bool(item.calibrated),
                provenance=(
                    "initialization"
                    if int(item.provenance) == _INITIALIZATION_PROVENANCE
                    else "recalibration"
                ),
                calibration_mode={
                    0: "identity",
                    1: "solo",
                    2: "bidirectional_concurrent",
                }.get(int(item.calibration_mode), "unknown"),
                concurrent_route_count=int(item.concurrent_route_count),
            )
            for item in handle
        )
        canonical = {
            "generation": int(generation.value),
            "pool_names": list(self._pool_names),
            "profiles": [profile.as_dict() for profile in profiles],
        }
        digest = hashlib.sha256(
            json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return TransferCapabilities(
            generation=int(generation.value),
            pool_names=self._pool_names,
            profiles=profiles,
            digest=digest,
        )

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeConfigurationError("ShadowSpill Runtime is closed")
        if self._unusable_reason is not None:
            raise RuntimeConfigurationError(
                f"ShadowSpill Runtime is unusable: {self._unusable_reason}"
            )


def _validate_topology(
    pools: Mapping[str, MemoryPoolConfig],
    routes: Mapping[str, TransferRouteConfig],
) -> tuple[dict[str, MemoryPoolConfig], dict[str, TransferRouteConfig]]:
    if not isinstance(pools, Mapping):
        raise TypeError("pools must be a mapping from names to pool configurations")
    normalized = dict(pools)
    if len(normalized) < 2:
        raise RuntimeConfigurationError("a runtime requires at least two memory pools")
    for name, config in normalized.items():
        if not isinstance(name, str) or not name or not name.isidentifier():
            raise RuntimeConfigurationError(
                f"pool name {name!r} must be a non-empty identifier"
            )
        if not isinstance(config, (DevicePool, PinnedHostPool)):
            raise TypeError(f"unsupported pool configuration for {name!r}")
    if sum(isinstance(value, DevicePool) for value in normalized.values()) != 1:
        raise RuntimeConfigurationError(
            "the current PyTorch allocator frontend requires exactly one device pool"
        )
    if not any(isinstance(value, PinnedHostPool) for value in normalized.values()):
        raise RuntimeConfigurationError(
            "the current runtime backend requires at least one pinned-host pool"
        )

    if not isinstance(routes, Mapping):
        raise TypeError("routes must be a mapping from names to route configurations")
    normalized_routes = dict(routes)
    if not normalized_routes:
        raise RuntimeConfigurationError(
            "a runtime requires at least one transfer route"
        )
    endpoint_pairs: set[tuple[str, str]] = set()
    for name, route in normalized_routes.items():
        if not isinstance(name, str) or not name or not name.isidentifier():
            raise RuntimeConfigurationError(
                f"route name {name!r} must be a non-empty identifier"
            )
        if not isinstance(route, TransferRouteConfig):
            raise TypeError(f"unsupported route configuration for {name!r}")
        for endpoint in (route.source, route.destination):
            if endpoint not in normalized:
                raise RuntimeConfigurationError(
                    f"route {name!r} references unknown pool {endpoint!r}"
                )
        pair = (route.source, route.destination)
        if pair in endpoint_pairs:
            raise RuntimeConfigurationError(
                "runtime route endpoint pairs must be unique; duplicate "
                f"{route.source!r} -> {route.destination!r}"
            )
        endpoint_pairs.add(pair)
        source = normalized[route.source]
        destination = normalized[route.destination]
        if isinstance(source, DevicePool) == isinstance(destination, DevicePool):
            raise RuntimeConfigurationError(
                "the current backend supports routes only between a device pool "
                "and a pinned-host pool"
            )
    return normalized, normalized_routes


def _resolve_budget(value: int | None, pool: MemoryPool, name: str) -> int:
    if value is None:
        return pool.capacity
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer byte count or None")
    if value <= 0:
        raise AdmissionError(f"{name} must be positive")
    if value > pool.capacity:
        raise AdmissionError(
            f"{name}={value} exceeds pool {pool.name!r} capacity={pool.capacity}"
        )
    return value


def _resolve_execution_budget(value: int | None, pool: MemoryPool) -> int:
    """Resolve the common physical-cap spelling to suballocatable bytes.

    Runtime initialization subtracts the one-time accelerator problem and
    provider allowance from ``physical_capacity`` before creating the
    execution pool.  Users naturally repeat that same physical cap at the
    planning boundary.  Treating it as a raw pool size charges those fixed
    bytes twice and rejects the most common call shape.

    Values at or below ``pool.capacity`` retain the existing per-plan logical
    limit semantics.  A value strictly between the derived pool capacity and
    the configured physical cap is ambiguous and is rejected rather than
    pretending that the already allocated process slab became smaller.
    """

    if value is None:
        return pool.capacity
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("execution_budget must be an integer byte count or None")
    if value <= 0:
        raise AdmissionError("execution_budget must be positive")
    physical_capacity = pool.physical_capacity
    if physical_capacity is not None and value == physical_capacity:
        return pool.capacity
    if value <= pool.capacity:
        return value
    if physical_capacity is not None and value < physical_capacity:
        raise AdmissionError(
            "execution_budget falls between the initialized execution-pool "
            "capacity and its complete physical cap; pass the runtime physical "
            "cap for the full pool, or a value no larger than the derived pool "
            f"capacity={pool.capacity}"
        )
    limit = physical_capacity if physical_capacity is not None else pool.capacity
    raise AdmissionError(
        f"execution_budget={value} exceeds pool {pool.name!r} physical capacity={limit}"
    )


def _resolve_dynamic_scratch_reserve(
    requested: int | None,
    *,
    execution_budget: int,
) -> int:
    """Validate an optional minimum for bounded task-path insertions."""

    if requested is None:
        return 0
    if isinstance(requested, bool) or not isinstance(requested, int):
        raise TypeError("dynamic_scratch_reserve_bytes must be an integer byte count")
    if requested < 0:
        raise AdmissionError("dynamic_scratch_reserve_bytes must be non-negative")
    if requested > execution_budget:
        raise AdmissionError(
            "dynamic_scratch_reserve_bytes exceeds execution_budget: "
            f"reserve={requested}, budget={execution_budget}"
        )
    return requested


def _resolve_execution_device(value: object | None, pool: MemoryPool) -> int:
    """Resolve and, when explicit, select the PyTorch execution device."""

    import torch

    pool_device = pool.device_ordinal
    if pool_device is None:
        raise RuntimeConfigurationError(
            f"execution pool {pool.name!r} has no accelerator device"
        )
    if value is None:
        resolved = int(torch.cuda.current_device())
    else:
        if isinstance(value, bool):
            raise TypeError(
                "execution_device must be an accelerator device, ordinal, or None"
            )
        if isinstance(value, int):
            resolved_device = accelerator_device(value)
        else:
            if not isinstance(value, (str, torch.device)):
                raise TypeError(
                    "execution_device must be an accelerator device, ordinal, or None"
                )
            try:
                resolved_device = torch.device(value)
            except (TypeError, RuntimeError) as exc:
                raise TypeError(
                    "execution_device must be an accelerator device, ordinal, or None"
                ) from exc
        if not is_accelerator(resolved_device):
            raise RuntimeConfigurationError(
                "the installed PyTorch adapter currently requires an accelerator "
                "execution device"
            )
        resolved = (
            int(torch.cuda.current_device())
            if resolved_device.index is None
            else int(resolved_device.index)
        )
    if resolved != pool_device:
        raise RuntimeConfigurationError(
            f"execution_device={resolved} does not match execution pool "
            f"{pool.name!r} device={pool_device}"
        )
    if value is not None:
        torch.cuda.set_device(resolved)
    return resolved


def _adapter_path(configured: str | Path | None) -> Path:
    if configured is not None:
        path = Path(configured).expanduser().resolve()
    else:
        discovered = resolve_library("libshadowspill_pytorch.so")
        if discovered is None:
            raise RuntimeConfigurationError(
                "ShadowSpill's PyTorch adapter was not found; install "
                "ShadowSpill or build the editable checkout at its configured "
                "build location"
            )
        path = discovered
    if not path.is_file():
        raise RuntimeConfigurationError(
            f"ShadowSpill's PyTorch adapter was not found: {path}"
        )
    return path


__all__ = [
    "MemoryPool",
    "PlanMemory",
    "Runtime",
    "RuntimeConfigurationError",
    "RuntimeRoute",
    "TransferCapabilities",
    "TransferProfile",
]
