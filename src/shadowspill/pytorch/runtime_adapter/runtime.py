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

from shadowspill._libraries import resolve_library
from shadowspill.memory import DevicePool, MemoryPoolConfig, PinnedHostPool
from shadowspill.pytorch.contracts import AdmissionError
from shadowspill.pytorch.runtime_adapter.abi import (
    TRANSFER_PROFILE_ABI_VERSION,
    TransferCalibrationConfig,
    TransferRouteKey,
)
from shadowspill.pytorch.runtime_adapter.abi import (
    TransferProfile as NativeTransferProfile,
)
from shadowspill.pytorch.runtime_adapter.allocator import (
    InstalledAllocator,
    install_allocator,
)
from shadowspill.pytorch.runtime_adapter.failures import (
    ExecutionTaskIdentity,
    RuntimeExecutionError,
    RuntimeFailureDiagnostics,
    allocator_oom_error,
    generic_runtime_error,
    read_allocator_failure,
)

_INITIALIZATION_PROVENANCE = 0
_RECALIBRATION_PROVENANCE = 1


class RuntimeConfigurationError(RuntimeError):
    """Raised when a runtime or plan asks for incompatible pool resources."""


@dataclass(frozen=True, slots=True)
class MemoryPool:
    """One initialized runtime pool visible to planning."""

    name: str
    pool_id: int
    kind: str
    capacity: int
    physical_capacity: int | None
    device_ordinal: int | None


@dataclass(frozen=True, slots=True)
class TransferProfile:
    """Measured performance for one directed pool-pair route."""

    source: str
    destination: str
    source_pool_id: int
    destination_pool_id: int
    generation: int
    latency_nanoseconds: int
    bandwidth_bytes_per_second: int
    calibrated_timestamp_nanoseconds: int
    small_copy_bytes: int
    large_copy_bytes: int
    measured_copies: int
    available: bool
    calibrated: bool
    provenance: str

    def as_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "destination": self.destination,
            "source_pool_id": self.source_pool_id,
            "destination_pool_id": self.destination_pool_id,
            "generation": self.generation,
            "latency_nanoseconds": self.latency_nanoseconds,
            "bandwidth_bytes_per_second": self.bandwidth_bytes_per_second,
            "calibrated_timestamp_nanoseconds": self.calibrated_timestamp_nanoseconds,
            "small_copy_bytes": self.small_copy_bytes,
            "large_copy_bytes": self.large_copy_bytes,
            "measured_copies": self.measured_copies,
            "available": self.available,
            "calibrated": self.calibrated,
            "provenance": self.provenance,
        }


@dataclass(frozen=True, slots=True)
class TransferCapabilities:
    """Immutable dense transfer matrix published by one runtime generation."""

    generation: int
    pool_names: tuple[str, ...]
    profiles: tuple[TransferProfile, ...]
    digest: str

    def route(self, source: str, destination: str) -> TransferProfile:
        try:
            source_id = self.pool_names.index(source)
            destination_id = self.pool_names.index(destination)
        except ValueError as exc:
            raise KeyError((source, destination)) from exc
        return self.profiles[source_id * len(self.pool_names) + destination_id]

    def as_dict(self) -> dict[str, object]:
        return {
            "generation": self.generation,
            "pool_names": list(self.pool_names),
            "profiles": [profile.as_dict() for profile in self.profiles],
            "digest": self.digest,
        }


@dataclass(frozen=True, slots=True)
class PlanMemory:
    """Resolved pool roles and capacities consumed by one planning call."""

    runtime: Runtime
    installed: InstalledAllocator
    execution: MemoryPool
    spill: MemoryPool
    execution_budget: int
    spill_budget: int
    execution_device: int
    transfers: TransferCapabilities


_runtime_lock = threading.Lock()
_active_runtime: Runtime | None = None


class Runtime:
    """Own ShadowSpill's process-lifetime allocator, pools, routes, and worker.

    Accelerator allocator selection is process-global and irreversible after
    PyTorch initializes the accelerator. Construct exactly one ``Runtime``
    before any accelerator tensor allocation, then pass it to every
    ``plan_step`` or ``plan_forward`` call.

    The initial release supports one device pool and one pinned-host pool. The
    public registry and route matrix are already N-pool representations; later
    backends can add peer, remote, or storage pools without changing planning
    call signatures.
    """

    def __init__(
        self,
        *,
        pools: Mapping[str, MemoryPoolConfig],
        library_path: str | Path | None = None,
        calibrate: bool = True,
        worker_poll_nanoseconds: int = 1_000,
    ) -> None:
        normalized = _validate_pool_configs(pools)
        device_name, device_config = next(
            (name, config)
            for name, config in normalized.items()
            if isinstance(config, DevicePool)
        )
        spill_name, spill_config = next(
            (name, config)
            for name, config in normalized.items()
            if isinstance(config, PinnedHostPool)
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
                spill_pool_bytes=spill_config.capacity,
                worker_poll_nanoseconds=worker_poll_nanoseconds,
            )
            self._installed = installed
            self._lock = threading.RLock()
            self._closed = False
            self._unusable_reason: str | None = None
            self._last_failure: RuntimeFailureDiagnostics | None = None
            self._active_plans = 0
            self._planning = False
            self._persistent_state_count = 0
            self._next_persistent_object_id = 1 << 62
            execution_pool = MemoryPool(
                name=device_name,
                pool_id=0,
                kind="device",
                capacity=int(installed.admission.execution_pool_bytes),
                physical_capacity=device_config.physical_capacity,
                device_ordinal=device_config.device,
            )
            spill_pool = MemoryPool(
                name=spill_name,
                pool_id=1,
                kind="pinned_host",
                capacity=spill_config.capacity,
                physical_capacity=spill_config.capacity,
                device_ordinal=None,
            )
            self._pools = MappingProxyType(
                {device_name: execution_pool, spill_name: spill_pool}
            )
            self._pool_names = (device_name, spill_name)
            _active_runtime = self
        if calibrate:
            self._calibrate(routes=None, provenance=_INITIALIZATION_PROVENANCE)

    @property
    def pools(self) -> Mapping[str, MemoryPool]:
        """Read-only initialized pool registry keyed by user names."""

        return self._pools

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
        large_copy_bytes: int = 64 << 20,
        warmup_copies: int = 2,
        measured_copies: int = 5,
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
        """Reject new plans after verifying that no callable remains active.

        PyTorch's selected allocator cannot be uninstalled, so the underlying C
        runtime remains process-owned. Planned callables must be closed first.
        """

        with self._lock:
            if self._closed:
                return
            if self._active_plans != 0 or self._planning:
                raise RuntimeConfigurationError(
                    "cannot close Runtime while a callable or in-progress plan owns it"
                )
            if self._persistent_state_count != 0:
                raise RuntimeConfigurationError(
                    "cannot close Runtime while persistent PyTorch state remains; "
                    "externalize it with release_runtime=True first"
                )
            status = int(
                self._installed.library.shadowspill_pytorch_allocator_wait_idle()
            )
            if status != 0:
                raise RuntimeConfigurationError(
                    f"runtime idle wait failed with status {status}"
                )
            self._closed = True

    def _reserve_persistent_object_ids(
        self,
        count: int,
        *,
        allow_in_progress_plan: bool = False,
    ) -> tuple[int, ...]:
        """Reserve frontend-owned runtime identities outside dense plan IDs."""

        if count < 0:
            raise ValueError("persistent object count must be non-negative")
        with self._lock:
            self._require_state_operation_allowed(
                allow_in_progress_plan=allow_in_progress_plan
            )
            first = self._next_persistent_object_id
            limit = first + count
            if limit >= (1 << 63):
                raise RuntimeConfigurationError(
                    "persistent PyTorch object identity space is exhausted"
                )
            self._next_persistent_object_id = limit
            return tuple(range(first, limit))

    def _retain_persistent_state(
        self, *, allow_in_progress_plan: bool = False
    ) -> None:
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
            if self._active_plans != 0 or (
                self._planning and not allow_in_progress_plan
            ):
                raise RuntimeConfigurationError(
                    "persistent state relocation requires an idle Runtime"
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
        execution_device: object | None,
    ) -> PlanMemory:
        with self._lock:
            self._require_open()
            if self._active_plans != 0 or self._planning:
                raise RuntimeConfigurationError(
                    "this Runtime already has an active callable or in-progress "
                    "plan; close it before creating another plan"
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
            resolved_execution = _resolve_execution_budget(
                execution_budget, execution_pool
            )
            resolved_spill = _resolve_budget(spill_budget, spill_pool, "spill_budget")
            transfers = self._read_transfer_capabilities()
            fetch = transfers.route(spill, execution)
            evict = transfers.route(execution, spill)
            if not fetch.available or not fetch.calibrated:
                raise RuntimeConfigurationError(
                    f"route {spill!r} -> {execution!r} is not calibrated"
                )
            if not evict.available or not evict.calibrated:
                raise RuntimeConfigurationError(
                    f"route {execution!r} -> {spill!r} is not calibrated"
                )
            memory = PlanMemory(
                runtime=self,
                installed=self._installed,
                execution=execution_pool,
                spill=spill_pool,
                execution_budget=resolved_execution,
                spill_budget=resolved_spill,
                execution_device=resolved_device,
                transfers=transfers,
            )
            self._planning = True
            return memory

    def _adopt_plan(self) -> None:
        with self._lock:
            self._require_open()
            if not self._planning or self._active_plans != 0:
                raise RuntimeConfigurationError(
                    "Runtime has no exclusive in-progress plan to adopt"
                )
            self._planning = False
            self._active_plans = 1

    def _release_plan(self) -> None:
        with self._lock:
            if self._active_plans != 1 or self._planning:
                raise RuntimeError("Runtime plan ownership underflow")
            try:
                self._clear_execution_plan()
            except BaseException as error:
                self._unusable_reason = f"execution-plan teardown failed: {error}"
                raise
            finally:
                self._active_plans = 0

    def _abort_plan(self) -> None:
        """Release cold-path execution records after a failed planning call."""

        with self._lock:
            if not self._planning or self._active_plans != 0:
                raise RuntimeError("Runtime planning ownership underflow")
            try:
                self._clear_execution_plan()
            except BaseException as error:
                self._unusable_reason = f"execution-plan rollback failed: {error}"
                raise
            finally:
                self._planning = False

    def _clear_execution_plan(self) -> None:
        status = int(self._installed.library.shadowspill_pytorch_clear_execution_plan())
        if status != 0:
            raise RuntimeConfigurationError(
                f"execution-plan teardown failed with status {status}"
            )

    def _translate_allocator_failure(
        self,
        cause: BaseException,
        *,
        operation: str,
        task: ExecutionTaskIdentity | None = None,
    ) -> RuntimeExecutionError | None:
        """Translate allocator and admitted-runtime contract failures."""

        if isinstance(cause, RuntimeExecutionError):
            diagnostics = cause.diagnostics
            if diagnostics is not None:
                self._record_failure(diagnostics)
            return cause if diagnostics is not None and (
                diagnostics.is_allocator_oom
                or diagnostics.is_shadowspill_contract_failure
            ) else None
        diagnostics = read_allocator_failure(
            self._installed.library, operation, task=task
        )
        if diagnostics is None:
            return None
        self._record_failure(diagnostics)
        if diagnostics.is_allocator_oom:
            return allocator_oom_error(diagnostics)
        if diagnostics.is_shadowspill_contract_failure:
            return generic_runtime_error(diagnostics)
        return None

    def _prepare_failure_cleanup(self, error: BaseException) -> None:
        """Quiesce compute and recover a no-progress latch before teardown."""

        if isinstance(error, RuntimeExecutionError) and not error._begin_cleanup():
            return
        diagnostics = (
            error.diagnostics if isinstance(error, RuntimeExecutionError) else None
        )
        if diagnostics is not None:
            self._record_failure(diagnostics)
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
        large_copy_bytes: int = 64 << 20,
        warmup_copies: int = 2,
        measured_copies: int = 5,
    ) -> TransferCapabilities:
        with self._lock:
            self._require_open()
            if self._active_plans != 0 or self._planning:
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
                abi_version=TRANSFER_PROFILE_ABI_VERSION,
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
        native = (NativeTransferProfile * count.value)()
        status = int(
            self._installed.library.shadowspill_pytorch_transfer_profiles(
                native,
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
            )
            for item in native
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


def _validate_pool_configs(
    pools: Mapping[str, MemoryPoolConfig],
) -> dict[str, MemoryPoolConfig]:
    if not isinstance(pools, Mapping):
        raise TypeError("pools must be a mapping from names to pool configurations")
    normalized = dict(pools)
    if len(normalized) != 2:
        raise RuntimeConfigurationError(
            "the initial device/pinned-host release requires exactly two pools"
        )
    for name, config in normalized.items():
        if not isinstance(name, str) or not name or not name.isidentifier():
            raise RuntimeConfigurationError(
                f"pool name {name!r} must be a non-empty identifier"
            )
        if not isinstance(config, (DevicePool, PinnedHostPool)):
            raise TypeError(f"unsupported pool configuration for {name!r}")
    if (
        sum(isinstance(value, DevicePool) for value in normalized.values()) != 1
        or sum(isinstance(value, PinnedHostPool) for value in normalized.values()) != 1
    ):
        raise RuntimeConfigurationError(
            "the initial release requires one device pool and one pinned-host pool"
        )
    return normalized


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

    Runtime initialization subtracts the one-time accelerator context and
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
            resolved_device = torch.device("cuda", value)
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
        if resolved_device.type != "cuda":
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
    "TransferCapabilities",
    "TransferProfile",
]
