"""Explicit process-lifetime runtime initialization for the PyTorch frontend."""

from __future__ import annotations

import ctypes
import gc
import hashlib
import json
import threading
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from types import FrameType, MappingProxyType, ModuleType
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
from shadowspill.pytorch.accelerator import accelerator_device, is_accelerator
from shadowspill.pytorch.runtime_adapter.abi import (
    INITIAL_ACTIONS_TASK_ID,
    PROFILING_SCOPE_BASE,
    RUNTIME_OBJECT_SCOPE_ID,
    Allocation,
    LiveAllocation,
    MemoryPoolStatistics,
    ObjectDescription,
    PlanDescription,
    TransferCalibrationConfig,
    TransferRouteKey,
    runtime_library,
)
from shadowspill.pytorch.runtime_adapter.abi import (
    TransferProfile as RuntimeTransferProfile,
)
from shadowspill.pytorch.runtime_adapter.allocator import (
    DEFAULT_BACKGROUND_WINDOW_BYTES,
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
    #: This plan's identity for the life of the runtime. Every lease its tasks
    #: make records it, and the allocation scopes opened for it name it too.
    plan_id: int


_runtime_lock = threading.Lock()
_active_runtime: Runtime | None = None


#: `SHADOWSPILL_RUNTIME_NO_ID`: the scope field when there was no scope.
_NO_SCOPE = (1 << 64) - 1

#: Scope ids above an execution task's range name a synthetic scope rather than
#: a task of the program. Highest base first, so the search stops at the right
#: one. `INITIAL_ACTIONS_TASK_ID` is a single id, not a base.
_SYNTHETIC_SCOPES: tuple[tuple[int, str], ...] = (
    (RUNTIME_OBJECT_SCOPE_ID, "runtime object"),
    (PROFILING_SCOPE_BASE, "profiling scope"),
    (1 << 61, "materialization task"),
    (INITIAL_ACTIONS_TASK_ID, "initial actions"),
)


def _references_in(container: object) -> tuple[tuple[str, int], ...]:
    """Where one container refers to other objects, as (where, identity) pairs.

    Only the shapes a reference is actually held in: a mapping's values, a
    sequence's items, a set's members, and the slots of an object carrying no
    `__dict__`. An attribute on an ordinary object is not one of them -- it is
    held in that object's `__dict__`, which is what the collector reports.
    """

    if isinstance(container, dict):
        return tuple(
            (
                f".{key}"
                if isinstance(key, str) and key.isidentifier()
                else f"[{key!r}]",
                id(value),
            )
            for key, value in tuple(container.items())
        )
    if isinstance(container, (list, tuple)):
        return tuple((f"[{index}]", id(value)) for index, value in enumerate(container))
    if isinstance(container, (set, frozenset)):
        return tuple(("{...}", id(value)) for value in container)
    slots = getattr(type(container), "__slots__", ())
    named = (slots,) if isinstance(slots, str) else slots
    found: list[tuple[str, int]] = []
    for name in named:
        try:
            found.append((f".{name}", id(getattr(container, name))))
        except AttributeError:
            continue
    return tuple(found)


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


@dataclass(frozen=True, slots=True)
class PoolAllocation:
    """One allocation a pool currently holds.

    `offset` is a byte offset into the pool's arena, and is what explains a
    contiguous-range refusal. `origin_task_id` is the scope that made it, so an
    allocation still held after its scope ended can be attributed rather than
    merely counted; `logical_freed` marks one the frontend has already given up,
    which is awaiting retirement rather than outliving anything.
    """

    allocation_id: int
    offset: int
    charged_bytes: int
    requested_bytes: int
    #: The plan whose scope made it, or `None` when no plan did. A pool outlives
    #: any one plan, so this is what separates a range an earlier plan left
    #: behind from one the current plan made -- task ids cannot, being
    #: plan-local and so shared between plans.
    origin_plan_id: int | None
    #: The scope that made it, or `None` when it was allocated outside any task
    #: or allocation scope -- a provider's retained state, say, which belongs to
    #: the process rather than to any scope.
    origin_task_id: int | None
    origin_task_invocation: int
    origin_task_allocation_ordinal: int
    #: The object this range is bound to, or `None` when it is bound to none.
    #: An unbound range is workspace: it holds no value the program named.
    object_id: int | None
    references: int
    scratch: bool
    plan_owned: bool
    #: Whether the plan ever owned it. Set without `plan_owned` means the range
    #: was promoted out to a named owner and is no longer the plan's to release.
    ever_plan_owned: bool
    logical_freed: bool
    framework_free_seen: bool

    @property
    def unclaimed_scope_workspace(self) -> bool:
        """Workspace a scope made that nobody took ownership of.

        Not every range a scope made is the scope's to reclaim. The contract has
        three ends, and two of them transfer ownership: a range promoted out to a
        named owner -- an output the caller holds, an object registered for
        sharing -- belongs to that owner now, and a range the plan placed is
        released by the plan's own teardown. One already logically freed is
        awaiting retirement rather than surviving.

        What is left is workspace nobody named, which is the thing that outlives a
        scope by accident, and the only thing a closing plan may take back.
        """

        return (
            self.origin_task_id is not None
            and self.object_id is None
            and not self.plan_owned
            and not self.ever_plan_owned
            and not self.logical_freed
        )

    @property
    def role(self) -> str:
        """What the range is for, as far as the runtime alone can tell.

        The runtime knows whether a range is bound to an object the program
        named, whether a plan placed it, and which scope made it -- enough to
        separate a planned value from workspace, and workspace made inside a
        profiling probe from workspace made by a task. It does not know whether
        a planned object is a parameter or an activation: that is the program's
        to say, and a caller holding one resolves `object_id` against it.
        """

        if self.origin_task_id == RUNTIME_OBJECT_SCOPE_ID:
            # Backs an object the program named, but no plan owns it and any
            # number may bind it, so it is neither planned nor unscoped.
            return "runtime-object"
        if self.plan_owned or self.object_id is not None:
            return "planned"
        if self.origin_task_id is None:
            return "unscoped"
        if self.origin_task_id >= PROFILING_SCOPE_BASE:
            # Workspace a probe left behind is provider or custom-operation
            # state the library keeps for itself, not anything the task owns.
            return "op-internal"
        return "workspace"

    @property
    def origin(self) -> str:
        """The scope that made it, named rather than numbered.

        Synthetic scopes carry ids far above any task of the program, so a bare
        number reads as noise. Naming them is what lets a reader tell profiling
        from execution without knowing the bases.

        The plan comes first, because a task id is plan-local: task 1112 exists
        in every plan, so the pair is the identity and neither half is one on
        its own.
        """

        plan = "" if self.origin_plan_id is None else f"plan {self.origin_plan_id} "
        if self.origin_task_id is None:
            return f"{plan}no scope" if plan else "no scope"
        scope = f"task {self.origin_task_id}"
        for base, name in _SYNTHETIC_SCOPES:
            if self.origin_task_id >= base:
                offset = self.origin_task_id - base
                scope = f"{name} #{offset}" if offset else name
                break
        return f"{plan}{scope}"


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
        background_transfer_window_bytes: int = DEFAULT_BACKGROUND_WINDOW_BYTES,
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
                background_transfer_window_bytes=background_transfer_window_bytes,
                backend=backend,
            )
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

    def live_allocations(self, pool: str = "execution") -> tuple[PoolAllocation, ...]:
        """Every allocation the named pool currently holds, in pool order.

        Statistics answer how many allocations are live; this answers which,
        and where. Position is what explains a layout refused for want of a
        contiguous range, because a small allocation in the wrong place costs
        the largest free range while leaving the free total nearly untouched.

        Each entry names the scope that made it, so an allocation that outlived
        its scope can be attributed rather than merely counted.
        """

        registered = self._pools.get(pool)
        if registered is None:
            raise KeyError(f"no pool named {pool!r}")
        library = runtime_library()
        count = ctypes.c_uint64()
        status = int(
            library.shadowspill_memory_pool_live_allocations(
                self._runtime_handle, registered.pool_id, None, 0, ctypes.byref(count)
            )
        )
        if status != 0:
            raise RuntimeConfigurationError(
                f"live allocation query failed with status {status}"
            )
        if count.value == 0:
            return ()
        buffer = (LiveAllocation * count.value)()
        copied = ctypes.c_uint64()
        status = int(
            library.shadowspill_memory_pool_live_allocations(
                self._runtime_handle,
                registered.pool_id,
                buffer,
                count.value,
                ctypes.byref(copied),
            )
        )
        if status != 0:
            raise RuntimeConfigurationError(
                f"live allocation query failed with status {status}"
            )
        entries = tuple(
            PoolAllocation(
                allocation_id=int(item.allocation_id),
                offset=int(item.offset),
                charged_bytes=int(item.charged_bytes),
                requested_bytes=int(item.requested_bytes),
                origin_plan_id=int(item.origin_plan_id) or None,
                origin_task_id=(
                    None
                    if int(item.origin_task_id) == _NO_SCOPE
                    else int(item.origin_task_id)
                ),
                origin_task_invocation=int(item.origin_task_invocation),
                origin_task_allocation_ordinal=int(item.origin_task_allocation_ordinal),
                object_id=(
                    None if int(item.object_id) == _NO_SCOPE else int(item.object_id)
                ),
                references=int(item.references),
                scratch=bool(item.scratch),
                plan_owned=bool(item.plan_owned),
                ever_plan_owned=bool(item.ever_plan_owned),
                logical_freed=bool(item.logical_freed),
                framework_free_seen=bool(item.framework_free_seen),
            )
            for item in buffer[: min(copied.value, count.value)]
        )
        return tuple(sorted(entries, key=lambda item: item.offset))

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

    def describe_live_allocations(self, pool: str = "execution") -> tuple[str, ...]:
        """Every range the pool holds, one line each, in pool order.

        The whole enumeration rather than the part a caller is entitled to act on:
        what is there, which plan and scope made each range, what it is bound to,
        and the flags that say who owns it now. Reading a pool's occupancy is the
        question this answers; deciding what to do about it is not.
        """

        # A plan's state is asked once per plan, not once per range: a pool holds
        # many ranges and usually one or two plans' worth.
        held = self.live_allocations(pool)
        states: dict[int, str] = {}
        for item in held:
            if item.origin_plan_id is not None and item.origin_plan_id not in states:
                try:
                    state = self.plan_state(item.origin_plan_id)
                except RuntimeExecutionError:  # a report must not fail
                    continue
                states[item.origin_plan_id] = (
                    "" if state is PlanState.LIVE else f" ({state.name.lower()})"
                )
        return tuple(
            f"id={item.allocation_id} offset={item.offset}"
            f" bytes={item.charged_bytes}"
            f" requested={item.requested_bytes} {item.role} from {item.origin}"
            f"{states.get(item.origin_plan_id or 0, '')}"
            f" object={'-' if item.object_id is None else item.object_id}"
            f" refs={item.references}"
            f"{' scratch' if item.scratch else ''}"
            f"{' planned' if item.plan_owned else ''}"
            f"{' promoted' if item.ever_plan_owned and not item.plan_owned else ''}"
            f"{' freed-pending-retirement' if item.logical_freed else ''}"
            f"{' framework-freed' if item.framework_free_seen else ''}"
            f"{' unclaimed' if item.unclaimed_scope_workspace else ''}"
            for item in held
        )

    def plan_scoped_residue(self, plan_handle: int) -> tuple[str, ...]:
        """Which of one plan's own ranges are still occupied, and by what.

        A plan that closes owes the pool every range its tasks and its profiling
        probes made, so anything still here is residue. Ranges carrying no plan --
        a provider taking its own workspace between tasks -- belong to no plan and
        are not counted.

        This reports; it does not release. A reference can only be dropped by
        whoever holds it, in that component's own teardown, and a pool of bytes is
        the wrong place to reach into the object graph and decide on someone
        else's behalf. What this gives the owner is the fact it needs: the range,
        the scope that made it, the object occupying it, and where that object is
        referenced from.

        Returns one description per surviving range.
        """

        plan_id = int(
            runtime_library().shadowspill_plan_id(ctypes.c_size_t(plan_handle))
        )
        if plan_id == 0:
            return ()
        remaining = tuple(
            item
            for item in self.live_allocations()
            if item.origin_plan_id == plan_id and item.unclaimed_scope_workspace
        )
        if not remaining:
            return ()
        survivors = self.occupants(remaining)
        occupying = [item for group in survivors.values() for item in group]
        retainers = self.retainers(
            occupying, ignore=(survivors, occupying, *survivors.values())
        )
        described: list[str] = []
        for item in remaining:
            objects = survivors.get(item.allocation_id, ())
            if not objects:
                where = "occupied by no frontend object"
            else:
                where = "occupied by " + ", ".join(
                    f"{type(obj).__name__}{tuple(getattr(obj, 'shape', ()))}"
                    " referenced from "
                    + (", ".join(retainers.get(id(obj), ())[:2]) or "nothing nameable")
                    for obj in objects[:2]
                )
            described.append(
                f"offset={item.offset} bytes={item.charged_bytes}"
                f" {item.role} from {item.origin}, {where}"
            )
        return tuple(described)

    def force_release_plan_scope(self, plan_handle: int) -> tuple[int, int]:
        """Take back everything one plan's own scopes allocated. Returns
        (storages detached, leases reclaimed).

        The forcing path, for a plan that is closing: its ranges go whether or not
        the framework has released them. Two halves, because two layers own the
        two facts.

        The frontend half detaches the storages. A tensor occupying one of these
        ranges stops referencing those bytes, so the Python object no longer points
        at memory about to be reclaimed, and a later read raises on an empty
        storage rather than reading whatever now lives there.

        The runtime half reclaims the leases, in bytes, which is what it owns. A
        lease the framework has not freed keeps its pointer indexed, so the free
        that eventually arrives still resolves.

        Ranges carrying no plan -- a provider's own workspace -- are not touched by
        either half.
        """

        plan_id = int(
            runtime_library().shadowspill_plan_id(ctypes.c_size_t(plan_handle))
        )
        if plan_id == 0:
            return (0, 0)
        held = tuple(
            item
            for item in self.live_allocations()
            if item.origin_plan_id == plan_id and item.unclaimed_scope_workspace
        )
        detached = 0
        if held:
            storages = [
                item
                for group in self.occupants(held).values()
                for item in group
                if isinstance(item, torch.Tensor)
                and item.untyped_storage().data_ptr() != 0
            ]
            if storages:
                torch.ops.shadowspill._dematerialize_storages(storages)
                detached = len(storages)
                storages.clear()
        reclaimed = ctypes.c_uint64()
        status = int(
            runtime_library().shadowspill_plan_reclaim_scoped_leases(
                ctypes.c_size_t(plan_handle), ctypes.byref(reclaimed)
            )
        )
        if status != 0:
            raise RuntimeExecutionError(
                f"failed to reclaim the ranges plan {plan_id} allocated:"
                f" status={status}"
            )
        return (detached, int(reclaimed.value))

    def occupants(
        self, allocations: Sequence[PoolAllocation]
    ) -> dict[int, tuple[object, ...]]:
        """The frontend objects whose storage lies inside each given range.

        Answers "what is this range, in the framework's terms". Every live
        accelerator tensor is mapped back to the allocation that owns its
        address, and matched against the allocations asked about.

        A range with no match is held by something the framework does not own --
        a library's retained state, say -- and that is itself the answer: there
        is no reference for a caller to drop.

        Addresses are compared rather than references kept. A runtime holding a
        reference to a frontend object would either keep it alive, which is
        wrong, or hold it weakly and be unable to act on it, so it holds
        neither.
        """

        wanted = {item.allocation_id for item in allocations}
        found: dict[int, list[object]] = {key: [] for key in wanted}
        library = self._installed.library
        record = Allocation()
        # Walking every object touches deprecated framework attributes whose
        # getters warn; the warning belongs to the object being looked at, not
        # to this query.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            candidates = [
                item for item in gc.get_objects() if isinstance(item, torch.Tensor)
            ]
        for candidate in candidates:
            try:
                if candidate.device.type in ("cpu", "meta"):
                    continue
                if candidate.is_meta or type(candidate).__name__ == "FakeTensor":
                    # Export leaves these behind. They report a device and have no
                    # storage, so asking for a data pointer warns and tells us
                    # nothing: one cannot occupy a range in a pool.
                    continue
                address = candidate.untyped_storage().data_ptr()
            except Exception:
                continue
            if not address:
                continue
            status = int(
                library.shadowspill_pytorch_allocation_for_pointer(
                    address, ctypes.byref(record)
                )
            )
            if status == 0 and int(record.allocation_id) in found:
                found[int(record.allocation_id)].append(candidate)
        return {key: tuple(value) for key, value in found.items()}

    def retainers(
        self, held: Sequence[object], *, ignore: Sequence[object] = ()
    ) -> dict[int, tuple[str, ...]]:
        """What keeps each given object alive, named where the reference is held.

        `occupants` answers which frontend object occupies a range; this answers
        why that object is still reachable, which is what a caller needs to
        release it. A reference can only be dropped where it is held, so what is
        named is the container -- an attribute on a class, an entry in a module's
        globals -- and not the object again. An object no container names is held
        by a library, and that is itself the answer: Python has nothing to drop.

        Keyed by `id`, so a caller reads answers back with the objects it passed
        in. Stack frames are excluded, this query's own among them; every other
        referrer is reported, including the caller's own container.

        Descriptions, never references, for the reason `occupants` keeps none:
        retaining what this walks would extend the lifetime of exactly the
        objects under investigation.
        """

        if not held:
            return {}
        found: dict[int, list[str]] = {id(item): [] for item in held}
        # This query's own containers refer to the objects it is asking about: the
        # sequence passed in, and whatever the caller built it from. Reporting
        # those would describe the question rather than the answer -- a dict keyed
        # by allocation id is `occupants`' own result, not a holder worth naming.
        mine = {id(held), *(id(item) for item in ignore)}
        referrers = [
            item
            for item in gc.get_referrers(*held)
            if not isinstance(item, FrameType) and id(item) not in mine
        ]
        # An attribute arrives as the owner's `__dict__`, which names neither the
        # owner nor the attribute. One further hop resolves it, batched, because
        # each hop walks the whole heap.
        owners: dict[int, str] = {}
        mappings = [item for item in referrers if isinstance(item, dict)]
        if mappings:
            for owner in gc.get_referrers(*mappings):
                if isinstance(owner, (FrameType, dict)):
                    continue
                mapping = getattr(owner, "__dict__", None)
                if mapping is None:
                    continue
                owners[id(mapping)] = (
                    owner.__name__
                    if isinstance(owner, ModuleType)
                    else type(owner).__name__
                )
        # A list or tuple carries no name of its own, so it is named by whatever
        # holds it: one more hop turns `list[2]` into `Owner.attribute[2]`.

        # Named top down, because a container's label depends on its holder's:
        # naming the container first and the dict afterwards cannot improve on
        # `dict[49362]`, which says nothing about whose dict it is.
        def _named(candidate: object) -> str | None:
            """What holds this, as a module or a type, if anything names it."""

            for holder in gc.get_referrers(candidate):
                if isinstance(holder, FrameType):
                    continue
                if isinstance(holder, ModuleType):
                    return holder.__name__
                mapping = getattr(holder, "__dict__", None)
                if isinstance(mapping, dict) and any(
                    value is candidate for value in tuple(mapping.values())
                ):
                    return type(holder).__name__
                if isinstance(holder, dict):
                    for owner in gc.get_referrers(holder):
                        if isinstance(owner, ModuleType):
                            return f"{owner.__name__}(globals)"
                        if getattr(owner, "__dict__", None) is holder:
                            return type(owner).__name__
                        # A decorator's cache lives in a closure cell, which is
                        # reached by neither a module nor an instance dict. The
                        # function that closed over it is the name worth having.
                        if type(owner).__name__ == "cell":
                            for closed in gc.get_referrers(owner):
                                name = getattr(closed, "__qualname__", None)
                                if name is not None:
                                    module = getattr(closed, "__module__", "?")
                                    return f"{module}.{name}"
            return None

        anonymous = [item for item in referrers if isinstance(item, (list, tuple, set))]
        for container in anonymous:
            for holder in gc.get_referrers(container):
                if isinstance(holder, FrameType) or not isinstance(holder, dict):
                    continue
                keys = [
                    key for key, value in tuple(holder.items()) if value is container
                ]
                if not keys:
                    continue
                whose = owners.get(id(holder)) or _named(holder) or "dict"
                owners[id(container)] = f"{whose}[{keys[0]}]"
                break

        for referrer in referrers:
            label = owners.get(id(referrer), type(referrer).__name__)
            try:
                references = _references_in(referrer)
            except Exception:  # a diagnostic must not fail on what it inspects
                continue
            for where, target in references:
                if target in found:
                    found[target].append(f"{label}{where}")
        return {key: tuple(dict.fromkeys(value)) for key, value in found.items()}

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

    def _register_object(
        self,
        object_id: int,
        size_bytes: int,
        *,
        pool_id: int,
        retain_spill_copy: bool,
        initially_resident: bool,
        source_address: int = 0,
    ) -> int:
        """Register one runtime object, and populate it when a source is given.

        Two neutral calls, made here; the bridge and the state module both
        register through this.
        """

        description = ObjectDescription(
            object_id=object_id,
            size_bytes=size_bytes,
            initial_pool_id=pool_id,
            retain_spill_copy=int(retain_spill_copy),
            initially_resident=int(initially_resident),
        )
        status = int(
            runtime_library().shadowspill_register_object(
                self._runtime_handle, ctypes.byref(description)
            )
        )
        if status != 0 or source_address == 0:
            return status
        return int(
            runtime_library().shadowspill_write_object(
                self._runtime_handle, object_id, pool_id, source_address, size_bytes
            )
        )

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
            resolved_execution = _resolve_execution_budget(
                execution_budget, execution_pool
            )
            resolved_spill = _resolve_budget(spill_budget, spill_pool, "spill_budget")
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
            plan_id = self.next_plan_id()
            status = int(
                runtime_library().shadowspill_plan_create(
                    self._runtime_handle,
                    ctypes.byref(
                        PlanDescription(
                            plan_id=plan_id,
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
                plan_id=plan_id,
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
                runtime_library().shadowspill_runtime_calibrate_transfer_capabilities(
                    self._runtime_handle, ctypes.byref(config), keys, count
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
            runtime_library().shadowspill_runtime_transfer_profiles(
                self._runtime_handle,
                None,
                0,
                ctypes.byref(count),
                ctypes.byref(generation),
            )
        )
        if status != 0:
            raise RuntimeConfigurationError(
                f"transfer-profile size query failed with status {status}"
            )
        handle = (RuntimeTransferProfile * count.value)()
        status = int(
            runtime_library().shadowspill_runtime_transfer_profiles(
                self._runtime_handle,
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


def planned_execution_budget(pool: MemoryPool, execution_budget: int | None) -> int:
    """The execution budget a plan against `pool` will actually be given.

    Planning resolves a requested budget against the pool it will run in, so the
    figure a plan is priced against is not always the figure that was asked for. A
    caller reporting what it planned against asks here rather than repeating that
    arithmetic.

    Asking is also the only way to see a reduction *before* planning. A budget equal
    to the pool's physical cap is the spelling for "the whole pool" -- the same thing
    `None` means -- so it resolves to the derived capacity, which is smaller than the
    cap by whatever runtime initialization carved out of it. A caller that meant "this
    exact budget" gets the derived capacity instead, and nothing else in the plan says
    so.
    """

    return _resolve_execution_budget(execution_budget, pool)


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
    "PoolAllocation",
    "Runtime",
    "RuntimeConfigurationError",
    "RuntimeRoute",
    "TransferCapabilities",
    "TransferProfile",
    "planned_execution_budget",
]
