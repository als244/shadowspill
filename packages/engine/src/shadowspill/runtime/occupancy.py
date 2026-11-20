"""What a pool holds, range by range.

Statistics say how many ranges a pool holds. These say which and where: every
live range with the plan and scope that made it, what it is bound to, and who
owns it now. Functions over a runtime rather than methods on it: reading
occupancy needs the runtime's handle and pool registry and nothing of its
state machine.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .abi import (
    INITIAL_ACTIONS_TASK_ID,
    PROFILING_SCOPE_BASE,
    RUNTIME_OBJECT_SCOPE_ID,
    Allocation,
    LiveAllocation,
    PlanSliceRecord,
    runtime_library,
)
from .configuration import RuntimeConfigurationError
from .failures import RuntimeExecutionError

if TYPE_CHECKING:
    from .core import Runtime

#: `SHADOWSPILL_RUNTIME_NO_ID`: the scope field when there was no scope.
_NO_SCOPE = (1 << 64) - 1

#: Scope ids above an execution task's range name a synthetic scope rather than
#: a task of the program. Highest base first, so the search stops at the right
#: one. `INITIAL_ACTIONS_TASK_ID` is a single id, not a base.
_SYNTHETIC_SCOPES: tuple[tuple[int, str], ...] = (
    (RUNTIME_OBJECT_SCOPE_ID, "runtime object"),
    (PROFILING_SCOPE_BASE, "profiling scope"),
    (INITIAL_ACTIONS_TASK_ID, "initial actions"),
)


@dataclass(frozen=True, slots=True)
class PoolAllocation:
    """One allocation a pool holds, with the scope that made it and its owner.

    ``live_allocations`` returns one for each allocation a pool holds, in
    pool order, and `RuntimeFrontend.occupants` receives them to say which
    of the frontend's objects stand on each.

    Attributes
    ----------
    allocation_id : int
        The runtime's id for the allocation.
    offset : int
        Its byte offset in the pool.
    charged_bytes : int
        Bytes of the pool it holds.
    requested_bytes : int
        Bytes its allocating call asked for.
    origin_plan_id : int or None
        The plan whose scope made it; None when no plan was named.
    origin_task_id : int or None
        The scope that made it; None when it was made outside any scope.
        Task ids are plan-local. Ids above a program's tasks name synthetic
        scopes, which `PoolAllocation.origin` names.
    origin_task_invocation : int
        Which invocation of its task made it, counting from one; zero when
        no running task allocated it.
    origin_task_allocation_ordinal : int
        The declared allocation step of its task it matched; ``2**64 - 1``
        when it matched none or no running task allocated it.
    object_id : int or None
        The object bound to it; None for workspace, which holds no value the
        program named.
    references : int
        References the runtime holds to it: one of its own, and one for each
        retirement, release or handoff still pending.
    scratch : bool
        Whether its task allocated it outside the task's declared allocation
        steps.
    plan_owned : bool
        Whether a plan owns it now, as an object's storage that the plan
        releases.
    ever_plan_owned : bool
        Whether a plan has ever owned it. True with ``plan_owned`` false means
        it passed to another owner, such as an output the caller holds.
    logical_freed : bool
        Whether it has been freed and waits only for the work still using it
        to complete.
    framework_free_seen : bool
        For a range a plan has owned, whether the frontend has freed it. A
        plan releases the ranges it owns regardless.
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
        """Whether it is workspace a scope made that nothing took over.

        True when a scope made it, no object is bound to it, no plan has
        ever owned it and it has not been freed. A closing plan reclaims the
        ones its own scopes made.
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
        """What the range is for, as far as the runtime can tell.

        ``"runtime-object"`` for the storage of an object the runtime owns,
        which no plan owns; ``"planned"`` when a plan owns it or an object is
        bound to it; ``"unscoped"`` when no scope made it; ``"op-internal"``
        for workspace a profiling scope made; ``"workspace"`` for workspace a
        task made.
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
        """The plan and scope that made it, as text such as ``"plan 3 task 12"``.

        A synthetic scope is named rather than numbered, as in
        ``"profiling scope #2"``, ``"initial actions"`` or
        ``"runtime object"``. Without a plan the plan part is left out;
        without a scope the scope part reads ``"no scope"``.
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


def allocation_id_for_address(runtime: Runtime, address: int) -> int | None:
    """Which allocation owns ``address``, or None if the runtime owns none.

    The one thing a frontend cannot answer about its own objects, and the reason
    `occupants` takes this as an argument: the frontend knows where its objects
    are, the runtime knows whose bytes those are.
    """

    record = Allocation()
    status = int(
        runtime._installed.library.shadowspill_pytorch_allocation_for_pointer(
            address, ctypes.byref(record)
        )
    )
    if status != 0:
        return None
    return int(record.allocation_id)


def live_allocations(
    runtime: Runtime, pool: str = "execution"
) -> tuple[PoolAllocation, ...]:
    """Every allocation the named pool currently holds, in pool order.

    Statistics answer how many allocations are live; this answers which, and
    where. Position is what explains a layout refused for want of a contiguous
    range, because a small allocation in the wrong place costs the largest free
    range while leaving the free total nearly untouched.

    Each entry names the scope that made it, so an allocation that outlived its
    scope can be attributed rather than merely counted.
    """

    registered = runtime._pools.get(pool)
    if registered is None:
        raise KeyError(f"no pool named {pool!r}")
    library = runtime_library()
    count = ctypes.c_uint64()
    status = int(
        library.shadowspill_memory_pool_live_allocations(
            runtime._runtime_handle, registered.pool_id, None, 0, ctypes.byref(count)
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
            runtime._runtime_handle,
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


@dataclass(frozen=True, slots=True)
class PlanSlice:
    """One admitted plan's fixed layout in a pool.

    A layout is a reserved range, not an allocation: what the plan's tasks place
    inside it are the allocations, which `live_allocations` lists while they
    live. `offset` and `bytes` are where the plan's own layout lies;
    `slab_plan_id` is the plan that reserved the range it lies in -- the plan
    itself, unless its layout was admitted into another's -- and `slab_bytes`
    that range's size.
    """

    plan_id: int
    offset: int
    bytes: int
    slab_plan_id: int
    slab_bytes: int


def plan_slices(runtime: Runtime, pool: str = "execution") -> tuple[PlanSlice, ...]:
    """The fixed layout of every plan admitted in the named pool, in pool order.

    With `live_allocations`, the pool's whole map: which ranges are held for
    which plans, and which plans share one.
    """

    registered = runtime._pools.get(pool)
    if registered is None:
        raise KeyError(f"no pool named {pool!r}")
    library = runtime_library()
    count = ctypes.c_uint64()
    status = int(
        library.shadowspill_memory_pool_plan_slices(
            runtime._runtime_handle, registered.pool_id, None, 0, ctypes.byref(count)
        )
    )
    if status != 0:
        raise RuntimeConfigurationError(f"plan slice query failed with status {status}")
    if count.value == 0:
        return ()
    buffer = (PlanSliceRecord * count.value)()
    copied = ctypes.c_uint64()
    status = int(
        library.shadowspill_memory_pool_plan_slices(
            runtime._runtime_handle,
            registered.pool_id,
            buffer,
            count.value,
            ctypes.byref(copied),
        )
    )
    if status != 0:
        raise RuntimeConfigurationError(f"plan slice query failed with status {status}")
    return tuple(
        sorted(
            (
                PlanSlice(
                    plan_id=int(item.plan_id),
                    offset=int(item.offset),
                    bytes=int(item.bytes),
                    slab_plan_id=int(item.slab_plan_id),
                    slab_bytes=int(item.slab_bytes),
                )
                for item in buffer[: min(copied.value, count.value)]
            ),
            key=lambda item: (item.offset, item.plan_id),
        )
    )


def describe_live_allocations(
    runtime: Runtime, pool: str = "execution"
) -> tuple[str, ...]:
    """Every range the pool holds, one line each, in pool order.

    The whole enumeration rather than the part a caller is entitled to act on:
    what is there, which plan and scope made each range, what it is bound to,
    and the flags that say who owns it now. Reading a pool's occupancy is the
    question this answers; deciding what to do about it is not.
    """

    from .core import PlanState

    # A plan's state is asked once per plan, not once per range: a pool holds
    # many ranges and usually one or two plans' worth.
    held = live_allocations(runtime, pool)
    states: dict[int, str] = {}
    for item in held:
        if item.origin_plan_id is not None and item.origin_plan_id not in states:
            try:
                state = runtime.plan_state(item.origin_plan_id)
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


__all__ = [
    "PlanSlice",
    "PoolAllocation",
    "allocation_id_for_address",
    "describe_live_allocations",
    "live_allocations",
    "plan_slices",
]
