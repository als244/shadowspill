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

from shadowspill.pytorch.runtime_adapter.abi import (
    INITIAL_ACTIONS_TASK_ID,
    PROFILING_SCOPE_BASE,
    RUNTIME_OBJECT_SCOPE_ID,
    LiveAllocation,
    runtime_library,
)
from shadowspill.pytorch.runtime_adapter.failures import RuntimeExecutionError

from .configuration import RuntimeConfigurationError

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
    (1 << 61, "materialization task"),
    (INITIAL_ACTIONS_TASK_ID, "initial actions"),
)


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


__all__ = ["PoolAllocation", "describe_live_allocations", "live_allocations"]
