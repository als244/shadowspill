"""What a closing plan leaves behind, and the one path that takes it back.

Nothing a plan's own scopes allocated outlives the plan. `plan_scoped_residue`
names what still does -- the range, the scope that made it, the object occupying
it and where that object is referenced from -- and `force_release_plan_scope`
reclaims it; `reclaim_plan_scoped_residue` is the two together as a closing
callable runs them, reporting rather than raising.
"""

from __future__ import annotations

import ctypes
import os
import warnings
from typing import TYPE_CHECKING

import torch

from shadowspill.pytorch.runtime_adapter.abi import runtime_library
from shadowspill.pytorch.runtime_adapter.failures import RuntimeExecutionError

from .occupancy import PoolAllocation, describe_live_allocations, live_allocations
from .retainers import occupants, retainers

if TYPE_CHECKING:
    from .core import Runtime


def _unclaimed_by_plan(
    runtime: Runtime, plan_handle: int
) -> tuple[PoolAllocation, ...]:
    """The execution pool's unclaimed scope workspace that one plan's scopes made."""

    plan_id = int(runtime_library().shadowspill_plan_id(ctypes.c_size_t(plan_handle)))
    if plan_id == 0:
        return ()
    return tuple(
        item
        for item in live_allocations(runtime)
        if item.origin_plan_id == plan_id and item.unclaimed_scope_workspace
    )


def plan_scoped_residue(runtime: Runtime, plan_handle: int) -> tuple[str, ...]:
    """Which of one plan's own ranges are still occupied, and by what.

    A plan that closes owes the pool every range its tasks and its profiling
    probes made, so anything still here is residue. Ranges carrying no plan -- a
    provider taking its own workspace between tasks -- belong to no plan and are
    not counted.

    This reports; it does not release. A reference can only be dropped by
    whoever holds it, in that component's own teardown, and a pool of bytes is
    the wrong place to reach into the object graph and decide on someone else's
    behalf. What this gives the owner is the fact it needs: the range, the scope
    that made it, the object occupying it, and where that object is referenced
    from.

    Returns one description per surviving range.
    """

    remaining = _unclaimed_by_plan(runtime, plan_handle)
    if not remaining:
        return ()
    survivors = occupants(runtime, remaining)
    occupying = [item for group in survivors.values() for item in group]
    holders = retainers(occupying, ignore=(survivors, occupying, *survivors.values()))
    described: list[str] = []
    for item in remaining:
        objects = survivors.get(item.allocation_id, ())
        if not objects:
            where = "occupied by no frontend object"
        else:
            where = "occupied by " + ", ".join(
                f"{type(obj).__name__}{tuple(getattr(obj, 'shape', ()))}"
                " referenced from "
                + (", ".join(holders.get(id(obj), ())[:2]) or "nothing nameable")
                for obj in objects[:2]
            )
        described.append(
            f"offset={item.offset} bytes={item.charged_bytes}"
            f" {item.role} from {item.origin}, {where}"
        )
    return tuple(described)


def force_release_plan_scope(runtime: Runtime, plan_handle: int) -> tuple[int, int]:
    """Take back everything one plan's own scopes allocated. Returns
    (storages detached, leases reclaimed).

    The forcing path, for a plan that is closing: its ranges go whether or not
    the framework has released them. Two halves, because two layers own the two
    facts.

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

    held = _unclaimed_by_plan(runtime, plan_handle)
    if not held:
        return (0, 0)
    detached = 0
    storages = [
        item
        for group in occupants(runtime, held).values()
        for item in group
        if isinstance(item, torch.Tensor) and item.untyped_storage().data_ptr() != 0
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
        plan_id = int(
            runtime_library().shadowspill_plan_id(ctypes.c_size_t(plan_handle))
        )
        raise RuntimeExecutionError(
            f"failed to reclaim the ranges plan {plan_id} allocated: status={status}"
        )
    return (detached, int(reclaimed.value))


def reclaim_plan_scoped_residue(runtime: Runtime, plan_handle: int) -> None:
    """Reclaim what one plan's own scopes allocated and nobody released, as a
    closing callable does.

    The contract is that nothing a plan's own scopes allocated outlives the
    plan. Anything that does is named here, with the object occupying it and
    where that object is referenced from, so the component holding it can
    release it in its own teardown. Reported rather than raised: a kernel is
    allowed to keep state between tasks, and what is wanted at close is
    visibility rather than a failure during teardown. Setting
    `SHADOWSPILL_REPORT_LIVE_ALLOCATIONS` adds a second warning describing the
    whole execution pool before anything is released, so the residue can be
    read against the rest of the pool rather than on its own.
    """

    if os.environ.get("SHADOWSPILL_REPORT_LIVE_ALLOCATIONS"):
        held = describe_live_allocations(runtime)
        warnings.warn(
            f"the execution pool holds {len(held)} range(s) as this callable"
            " closes:\n  " + "\n  ".join(held),
            RuntimeWarning,
            stacklevel=3,
        )
    survivors = plan_scoped_residue(runtime, plan_handle)
    if not survivors:
        return
    detached, reclaimed = force_release_plan_scope(runtime, plan_handle)
    warnings.warn(
        f"closing this callable reclaimed {reclaimed} allocation(s) that its"
        f" own scopes made and nothing released, detaching {detached}"
        " storage(s) first:\n  " + "\n  ".join(survivors),
        RuntimeWarning,
        stacklevel=3,
    )


__all__ = [
    "force_release_plan_scope",
    "plan_scoped_residue",
    "reclaim_plan_scoped_residue",
]
