"""A plan's life on the runtime: begun, adopted by a callable, released.

Planning begins a plan -- pools, routes, budgets and the device resolved, the
C plan created -- and hands back a `PlanMemory`; a planned callable adopts it
and releases it when it closes; a failed planning call aborts it. The runtime
owns at most one plan in progress and any number adopted, and these are the
only functions that move a plan between those states.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..abi import PlanDescription, runtime_library
from ..bootstrap import InstalledRuntime
from ..calibration import read_transfer_capabilities
from ..configuration import (
    RuntimeConfigurationError,
    resolve_budget,
    resolve_dynamic_scratch_reserve,
    resolve_execution_budget,
    resolve_execution_device,
)
from ..topology import MemoryPool, RuntimeRoute, TransferCapabilities

if TYPE_CHECKING:
    from ..core import Runtime


@dataclass(frozen=True, slots=True)
class PlanMemory:
    """Resolved pool roles and capacities consumed by one planning call."""

    runtime: Runtime
    installed: InstalledRuntime
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


def begin_plan(
    runtime: Runtime,
    *,
    execution: str,
    spill: str,
    execution_budget: int | None,
    spill_budget: int | None,
    dynamic_scratch_reserve_bytes: int | None,
    execution_device: object | None,
) -> PlanMemory:
    with runtime._lock:
        runtime._require_open()
        if runtime._planning_plan_handle is not None:
            raise RuntimeConfigurationError(
                "this Runtime already has an in-progress planning call"
            )
        if execution == spill:
            raise RuntimeConfigurationError(
                "execution and spill must select distinct pools"
            )
        try:
            execution_pool = runtime._pools[execution]
            spill_pool = runtime._pools[spill]
        except KeyError as exc:
            raise RuntimeConfigurationError(
                f"unknown runtime pool {exc.args[0]!r}"
            ) from exc
        if execution_pool.kind != "device":
            raise RuntimeConfigurationError(
                "the frontend requires an accelerator execution pool"
            )
        resolved_device = resolve_execution_device(
            runtime.frontend, execution_device, execution_pool
        )
        resolved_execution = resolve_execution_budget(execution_budget, execution_pool)
        resolved_spill = resolve_budget(spill_budget, spill_pool, "spill_budget")
        resolved_scratch = resolve_dynamic_scratch_reserve(
            dynamic_scratch_reserve_bytes,
            execution_budget=resolved_execution,
        )
        transfers = read_transfer_capabilities(
            runtime._runtime_handle, runtime._pool_names
        )
        try:
            fetch_route = runtime._route_by_pair[(spill, execution)]
            evict_route = runtime._route_by_pair[(execution, spill)]
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
        plan_id = runtime.next_plan_id()
        status = int(
            runtime_library().shadowspill_plan_create(
                runtime._runtime_handle,
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
            runtime=runtime,
            installed=runtime._installed,
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
        runtime._planning_plan_handle = plan_handle
        return memory


def adopt_plan(runtime: Runtime, plan_handle: int) -> None:
    with runtime._lock:
        runtime._require_open()
        if runtime._planning_plan_handle != plan_handle:
            raise RuntimeConfigurationError(
                "Runtime does not own this in-progress plan"
            )
        runtime._planning_plan_handle = None
        runtime._active_plan_handles.add(plan_handle)


def release_plan(runtime: Runtime, plan_handle: int) -> None:
    with runtime._lock:
        if plan_handle not in runtime._active_plan_handles:
            raise RuntimeError("Runtime plan ownership underflow")
        try:
            _close_and_destroy_plan(plan_handle)
        except BaseException as error:
            runtime._unusable_reason = f"execution-plan teardown failed: {error}"
            raise
        finally:
            runtime._active_plan_handles.discard(plan_handle)


def abort_plan(runtime: Runtime, plan_handle: int | None = None) -> None:
    """Release cold-path task records after a failed planning call."""

    with runtime._lock:
        target = runtime._planning_plan_handle if plan_handle is None else plan_handle
        if target is None or runtime._planning_plan_handle != target:
            raise RuntimeError("Runtime planning ownership underflow")
        try:
            _close_and_destroy_plan(target)
        except BaseException as error:
            runtime._unusable_reason = f"execution-plan rollback failed: {error}"
            raise
        finally:
            runtime._planning_plan_handle = None


def wait_plan_idle(plan_handle: int) -> None:
    """Block until the plan has no work in flight."""

    status = int(runtime_library().shadowspill_plan_wait_idle(plan_handle))
    if status != 0:
        raise RuntimeError(f"compiled executor did not become idle (status {status})")


def _close_and_destroy_plan(plan_handle: int) -> None:
    status = int(runtime_library().shadowspill_plan_close(plan_handle))
    if status != 0:
        raise RuntimeConfigurationError(
            f"handle plan close failed with status {status}"
        )
    runtime_library().shadowspill_plan_destroy(plan_handle)


__all__ = [
    "PlanMemory",
    "abort_plan",
    "adopt_plan",
    "begin_plan",
    "release_plan",
    "wait_plan_idle",
]
