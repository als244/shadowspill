"""Create, admit, and close worker-local handles for already planned execution."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..abi import PlanDescription, runtime_library
from ..configuration import RuntimeConfigurationError, resolve_execution_device
from ..residue import reclaim_plan_scoped_residue
from ..topology import MemoryPool

if TYPE_CHECKING:
    from ..core import Runtime


@dataclass(frozen=True, slots=True)
class PlanMemory:
    """Worker-local pool bindings and handle; costs and budgets belong to Plan."""

    execution: MemoryPool
    spill: MemoryPool
    execution_device: int
    plan_handle: int


def begin_plan(
    runtime: Runtime,
    *,
    execution: str,
    spill: str,
    execution_device: object | None,
) -> PlanMemory:
    """Bind pool routes without profiling, calibrating, or borrowing other arenas."""

    with runtime._lock:
        runtime._require_open()
        if runtime._pending_plan_handle is not None:
            raise RuntimeConfigurationError("this Runtime already has a plan in setup")
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
        device = resolve_execution_device(
            runtime.frontend, execution_device, execution_pool
        )
        try:
            fetch = runtime._route_by_pair[(spill, execution)]
            evict = runtime._route_by_pair[(execution, spill)]
        except KeyError as exc:
            source, destination = exc.args[0]
            raise RuntimeConfigurationError(
                f"runtime has no directed route {source!r} -> {destination!r}"
            ) from exc
        handle = ctypes.c_size_t()
        status = int(
            runtime_library().shadowspill_plan_create(
                runtime._runtime_handle,
                ctypes.byref(
                    PlanDescription(
                        plan_id=runtime.next_plan_id(),
                        execution_pool_id=execution_pool.pool_id,
                        spill_pool_id=spill_pool.pool_id,
                        fetch_route_id=fetch.route_id,
                        evict_route_id=evict.route_id,
                    )
                ),
                ctypes.byref(handle),
            )
        )
        if status != 0 or handle.value == 0:
            raise RuntimeConfigurationError(
                f"handle plan creation failed: status={status}, "
                f"execution={execution!r}, spill={spill!r}"
            )
        runtime._pending_plan_handle = handle.value
        return PlanMemory(execution_pool, spill_pool, device, handle.value)


def adopt_plan(runtime: Runtime, plan_handle: int) -> None:
    with runtime._lock:
        runtime._require_open()
        if runtime._pending_plan_handle != plan_handle:
            raise RuntimeConfigurationError(
                "Runtime does not own this in-progress plan"
            )
        runtime._pending_plan_handle = None
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
    """Roll back worker setup, reclaiming any allocations owned by this plan."""

    with runtime._lock:
        target = runtime._pending_plan_handle if plan_handle is None else plan_handle
        if target is None or runtime._pending_plan_handle != target:
            raise RuntimeError("Runtime plan setup ownership underflow")
    # Outside the lock: naming what occupies a range
    # walks the frontend's objects. A failure here must not keep the plan open,
    # so it is raised once the plan is closed.
    residue_error: BaseException | None = None
    try:
        reclaim_plan_scoped_residue(runtime, target)
    except BaseException as error:
        residue_error = error
    with runtime._lock:
        if runtime._pending_plan_handle != target:
            raise RuntimeError("Runtime plan setup ownership underflow")
        try:
            _close_and_destroy_plan(target)
        except BaseException as error:
            runtime._unusable_reason = f"execution-plan rollback failed: {error}"
            if residue_error is not None:
                error.add_note(f"Reclaiming the plan's residue failed: {residue_error}")
            raise
        finally:
            runtime._pending_plan_handle = None
    if residue_error is not None:
        raise residue_error


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
