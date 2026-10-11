"""Runtime objects and who owns them: identities, registration, references.

A runtime object is a value the runtime moves between pools. These functions
take the identities the frontend names them by, register them with the neutral
runtime, hand out the public references (`ObjectRef`) that keep a runtime open
until they close, and count the persistent state that does the same. The
counters live on the runtime; the rules about when each operation is allowed
live here.
"""

from __future__ import annotations

import ctypes
from typing import TYPE_CHECKING

from ..abi import ObjectDescription, runtime_library
from ..configuration import RuntimeConfigurationError
from ..failures import RuntimeExecutionError
from .references import ObjectRef

if TYPE_CHECKING:
    from ..core import Runtime


def reserve_persistent_object_ids(
    runtime: Runtime,
    count: int,
    *,
    allow_in_progress_plan: bool = False,
) -> tuple[int, ...]:
    """Reserve globally unique runtime object identities."""

    if count < 0:
        raise ValueError("persistent object count must be non-negative")
    with runtime._lock:
        require_state_operation_allowed(
            runtime, allow_in_progress_plan=allow_in_progress_plan
        )
        return reserve_runtime_object_ids(runtime, count)


def reserve_runtime_object_ids(runtime: Runtime, count: int) -> tuple[int, ...]:
    """Reserve runtime-global identities without changing state ownership."""

    if count < 0:
        raise ValueError("runtime object count must be non-negative")
    with runtime._lock:
        runtime._require_open()
        first = runtime._next_persistent_object_id
        limit = first + count
        if limit >= (1 << 63):
            raise RuntimeConfigurationError(
                "persistent object identity space is exhausted"
            )
        runtime._next_persistent_object_id = limit
        return tuple(range(first, limit))


def register_object(
    runtime: Runtime,
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
            runtime._runtime_handle, ctypes.byref(description)
        )
    )
    if status != 0 or source_address == 0:
        return status
    return int(
        runtime_library().shadowspill_write_object(
            runtime._runtime_handle, object_id, pool_id, source_address, size_bytes
        )
    )


def register_placed_object(
    runtime: Runtime,
    object_id: int,
    size_bytes: int,
    *,
    pool_id: int,
    offset: int,
) -> int:
    """Register one object over the bytes already at ``offset`` of its pool.

    The range was reserved for the runtime's lifetime with
    :func:`reserve_pool_range`; nothing is written, and unregistering the
    object leaves the range reserved. This is how state a pool image holds
    is bound without a copy.
    """

    description = ObjectDescription(
        object_id=object_id,
        size_bytes=size_bytes,
        initial_pool_id=pool_id,
        retain_spill_copy=1,
        initially_resident=1,
    )
    return int(
        runtime_library().shadowspill_register_placed_object(
            runtime._runtime_handle, ctypes.byref(description), offset
        )
    )


def reserve_pool_range(
    runtime: Runtime, *, pool_id: int, offset: int, size_bytes: int
) -> int:
    """Reserve one range of a pool for the runtime's lifetime."""

    return int(
        runtime_library().shadowspill_memory_pool_reserve_range(
            runtime._runtime_handle, pool_id, offset, size_bytes
        )
    )


def pool_base_address(runtime: Runtime, pool_id: int) -> int:
    """The address a pool's offset 0 presents; lease pointers are offsets from it."""

    base = ctypes.c_void_p()
    status = int(
        runtime_library().shadowspill_memory_pool_base(
            runtime._runtime_handle, pool_id, ctypes.byref(base)
        )
    )
    if status != 0:
        raise RuntimeExecutionError(
            f"failed to read the base of pool {pool_id}: status={status}"
        )
    return int(base.value or 0)


def acquire_object_reference(
    runtime: Runtime,
    *,
    object_id: int,
    size_bytes: int,
) -> ObjectRef:
    """Create one public owner for an existing runtime object."""

    with runtime._lock:
        runtime._require_open()
        handle = ctypes.c_size_t()
        status = int(
            runtime_library().shadowspill_object_handle_acquire(
                runtime._runtime_handle, object_id, ctypes.byref(handle)
            )
        )
        if status != 0 or handle.value == 0:
            raise RuntimeExecutionError(
                f"failed to retain runtime object {object_id}: status={status}"
            )
        try:
            reference = ObjectRef(
                runtime,
                object_id=object_id,
                size_bytes=size_bytes,
                handle=int(handle.value),
            )
        except BaseException:
            runtime_library().shadowspill_object_handle_release(handle.value)
            raise
        runtime._active_object_references += 1
        return reference


def release_object_reference(runtime: Runtime, reference: ObjectRef) -> None:
    """Release exactly one public runtime-object owner."""

    with runtime._lock:
        if not reference._belongs_to(runtime):
            raise RuntimeError("runtime object reference belongs to another Runtime")
        if runtime._active_object_references <= 0:
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
        runtime._active_object_references -= 1


def release_object_generation(
    runtime: Runtime,
    *,
    object_id: int,
    expected_generation: int,
) -> None:
    """Release a completed value while retaining its logical identity."""

    with runtime._lock:
        runtime._require_open()
        handle = ctypes.c_size_t()
        status = int(
            runtime_library().shadowspill_object_handle_acquire(
                runtime._runtime_handle, object_id, ctypes.byref(handle)
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


def retain_persistent_state(
    runtime: Runtime, *, allow_in_progress_plan: bool = False
) -> None:
    with runtime._lock:
        require_state_operation_allowed(
            runtime, allow_in_progress_plan=allow_in_progress_plan
        )
        runtime._persistent_state_count += 1


def release_persistent_state(runtime: Runtime) -> None:
    with runtime._lock:
        if runtime._persistent_state_count <= 0:
            raise RuntimeError("persistent state ownership underflow")
        runtime._persistent_state_count -= 1


def require_state_operation_allowed(
    runtime: Runtime, *, allow_in_progress_plan: bool = False
) -> None:
    with runtime._lock:
        runtime._require_open()
        if allow_in_progress_plan:
            # A plan being made imports state of its own -- new objects that
            # no admitted plan binds -- so the others may stay admitted.
            return
        if runtime._active_plan_handles or runtime._planning_plan_handle is not None:
            raise RuntimeConfigurationError(
                "persistent state import requires an idle Runtime"
            )


__all__ = [
    "acquire_object_reference",
    "register_object",
    "release_object_generation",
    "release_object_reference",
    "release_persistent_state",
    "require_state_operation_allowed",
    "reserve_persistent_object_ids",
    "reserve_runtime_object_ids",
    "retain_persistent_state",
]
