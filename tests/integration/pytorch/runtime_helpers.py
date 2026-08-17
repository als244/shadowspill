"""Small ctypes helpers shared by fresh-process runtime canaries."""

from __future__ import annotations

import ctypes

from shadowspill.pytorch.runtime_adapter.abi import ObjectBinding
from shadowspill.pytorch.runtime_adapter.allocator import (
    PoolBootstrap,
    RouteBootstrap,
)


def two_pool_topology(spill_bytes: int) -> dict[str, object]:
    """Return the canonical execution/spill topology for adapter canaries."""

    return {
        "allocator_pool_id": 0,
        "pools": (
            PoolBootstrap(pool_id=0, backend_kind=0, capacity_bytes=0),
            PoolBootstrap(pool_id=1, backend_kind=1, capacity_bytes=spill_bytes),
        ),
        "routes": (
            RouteBootstrap(
                route_id=0,
                name="fetch",
                source_pool_id=1,
                destination_pool_id=0,
            ),
            RouteBootstrap(
                route_id=1,
                name="evict",
                source_pool_id=0,
                destination_pool_id=1,
            ),
        ),
    }


def begin_task(
    library: object,
    task_handle: int,
    task_id: int,
    stream_address: int,
    *,
    expected_bindings: int,
) -> tuple[ObjectBinding, ...]:
    """Enter one admitted task and expose its borrowed binding view."""

    bindings = ctypes.POINTER(ObjectBinding)()
    count = ctypes.c_uint32()
    status = int(
        library.shadowspill_pytorch_before_task_handle(
            task_handle,
            stream_address,
            ctypes.byref(bindings),
            ctypes.byref(count),
        )
    )
    if status != 0:
        raise AssertionError(f"task {task_id} entry failed with status {status}")
    if count.value != expected_bindings:
        raise AssertionError(
            f"task {task_id} returned {count.value} bindings, "
            f"expected {expected_bindings}"
        )
    if count.value != 0 and not bindings:
        raise AssertionError(f"task {task_id} returned a null binding view")
    return tuple(bindings[index] for index in range(count.value))


__all__ = ["begin_task", "two_pool_topology"]
