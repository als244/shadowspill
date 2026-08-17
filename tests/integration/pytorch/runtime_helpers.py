"""Small ctypes helpers shared by fresh-process runtime canaries."""

from __future__ import annotations

import ctypes

from shadowspill.pytorch.runtime_adapter.abi import ObjectBinding


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
            task_id,
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


__all__ = ["begin_task"]
