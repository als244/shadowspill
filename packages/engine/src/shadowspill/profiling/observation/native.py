"""Bounded native-allocator telemetry; no allocator replacement or pool setup."""

import ctypes
from collections.abc import Iterator
from contextlib import contextmanager
from functools import cache
from typing import Any

from shadowspill.libraries import resolve_library


class _Record(ctypes.Structure):
    _fields_ = [
        (name, ctypes.c_uint64)
        for name in (
            "address",
            "requested_bytes",
            "stream",
            "pool_first",
            "pool_second",
        )
    ] + [("device", ctypes.c_int32), ("action", ctypes.c_int32)]


_ACTIONS = (
    "allocate",
    "free_requested",
    "free_completed",
    "segment_allocate",
    "segment_free",
    "segment_map",
    "segment_unmap",
    "snapshot",
    "out_of_memory",
)


@cache
def _library() -> Any:
    # torch must be loaded before its adapter library; this does not install it.
    import torch

    if torch.cuda.memory.get_allocator_backend() != "native":
        raise RuntimeError("allocation observation requires PyTorch's native allocator")
    path = resolve_library("libshadowspill_pytorch.so")
    if path is None:
        raise RuntimeError("build/install the PyTorch adapter to observe allocations")
    library = ctypes.CDLL(str(path))
    library.shadowspill_pytorch_observe_allocations_begin.argtypes = [
        ctypes.c_int32,
        ctypes.c_uint64,
    ]
    library.shadowspill_pytorch_observe_allocations_begin.restype = ctypes.c_int
    library.shadowspill_pytorch_observe_allocations_end.argtypes = [
        ctypes.POINTER(_Record),
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint32),
    ]
    library.shadowspill_pytorch_observe_allocations_end.restype = ctypes.c_int
    return library


@contextmanager
def observe_allocations(
    device: int, *, capacity: int = 131072
) -> Iterator[list[dict[str, Any]]]:
    """Yield a list populated at scope exit; overflow never yields partial evidence."""
    if not 0 < capacity <= 1000000:
        raise ValueError("observation capacity must be between 1 and 1,000,000")
    library = _library()
    records = (_Record * capacity)()
    count, overflowed = ctypes.c_uint64(), ctypes.c_uint32()
    status = library.shadowspill_pytorch_observe_allocations_begin(device, capacity)
    if status:
        raise RuntimeError(f"cannot start allocation observation (status {status})")
    result: list[dict[str, Any]] = []
    failure = None
    try:
        yield result
    except BaseException as error:
        failure = error
        raise
    finally:
        status = library.shadowspill_pytorch_observe_allocations_end(
            records, capacity, ctypes.byref(count), ctypes.byref(overflowed)
        )
        message = (
            f"cannot stop allocation observation (status {status})"
            if status
            else f"allocation observation overflowed {capacity} events"
            if overflowed.value
            else None
        )
        if message:
            if failure is not None:
                failure.add_note(message)
            else:
                raise RuntimeError(message)
        else:
            for record in records[: count.value]:
                item = {
                    field[0]: getattr(record, field[0]) for field in _Record._fields_
                }
                item["action"] = _ACTIONS[record.action]
                result.append(item)
