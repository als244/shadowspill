"""The ctypes buffers a C program points into, and the codes it reads."""

from __future__ import annotations

import ctypes
from array import array
from dataclasses import dataclass, field
from typing import TypeVar

from shadowspill.ir import (
    MemoryActionKind,
    MemoryLocation,
    ResourceKind,
)

_RESOURCE_CODE = {
    ResourceKind.COMPUTE: 0,
    ResourceKind.COMMUNICATION: 1,
    ResourceKind.CONTROL: 2,
}
_LOCATION_CODE = {
    MemoryLocation.DEVICE: 0,
    MemoryLocation.SPILL: 1,
}
_ACTION_CODE = {
    MemoryActionKind.RELEASE: 0,
    MemoryActionKind.EVICT: 1,
    MemoryActionKind.FETCH: 2,
    MemoryActionKind.WRITE_BACK: 3,
}
_STALL_REASONS = (
    (1 << 0, "input-residency"),
    (1 << 1, "device-capacity"),
    (1 << 2, "source-readiness"),
    (1 << 3, "host-capacity"),
    (1 << 4, "memory-reuse"),
)
_VIOLATION_REASONS = (
    "initial-device-capacity",
    "initial-spill-capacity",
    "fetch-device-capacity",
    "evict-spill-capacity",
    "task-device-capacity",
)
_VIOLATION_LOCATIONS = ("device", "spill")
_DEFAULT_PHYSICAL_DELTA = -(1 << 63)

_RESOURCE_CODE = {
    ResourceKind.COMPUTE: 0,
    ResourceKind.COMMUNICATION: 1,
    ResourceKind.CONTROL: 2,
}
_LOCATION_CODE = {
    MemoryLocation.DEVICE: 0,
    MemoryLocation.SPILL: 1,
}
_ACTION_CODE = {
    MemoryActionKind.RELEASE: 0,
    MemoryActionKind.EVICT: 1,
    MemoryActionKind.FETCH: 2,
    MemoryActionKind.WRITE_BACK: 3,
}
_DEFAULT_PHYSICAL_DELTA = -(1 << 63)


def _u32_array(values: tuple[int, ...]) -> ctypes.Array[ctypes.c_uint32]:
    array_type = ctypes.c_uint32 * max(1, len(values))
    return array_type.from_buffer_copy(array("I", values)) if values else array_type()


def _u64_array(values: tuple[int, ...]) -> ctypes.Array[ctypes.c_uint64]:
    array_type = ctypes.c_uint64 * max(1, len(values))
    return array_type.from_buffer_copy(array("Q", values)) if values else array_type()


def _i64_array(values: tuple[int, ...]) -> ctypes.Array[ctypes.c_int64]:
    array_type = ctypes.c_int64 * max(1, len(values))
    return array_type(*values) if values else array_type()


def _u8_array(values: tuple[int, ...]) -> ctypes.Array[ctypes.c_uint8]:
    array_type = ctypes.c_uint8 * max(1, len(values))
    return array_type.from_buffer_copy(bytes(values)) if values else array_type()


_T = TypeVar("_T")


@dataclass(slots=True)
class _Arena:
    """Every ctypes buffer a C program points into, kept alive alongside it.

    ctypes does not own what a pointer field is assigned, so each array has to
    outlive the structure that points at it; the arena is that lifetime.
    """

    buffers: list[object] = field(default_factory=list)

    def keep(self, value: _T) -> _T:
        self.buffers.append(value)
        return value

    def u32(self, values: tuple[int, ...]) -> ctypes.Array[ctypes.c_uint32]:
        return self.keep(_u32_array(values))

    def u64(self, values: tuple[int, ...]) -> ctypes.Array[ctypes.c_uint64]:
        return self.keep(_u64_array(values))

    def u8(self, values: tuple[int, ...]) -> ctypes.Array[ctypes.c_uint8]:
        return self.keep(_u8_array(values))

    def i64(self, values: tuple[int, ...]) -> ctypes.Array[ctypes.c_int64]:
        return self.keep(_i64_array(values))
