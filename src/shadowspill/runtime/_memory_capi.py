"""Declarative ctypes surface for the neutral MemoryPool replay ABI."""

from __future__ import annotations

import ctypes
from functools import cache

from shadowspill._libraries import resolve_library

ABI_VERSION = 1
NO_ID = (1 << 64) - 1


class CMemoryReplayOperation(ctypes.Structure):
    _fields_ = [
        ("sequence", ctypes.c_uint64),
        ("lease_id", ctypes.c_uint64),
        ("dependency_id", ctypes.c_uint64),
        ("bytes", ctypes.c_uint64),
        ("alignment", ctypes.c_uint64),
        ("kind", ctypes.c_uint8),
        ("dependency_expected", ctypes.c_uint8),
    ]


class CMemoryReplayProgram(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("capacity_bytes", ctypes.c_uint64),
        ("minimum_alignment", ctypes.c_uint64),
        ("lease_count", ctypes.c_uint64),
        ("dependency_count", ctypes.c_uint64),
        ("operations", ctypes.POINTER(CMemoryReplayOperation)),
        ("operation_count", ctypes.c_uint64),
    ]


class CMemoryReplayDecision(ctypes.Structure):
    _fields_ = [
        ("operation_index", ctypes.c_uint64),
        ("sequence", ctypes.c_uint64),
        ("lease_id", ctypes.c_uint64),
        ("predecessor_lease_id", ctypes.c_uint64),
        ("dependency_id", ctypes.c_uint64),
        ("offset", ctypes.c_uint64),
        ("requested_bytes", ctypes.c_uint64),
        ("charged_bytes", ctypes.c_uint64),
        ("physical_bytes_delta", ctypes.c_int64),
        ("resulting_state", ctypes.c_uint8),
    ]


class CMemoryReuseDependency(ctypes.Structure):
    _fields_ = [
        ("predecessor_lease_id", ctypes.c_uint64),
        ("successor_lease_id", ctypes.c_uint64),
        ("dependency_id", ctypes.c_uint64),
        ("consumer_operation_index", ctypes.c_uint64),
    ]


class CMemoryReplayResult(ctypes.Structure):
    _fields_ = [
        ("status", ctypes.c_uint32),
        ("error_operation_index", ctypes.c_uint64),
        ("error_lease_id", ctypes.c_uint64),
        ("error_requested_bytes", ctypes.c_uint64),
        ("error_free_bytes", ctypes.c_uint64),
        ("error_largest_free_range_bytes", ctypes.c_uint64),
        ("peak_allocated_bytes", ctypes.c_uint64),
        ("peak_reserved_bytes", ctypes.c_uint64),
        ("peak_fragmentation_bytes", ctypes.c_uint64),
        ("final_allocated_bytes", ctypes.c_uint64),
        ("final_reserved_bytes", ctypes.c_uint64),
        ("final_largest_free_range_bytes", ctypes.c_uint64),
        ("decision_digest", ctypes.c_uint64),
        ("decisions", ctypes.POINTER(CMemoryReplayDecision)),
        ("decision_capacity", ctypes.c_uint64),
        ("decision_count", ctypes.c_uint64),
        ("dependencies", ctypes.POINTER(CMemoryReuseDependency)),
        ("dependency_capacity", ctypes.c_uint64),
        ("dependency_result_count", ctypes.c_uint64),
    ]


@cache
def load_memory_replay_library() -> ctypes.CDLL:
    path = resolve_library("libshadowspill_runtime.so")
    if path is None:
        raise RuntimeError(
            "libshadowspill_runtime.so was not found; install ShadowSpill or "
            "build the editable checkout at its configured build location"
        )
    library = ctypes.CDLL(str(path))
    library.shadowspill_memory_replay_abi_version.argtypes = []
    library.shadowspill_memory_replay_abi_version.restype = ctypes.c_uint32
    actual_abi = int(library.shadowspill_memory_replay_abi_version())
    if actual_abi != ABI_VERSION:
        raise RuntimeError(
            "ShadowSpill memory replay ABI mismatch: "
            f"expected {ABI_VERSION}, found {actual_abi}"
        )
    library.shadowspill_memory_replay_run.argtypes = [
        ctypes.POINTER(CMemoryReplayProgram),
        ctypes.POINTER(CMemoryReplayResult),
    ]
    library.shadowspill_memory_replay_run.restype = ctypes.c_uint32
    library.shadowspill_memory_replay_status_string.argtypes = [ctypes.c_uint32]
    library.shadowspill_memory_replay_status_string.restype = ctypes.c_char_p
    return library


__all__ = [
    "ABI_VERSION",
    "NO_ID",
    "CMemoryReplayDecision",
    "CMemoryReplayOperation",
    "CMemoryReplayProgram",
    "CMemoryReplayResult",
    "CMemoryReuseDependency",
    "load_memory_replay_library",
]
