"""Declarative ctypes surface for the neutral AdmissionReplay ABI."""

from __future__ import annotations

import ctypes
from functools import cache

from shadowspill._libraries import (
    load_shadowspill_library,
)

NO_ID = (1 << 64) - 1


class CAdmissionReplayOperation(ctypes.Structure):
    _fields_ = [
        ("sequence", ctypes.c_uint64),
        ("lease_id", ctypes.c_uint64),
        ("dependency_id", ctypes.c_uint64),
        ("bytes", ctypes.c_uint64),
        ("alignment", ctypes.c_uint64),
        ("kind", ctypes.c_uint8),
        ("dependency_expected", ctypes.c_uint8),
    ]


class CAdmissionReplayProgram(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("capacity_bytes", ctypes.c_uint64),
        ("minimum_alignment", ctypes.c_uint64),
        ("large_request_threshold_bytes", ctypes.c_uint64),
        ("lease_count", ctypes.c_uint64),
        ("dependency_count", ctypes.c_uint64),
        ("operations", ctypes.POINTER(CAdmissionReplayOperation)),
        ("operation_count", ctypes.c_uint64),
    ]


class CAdmissionReplayDecision(ctypes.Structure):
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


class CAdmissionReuseDependency(ctypes.Structure):
    _fields_ = [
        ("predecessor_lease_id", ctypes.c_uint64),
        ("successor_lease_id", ctypes.c_uint64),
        ("dependency_id", ctypes.c_uint64),
        ("consumer_operation_index", ctypes.c_uint64),
    ]


class CAdmissionReplayLiveLease(ctypes.Structure):
    _fields_ = [
        ("lease_id", ctypes.c_uint64),
        ("offset", ctypes.c_uint64),
        ("requested_bytes", ctypes.c_uint64),
        ("charged_bytes", ctypes.c_uint64),
        ("state", ctypes.c_uint8),
    ]


class CAdmissionReplayResult(ctypes.Structure):
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
        ("decisions", ctypes.POINTER(CAdmissionReplayDecision)),
        ("decision_capacity", ctypes.c_uint64),
        ("decision_count", ctypes.c_uint64),
        ("dependencies", ctypes.POINTER(CAdmissionReuseDependency)),
        ("dependency_capacity", ctypes.c_uint64),
        ("dependency_result_count", ctypes.c_uint64),
        ("live_leases", ctypes.POINTER(CAdmissionReplayLiveLease)),
        ("live_lease_capacity", ctypes.c_uint64),
        ("live_lease_count", ctypes.c_uint64),
    ]


@cache
def load_admission_replay_library() -> ctypes.CDLL:
    library = load_shadowspill_library()
    library.shadowspill_admission_replay_run.argtypes = [
        ctypes.POINTER(CAdmissionReplayProgram),
        ctypes.POINTER(CAdmissionReplayResult),
    ]
    library.shadowspill_admission_replay_run.restype = ctypes.c_uint32
    library.shadowspill_status_string.argtypes = [ctypes.c_uint32]
    library.shadowspill_status_string.restype = ctypes.c_char_p
    return library


__all__ = [
    "NO_ID",
    "CAdmissionReplayDecision",
    "CAdmissionReplayLiveLease",
    "CAdmissionReplayOperation",
    "CAdmissionReplayProgram",
    "CAdmissionReplayResult",
    "CAdmissionReuseDependency",
    "load_admission_replay_library",
]
