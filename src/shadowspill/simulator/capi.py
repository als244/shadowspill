"""Declarative ctypes surface for the versioned simulator C ABI."""

from __future__ import annotations

import ctypes
from functools import cache

from shadowspill.libraries import (
    load_shadowspill_library,
)

NO_INDEX = (1 << 32) - 1


class CDevice(ctypes.Structure):
    _fields_ = [
        ("capacity_bytes", ctypes.c_uint64),
        ("fetch_bandwidth_bytes_per_second", ctypes.c_uint64),
        ("evict_bandwidth_bytes_per_second", ctypes.c_uint64),
        ("fetch_latency_ns", ctypes.c_uint64),
        ("evict_latency_ns", ctypes.c_uint64),
    ]


class CProgram(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("device_count", ctypes.c_uint32),
        ("alias_count", ctypes.c_uint32),
        ("task_count", ctypes.c_uint32),
        ("action_count", ctypes.c_uint32),
        ("initial_count", ctypes.c_uint32),
        ("final_count", ctypes.c_uint32),
        ("dependency_count", ctypes.c_uint32),
        ("input_count", ctypes.c_uint32),
        ("output_count", ctypes.c_uint32),
        ("mutation_count", ctypes.c_uint32),
        ("reuse_dependency_count", ctypes.c_uint32),
        ("use_admission_accounting", ctypes.c_uint32),
        ("spill_capacity_bytes", ctypes.c_uint64),
        ("devices", ctypes.POINTER(CDevice)),
        ("alias_device", ctypes.POINTER(ctypes.c_uint32)),
        ("alias_size_bytes", ctypes.POINTER(ctypes.c_uint64)),
        ("alias_initial_version", ctypes.POINTER(ctypes.c_uint64)),
        ("alias_retain_spill_copy", ctypes.POINTER(ctypes.c_uint8)),
        ("task_device", ctypes.POINTER(ctypes.c_uint32)),
        ("task_resource_kind", ctypes.POINTER(ctypes.c_uint8)),
        ("task_resource_lane", ctypes.POINTER(ctypes.c_uint32)),
        ("task_runtime_ns", ctypes.POINTER(ctypes.c_uint64)),
        ("task_workspace_bytes", ctypes.POINTER(ctypes.c_uint64)),
        ("task_start_physical_deltas", ctypes.POINTER(ctypes.c_int64)),
        ("task_completion_physical_deltas", ctypes.POINTER(ctypes.c_int64)),
        ("dependency_offsets", ctypes.POINTER(ctypes.c_uint32)),
        ("dependencies", ctypes.POINTER(ctypes.c_uint32)),
        ("input_offsets", ctypes.POINTER(ctypes.c_uint32)),
        ("input_aliases", ctypes.POINTER(ctypes.c_uint32)),
        ("output_offsets", ctypes.POINTER(ctypes.c_uint32)),
        ("output_aliases", ctypes.POINTER(ctypes.c_uint32)),
        ("mutation_offsets", ctypes.POINTER(ctypes.c_uint32)),
        ("mutation_aliases", ctypes.POINTER(ctypes.c_uint32)),
        ("mutation_version_deltas", ctypes.POINTER(ctypes.c_uint64)),
        ("action_trigger_tasks", ctypes.POINTER(ctypes.c_uint32)),
        ("action_aliases", ctypes.POINTER(ctypes.c_uint32)),
        ("action_kinds", ctypes.POINTER(ctypes.c_uint8)),
        ("action_trigger_physical_deltas", ctypes.POINTER(ctypes.c_int64)),
        ("action_completion_physical_deltas", ctypes.POINTER(ctypes.c_int64)),
        ("initial_aliases", ctypes.POINTER(ctypes.c_uint32)),
        ("initial_locations", ctypes.POINTER(ctypes.c_uint8)),
        ("initial_physical_bytes", ctypes.POINTER(ctypes.c_uint64)),
        ("final_aliases", ctypes.POINTER(ctypes.c_uint32)),
        ("final_locations", ctypes.POINTER(ctypes.c_uint8)),
        ("reuse_predecessor_actions", ctypes.POINTER(ctypes.c_uint32)),
        ("reuse_successor_tasks", ctypes.POINTER(ctypes.c_uint32)),
        ("reuse_successor_actions", ctypes.POINTER(ctypes.c_uint32)),
    ]


class CTaskInterval(ctypes.Structure):
    _fields_ = [
        ("task", ctypes.c_uint32),
        ("ready_ns", ctypes.c_uint64),
        ("start_ns", ctypes.c_uint64),
        ("end_ns", ctypes.c_uint64),
        ("workspace_bytes", ctypes.c_uint64),
        ("stall_mask", ctypes.c_uint32),
    ]


class CTransferInterval(ctypes.Structure):
    _fields_ = [
        ("alias", ctypes.c_uint32),
        ("trigger_task", ctypes.c_uint32),
        ("device", ctypes.c_uint32),
        ("direction", ctypes.c_uint8),
        ("sequence", ctypes.c_uint32),
        ("ready_ns", ctypes.c_uint64),
        ("start_ns", ctypes.c_uint64),
        ("end_ns", ctypes.c_uint64),
        ("bytes", ctypes.c_uint64),
        ("stall_mask", ctypes.c_uint32),
    ]


class CDevicePeak(ctypes.Structure):
    _fields_ = [
        ("object_bytes", ctypes.c_uint64),
        ("workspace_bytes", ctypes.c_uint64),
        ("total_bytes", ctypes.c_uint64),
    ]


class CResult(ctypes.Structure):
    _fields_ = [
        ("status", ctypes.c_uint32),
        ("error_task", ctypes.c_uint32),
        ("error_alias", ctypes.c_uint32),
        ("error_device", ctypes.c_uint32),
        ("error_location", ctypes.c_uint8),
        ("error_time_ns", ctypes.c_uint64),
        ("error_capacity_bytes", ctypes.c_uint64),
        ("error_used_bytes", ctypes.c_uint64),
        ("error_requested_bytes", ctypes.c_uint64),
        ("makespan_ns", ctypes.c_uint64),
        ("spill_peak_bytes", ctypes.c_uint64),
        ("task_intervals", ctypes.POINTER(CTaskInterval)),
        ("task_interval_capacity", ctypes.c_uint32),
        ("task_interval_count", ctypes.c_uint32),
        ("transfer_intervals", ctypes.POINTER(CTransferInterval)),
        ("transfer_interval_capacity", ctypes.c_uint32),
        ("transfer_interval_count", ctypes.c_uint32),
        ("device_peaks", ctypes.POINTER(CDevicePeak)),
        ("device_peak_capacity", ctypes.c_uint32),
    ]


@cache
def simulator_api() -> ctypes.CDLL:
    library = load_shadowspill_library()
    library.shadowspill_simulate.argtypes = [
        ctypes.POINTER(CProgram),
        ctypes.POINTER(CResult),
    ]
    library.shadowspill_simulate.restype = ctypes.c_uint32
    library.shadowspill_status_string.argtypes = [ctypes.c_uint32]
    library.shadowspill_status_string.restype = ctypes.c_char_p
    return library


__all__ = [
    "NO_INDEX",
    "CDevice",
    "CDevicePeak",
    "CProgram",
    "CResult",
    "CTaskInterval",
    "CTransferInterval",
    "simulator_api",
]
