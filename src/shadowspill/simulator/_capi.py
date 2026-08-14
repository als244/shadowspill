"""Declarative ctypes surface for the versioned simulator C ABI."""

from __future__ import annotations

import ctypes
from functools import cache
from pathlib import Path

from shadowspill._libraries import resolve_library

ABI_VERSION = 2
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
        ("host_capacity_bytes", ctypes.c_uint64),
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
        ("initial_aliases", ctypes.POINTER(ctypes.c_uint32)),
        ("initial_locations", ctypes.POINTER(ctypes.c_uint8)),
        ("final_aliases", ctypes.POINTER(ctypes.c_uint32)),
        ("final_locations", ctypes.POINTER(ctypes.c_uint8)),
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
        ("host_peak_bytes", ctypes.c_uint64),
        ("task_intervals", ctypes.POINTER(CTaskInterval)),
        ("task_interval_capacity", ctypes.c_uint32),
        ("task_interval_count", ctypes.c_uint32),
        ("transfer_intervals", ctypes.POINTER(CTransferInterval)),
        ("transfer_interval_capacity", ctypes.c_uint32),
        ("transfer_interval_count", ctypes.c_uint32),
        ("device_peaks", ctypes.POINTER(CDevicePeak)),
        ("device_peak_capacity", ctypes.c_uint32),
    ]


def simulator_library_path() -> Path | None:
    return resolve_library("libshadowspill_simulator.so")


@cache
def load_simulator_library() -> ctypes.CDLL:
    path = simulator_library_path()
    if path is None:
        raise RuntimeError(
            "libshadowspill_simulator.so was not found; install ShadowSpill or "
            "build the editable checkout at its configured build location"
        )
    library = ctypes.CDLL(str(path))
    library.shadowspill_simulator_abi_version.argtypes = []
    library.shadowspill_simulator_abi_version.restype = ctypes.c_uint32
    found = int(library.shadowspill_simulator_abi_version())
    if found != ABI_VERSION:
        raise RuntimeError(
            f"simulator ABI mismatch: Python expects {ABI_VERSION}, library has {found}"
        )
    library.shadowspill_simulate.argtypes = [
        ctypes.POINTER(CProgram),
        ctypes.POINTER(CResult),
    ]
    library.shadowspill_simulate.restype = ctypes.c_uint32
    library.shadowspill_simulation_status_string.argtypes = [ctypes.c_uint32]
    library.shadowspill_simulation_status_string.restype = ctypes.c_char_p
    return library


__all__ = [
    "ABI_VERSION",
    "NO_INDEX",
    "CDevice",
    "CDevicePeak",
    "CProgram",
    "CResult",
    "CTaskInterval",
    "CTransferInterval",
    "load_simulator_library",
    "simulator_library_path",
]
