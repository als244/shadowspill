"""What a caller hands the adapter to bring a runtime up, and what it asks back."""

from __future__ import annotations

import ctypes


class PoolConfig(ctypes.Structure):
    _fields_ = [
        ("pool_id", ctypes.c_uint32),
        ("kind", ctypes.c_uint8),
        ("capacity_bytes", ctypes.c_uint64),
        # Forwarded to this pool's kind untouched; nothing between here and
        # there reads it. NULL for a kind that needs none.
        ("configuration", ctypes.c_void_p),
    ]


class RouteConfig(ctypes.Structure):
    _fields_ = [
        ("route_id", ctypes.c_uint32),
        ("source_pool_id", ctypes.c_uint32),
        ("destination_pool_id", ctypes.c_uint32),
        ("name", ctypes.c_char_p),
    ]


class AdapterConfig(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("device_ordinal", ctypes.c_int32),
        ("device_budget_bytes", ctypes.c_uint64),
        ("provider_headroom_bytes", ctypes.c_uint64),
        ("allocator_pool_id", ctypes.c_uint32),
        ("pools", ctypes.POINTER(PoolConfig)),
        ("pool_count", ctypes.c_uint32),
        ("routes", ctypes.POINTER(RouteConfig)),
        ("route_count", ctypes.c_uint32),
        ("worker_poll_nanoseconds", ctypes.c_uint64),
        ("background_transfer_window_bytes", ctypes.c_uint64),
        ("backend_library", ctypes.c_char_p),
        # Extension libraries supplying pool kinds and lanes, loaded in order
        # and kept open for the runtime's life.
        ("libraries", ctypes.POINTER(ctypes.c_char_p)),
        ("library_count", ctypes.c_uint32),
    ]


class PhysicalAdmission(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("device_ordinal", ctypes.c_int32),
        ("device_budget_bytes", ctypes.c_uint64),
        ("baseline_bytes", ctypes.c_uint64),
        ("provider_headroom_bytes", ctypes.c_uint64),
        ("allocator_pool_id", ctypes.c_uint32),
        ("pool_count", ctypes.c_uint32),
        ("allocator_pool_bytes", ctypes.c_uint64),
        ("bootstrap_process_bytes", ctypes.c_uint64),
        ("device_used_bytes", ctypes.c_uint64),
        ("device_total_bytes", ctypes.c_uint64),
    ]


class PhysicalMemory(ctypes.Structure):
    _fields_ = [
        ("process_bytes", ctypes.c_uint64),
        ("device_used_bytes", ctypes.c_uint64),
        ("device_total_bytes", ctypes.c_uint64),
    ]


class AdapterCapabilities(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("runtime_abi_version", ctypes.c_uint32),
        ("backend_abi_version", ctypes.c_uint32),
        ("storage_rebinding", ctypes.c_uint8),
    ]


class TraceConfig(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("event_capacity", ctypes.c_uint64),
        ("allocation_event_capacity", ctypes.c_uint64),
    ]


class TransferRouteKey(ctypes.Structure):
    _fields_ = [
        ("source_pool_id", ctypes.c_uint32),
        ("destination_pool_id", ctypes.c_uint32),
    ]


class TransferCalibrationConfig(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("small_copy_bytes", ctypes.c_uint64),
        ("large_copy_bytes", ctypes.c_uint64),
        ("warmup_copies", ctypes.c_uint32),
        ("measured_copies", ctypes.c_uint32),
        ("provenance", ctypes.c_uint8),
    ]
