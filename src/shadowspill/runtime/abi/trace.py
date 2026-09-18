"""The event records a runtime trace and the allocation telemetry emit."""

from __future__ import annotations

import ctypes


class AllocationEvent(ctypes.Structure):
    _fields_ = [
        ("sequence", ctypes.c_uint64),
        ("pool_id", ctypes.c_uint32),
        ("task_id", ctypes.c_uint64),
        ("allocation_id", ctypes.c_uint64),
        ("generation", ctypes.c_uint64),
        ("requested_bytes", ctypes.c_uint64),
        ("charged_bytes", ctypes.c_uint64),
        ("alignment_bytes", ctypes.c_uint64),
        ("slab_offset", ctypes.c_uint64),
        ("kind", ctypes.c_uint8),
        ("category", ctypes.c_uint8),
    ]


class TraceEvent(ctypes.Structure):
    _fields_ = [
        ("sequence", ctypes.c_uint64),
        ("timestamp_ns", ctypes.c_uint64),
        ("step_id", ctypes.c_uint64),
        ("task_id", ctypes.c_uint64),
        ("object_id", ctypes.c_uint64),
        ("allocation_id", ctypes.c_uint64),
        ("bytes", ctypes.c_uint64),
        ("detail_0", ctypes.c_uint64),
        ("detail_1", ctypes.c_uint64),
        ("lane_issued_at_ns", ctypes.c_uint64),
        ("lane_started_at_ns", ctypes.c_uint64),
        ("lane_finished_at_ns", ctypes.c_uint64),
        ("kind", ctypes.c_uint8),
    ]


class TraceSummary(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("step_id", ctypes.c_uint64),
        ("event_count", ctypes.c_uint64),
        ("allocation_event_count", ctypes.c_uint64),
        ("event_capacity", ctypes.c_uint64),
        ("allocation_event_capacity", ctypes.c_uint64),
        ("began_at_ns", ctypes.c_uint64),
        ("ended_at_ns", ctypes.c_uint64),
        ("origin_host_ns", ctypes.c_uint64),
        ("active", ctypes.c_uint8),
        ("event_overflow", ctypes.c_uint8),
        ("allocation_event_overflow", ctypes.c_uint8),
    ]
