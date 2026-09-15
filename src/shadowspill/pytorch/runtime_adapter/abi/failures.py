"""What a failed call latches, as the runtime and the adapter each record it."""

from __future__ import annotations

import ctypes


class RuntimeFailure(ctypes.Structure):
    _fields_ = [
        ("status", ctypes.c_uint32),
        ("reason", ctypes.c_uint32),
        ("pool_id", ctypes.c_uint32),
        ("task_id", ctypes.c_uint64),
        ("object_id", ctypes.c_uint64),
        ("allocation_id", ctypes.c_uint64),
        ("requested_bytes", ctypes.c_uint64),
        ("free_bytes", ctypes.c_uint64),
        ("largest_free_range_bytes", ctypes.c_uint64),
        ("task_live_requested_bytes", ctypes.c_uint64),
        ("task_live_charged_bytes", ctypes.c_uint64),
        ("task_live_requested_limit_bytes", ctypes.c_uint64),
        ("task_live_charged_limit_bytes", ctypes.c_uint64),
        ("task_maximum_requested_allocation_bytes", ctypes.c_uint64),
        ("task_maximum_charged_allocation_bytes", ctypes.c_uint64),
        ("task_allocation_operation_index", ctypes.c_uint64),
        ("task_allocation_expected_ordinal", ctypes.c_uint64),
        ("task_allocation_actual_ordinal", ctypes.c_uint64),
        ("task_allocation_expected_requested_bytes", ctypes.c_uint64),
        ("task_allocation_actual_requested_bytes", ctypes.c_uint64),
        ("task_allocation_expected_charged_bytes", ctypes.c_uint64),
        ("task_allocation_actual_charged_bytes", ctypes.c_uint64),
        ("task_allocation_expected_alignment_bytes", ctypes.c_uint64),
        ("task_allocation_actual_alignment_bytes", ctypes.c_uint64),
        ("task_allocation_expected_operation", ctypes.c_uint8),
        ("task_allocation_actual_operation", ctypes.c_uint8),
    ]


class AdapterFailure(ctypes.Structure):
    _fields_ = [
        ("status", ctypes.c_uint32),
        ("device_ordinal", ctypes.c_int32),
        ("address", ctypes.c_uint64),
        ("requested_bytes", ctypes.c_uint64),
        ("runtime", RuntimeFailure),
    ]
