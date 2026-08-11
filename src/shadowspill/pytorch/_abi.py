"""Declarative ctypes projection of the private PyTorch adapter ABI."""

from __future__ import annotations

import ctypes
from typing import Any, Final

ADAPTER_ABI_VERSION: Final = 2


class AdapterConfig(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("device_ordinal", ctypes.c_int32),
        ("device_budget_bytes", ctypes.c_uint64),
        ("provider_headroom_bytes", ctypes.c_uint64),
        ("host_arena_bytes", ctypes.c_uint64),
        ("progress_poll_nanoseconds", ctypes.c_uint64),
    ]


class PhysicalAdmission(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("device_ordinal", ctypes.c_int32),
        ("device_budget_bytes", ctypes.c_uint64),
        ("context_bytes", ctypes.c_uint64),
        ("provider_headroom_bytes", ctypes.c_uint64),
        ("slab_bytes", ctypes.c_uint64),
        ("bootstrap_process_bytes", ctypes.c_uint64),
        ("device_used_bytes", ctypes.c_uint64),
        ("device_total_bytes", ctypes.c_uint64),
        ("host_arena_bytes", ctypes.c_uint64),
    ]


class PhysicalMemory(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("process_bytes", ctypes.c_uint64),
        ("device_used_bytes", ctypes.c_uint64),
        ("device_total_bytes", ctypes.c_uint64),
    ]


class AdapterCapabilities(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("runtime_abi_version", ctypes.c_uint32),
        ("backend_abi_version", ctypes.c_uint32),
        ("slab_memory_strategy", ctypes.c_uint8),
        ("record_stream_callback", ctypes.c_uint8),
        ("storage_rebinding", ctypes.c_uint8),
    ]


class RuntimeStatistics(ctypes.Structure):
    _fields_ = [
        ("slab_bytes", ctypes.c_uint64),
        ("allocated_bytes", ctypes.c_uint64),
        ("free_bytes", ctypes.c_uint64),
        ("largest_free_range_bytes", ctypes.c_uint64),
        ("external_fragmentation_bytes", ctypes.c_uint64),
        ("peak_allocated_bytes", ctypes.c_uint64),
        ("host_arena_bytes", ctypes.c_uint64),
        ("host_allocated_bytes", ctypes.c_uint64),
        ("host_peak_allocated_bytes", ctypes.c_uint64),
        ("live_allocations", ctypes.c_uint64),
        ("blocked_allocators", ctypes.c_uint64),
        ("pending_retirements", ctypes.c_uint64),
        ("registered_objects", ctypes.c_uint64),
        ("queued_actions", ctypes.c_uint64),
        ("transfers_to_device", ctypes.c_uint64),
        ("transfers_to_host", ctypes.c_uint64),
        ("bytes_to_device", ctypes.c_uint64),
        ("bytes_to_host", ctypes.c_uint64),
        ("wait_events_inserted", ctypes.c_uint64),
    ]


class CudaStatistics(ctypes.Structure):
    _fields_ = [
        ("device_allocations", ctypes.c_uint64),
        ("device_frees", ctypes.c_uint64),
        ("pinned_host_allocations", ctypes.c_uint64),
        ("pinned_host_frees", ctypes.c_uint64),
        ("streams_created", ctypes.c_uint64),
        ("streams_destroyed", ctypes.c_uint64),
        ("events_created", ctypes.c_uint64),
        ("events_destroyed", ctypes.c_uint64),
        ("copies_to_device", ctypes.c_uint64),
        ("copies_to_host", ctypes.c_uint64),
        ("bytes_to_device", ctypes.c_uint64),
        ("bytes_to_host", ctypes.c_uint64),
        ("event_queries", ctypes.c_uint64),
        ("stream_waits", ctypes.c_uint64),
        ("stream_synchronizations", ctypes.c_uint64),
        ("context_activations", ctypes.c_uint64),
    ]


class RuntimeFailure(ctypes.Structure):
    _fields_ = [
        ("status", ctypes.c_uint32),
        ("object_id", ctypes.c_uint64),
        ("allocation_id", ctypes.c_uint64),
        ("requested_bytes", ctypes.c_uint64),
        ("free_bytes", ctypes.c_uint64),
        ("largest_free_range_bytes", ctypes.c_uint64),
    ]


class AdapterFailure(ctypes.Structure):
    _fields_ = [
        ("status", ctypes.c_uint32),
        ("device_ordinal", ctypes.c_int32),
        ("address", ctypes.c_uint64),
        ("requested_bytes", ctypes.c_uint64),
        ("runtime", RuntimeFailure),
    ]


class AdapterStatistics(ctypes.Structure):
    _fields_ = [
        ("allocation_callbacks", ctypes.c_uint64),
        ("zero_size_allocation_callbacks", ctypes.c_uint64),
        ("free_callbacks", ctypes.c_uint64),
        ("record_stream_callbacks", ctypes.c_uint64),
        ("pointer_lookup_failures", ctypes.c_uint64),
        ("callback_failures", ctypes.c_uint64),
        ("runtime", RuntimeStatistics),
        ("cuda", CudaStatistics),
    ]


class ObjectBinding(ctypes.Structure):
    _fields_ = [
        ("object_id", ctypes.c_uint64),
        ("generation", ctypes.c_uint64),
        ("allocation_id", ctypes.c_uint64),
        ("authoritative_version", ctypes.c_uint64),
        ("pointer", ctypes.c_void_p),
    ]


class ObjectUpdate(ctypes.Structure):
    _fields_ = [
        ("object_id", ctypes.c_uint64),
        ("version_delta", ctypes.c_uint64),
    ]


class RuntimeAction(ctypes.Structure):
    _fields_ = [
        ("object_id", ctypes.c_uint64),
        ("kind", ctypes.c_uint8),
    ]


class ObjectSnapshot(ctypes.Structure):
    _fields_ = [
        ("object_id", ctypes.c_uint64),
        ("size_bytes", ctypes.c_uint64),
        ("generation", ctypes.c_uint64),
        ("allocation_id", ctypes.c_uint64),
        ("authoritative_version", ctypes.c_uint64),
        ("device_version", ctypes.c_uint64),
        ("host_version", ctypes.c_uint64),
        ("residency", ctypes.c_uint8),
        ("host_current", ctypes.c_uint8),
        ("has_host_range", ctypes.c_uint8),
        ("device_pointer", ctypes.c_void_p),
    ]


def configure_adapter_library(library: Any) -> None:
    """Assign every non-callback adapter signature in one place."""

    library.shadowspill_pytorch_adapter_capabilities.argtypes = [
        ctypes.POINTER(AdapterCapabilities)
    ]
    library.shadowspill_pytorch_adapter_capabilities.restype = ctypes.c_uint32
    library.shadowspill_pytorch_physical_admission.argtypes = [
        ctypes.POINTER(PhysicalAdmission)
    ]
    library.shadowspill_pytorch_physical_admission.restype = ctypes.c_uint32
    library.shadowspill_pytorch_physical_memory.argtypes = [
        ctypes.POINTER(PhysicalMemory)
    ]
    library.shadowspill_pytorch_physical_memory.restype = ctypes.c_uint32
    library.shadowspill_pytorch_allocator_bootstrap.argtypes = [
        ctypes.POINTER(AdapterConfig)
    ]
    library.shadowspill_pytorch_allocator_bootstrap.restype = ctypes.c_uint32
    library.shadowspill_pytorch_allocator_statistics.argtypes = [
        ctypes.POINTER(AdapterStatistics)
    ]
    library.shadowspill_pytorch_allocator_statistics.restype = ctypes.c_uint32
    library.shadowspill_pytorch_allocator_failure.argtypes = [
        ctypes.POINTER(AdapterFailure)
    ]
    library.shadowspill_pytorch_allocator_failure.restype = ctypes.c_uint32
    library.shadowspill_pytorch_allocator_wait_idle.argtypes = []
    library.shadowspill_pytorch_allocator_wait_idle.restype = ctypes.c_uint32
    library.shadowspill_pytorch_promote_allocation.argtypes = [
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.POINTER(ObjectBinding),
    ]
    library.shadowspill_pytorch_promote_allocation.restype = ctypes.c_uint32
    library.shadowspill_pytorch_before_task.argtypes = [
        ctypes.c_uint64,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_uint32,
        ctypes.POINTER(ObjectBinding),
        ctypes.c_uint32,
    ]
    library.shadowspill_pytorch_before_task.restype = ctypes.c_uint32
    library.shadowspill_pytorch_after_task.argtypes = [
        ctypes.c_uint64,
        ctypes.c_size_t,
        ctypes.POINTER(ObjectUpdate),
        ctypes.c_uint32,
        ctypes.POINTER(RuntimeAction),
        ctypes.c_uint32,
    ]
    library.shadowspill_pytorch_after_task.restype = ctypes.c_uint32
    library.shadowspill_pytorch_object_snapshot.argtypes = [
        ctypes.c_uint64,
        ctypes.POINTER(ObjectSnapshot),
    ]
    library.shadowspill_pytorch_object_snapshot.restype = ctypes.c_uint32
    library.shadowspill_pytorch_abort_task_range.argtypes = []
    library.shadowspill_pytorch_abort_task_range.restype = None
