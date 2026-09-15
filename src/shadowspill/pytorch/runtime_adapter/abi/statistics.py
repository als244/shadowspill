"""What the pools, the runtime and the backend report when asked."""

from __future__ import annotations

import ctypes


class LiveAllocation(ctypes.Structure):
    """One live allocation, mirroring `ShadowSpillLiveAllocation`."""

    _fields_ = [
        ("allocation_id", ctypes.c_uint64),
        ("offset", ctypes.c_uint64),
        ("charged_bytes", ctypes.c_uint64),
        ("requested_bytes", ctypes.c_uint64),
        ("origin_plan_id", ctypes.c_uint64),
        ("origin_task_id", ctypes.c_uint64),
        ("origin_task_invocation", ctypes.c_uint64),
        ("origin_task_allocation_ordinal", ctypes.c_uint64),
        ("object_id", ctypes.c_uint64),
        ("references", ctypes.c_uint32),
        ("scratch", ctypes.c_uint8),
        ("plan_owned", ctypes.c_uint8),
        ("ever_plan_owned", ctypes.c_uint8),
        ("logical_freed", ctypes.c_uint8),
        ("framework_free_seen", ctypes.c_uint8),
    ]


class MemoryPoolStatistics(ctypes.Structure):
    """What one pool holds, mirroring `ShadowSpillMemoryPoolStatistics`.

    Per pool rather than flattened into named fields for two of them: a runtime
    may own any number of pools, and which of them a plan uses as its execution
    and spill pools is the plan's choice.
    """

    _fields_ = [
        ("pool_id", ctypes.c_uint32),
        ("kind", ctypes.c_uint8),
        ("capacity_bytes", ctypes.c_uint64),
        ("requested_allocated_bytes", ctypes.c_uint64),
        ("peak_requested_allocated_bytes", ctypes.c_uint64),
        ("allocated_bytes", ctypes.c_uint64),
        ("peak_allocated_bytes", ctypes.c_uint64),
        ("free_bytes", ctypes.c_uint64),
        ("free_prefix_bytes", ctypes.c_uint64),
        ("largest_free_range_bytes", ctypes.c_uint64),
        ("external_fragmentation_bytes", ctypes.c_uint64),
        ("live_allocations", ctypes.c_uint64),
        ("blocked_allocators", ctypes.c_uint64),
        ("memory_lease_record_capacity", ctypes.c_uint64),
        ("memory_lease_record_in_use", ctypes.c_uint64),
        ("memory_lease_record_peak_in_use", ctypes.c_uint64),
        ("memory_lease_record_growth_rejections", ctypes.c_uint64),
        ("lease_use_record_capacity", ctypes.c_uint64),
        ("lease_use_record_in_use", ctypes.c_uint64),
        ("lease_use_record_peak_in_use", ctypes.c_uint64),
        ("lease_use_record_growth_rejections", ctypes.c_uint64),
    ]


class RuntimeStatistics(ctypes.Structure):
    """What the runtime holds that no pool does.

    The work in flight, the records it owns, and how many pools there are to ask
    about. A pool's own numbers are in `MemoryPoolStatistics`.
    """

    _fields_ = [
        ("pool_count", ctypes.c_uint32),
        ("pending_retirements", ctypes.c_uint64),
        ("retirement_records_fenced", ctypes.c_uint64),
        ("retirement_records_evented", ctypes.c_uint64),
        ("retirement_records_preparing", ctypes.c_uint64),
        ("retirement_records_unfenced", ctypes.c_uint64),
        ("registered_objects", ctypes.c_uint64),
        ("queued_actions", ctypes.c_uint64),
        ("fetch_transfers", ctypes.c_uint64),
        ("evict_transfers", ctypes.c_uint64),
        ("bytes_fetched", ctypes.c_uint64),
        ("bytes_evicted", ctypes.c_uint64),
        ("wait_events_inserted", ctypes.c_uint64),
        ("allocation_events", ctypes.c_uint64),
        ("allocation_event_capacity", ctypes.c_uint64),
        ("allocation_event_overflow", ctypes.c_uint64),
        ("event_lease_capacity", ctypes.c_uint64),
        ("event_lease_in_use", ctypes.c_uint64),
        ("event_lease_peak_in_use", ctypes.c_uint64),
        ("event_lease_growth_rejections", ctypes.c_uint64),
        ("event_lease_driver_creates", ctypes.c_uint64),
        ("event_lease_sealed", ctypes.c_uint64),
        ("timing_event_capacity", ctypes.c_uint64),
        ("timing_event_in_use", ctypes.c_uint64),
        ("timing_event_peak_in_use", ctypes.c_uint64),
        ("timing_event_driver_creates", ctypes.c_uint64),
        ("retirement_record_capacity", ctypes.c_uint64),
        ("retirement_record_in_use", ctypes.c_uint64),
        ("retirement_record_peak_in_use", ctypes.c_uint64),
        ("retirement_record_growth_rejections", ctypes.c_uint64),
        ("caller_owned_allocations", ctypes.c_uint64),
    ]


class BackendStatistics(ctypes.Structure):
    _fields_ = [
        ("device_allocations", ctypes.c_uint64),
        ("device_frees", ctypes.c_uint64),
        ("bytes_device_allocated", ctypes.c_uint64),
        ("bytes_device_freed", ctypes.c_uint64),
        ("pinned_host_registrations", ctypes.c_uint64),
        ("pinned_host_unregistrations", ctypes.c_uint64),
        ("bytes_pinned_host_registered", ctypes.c_uint64),
        ("bytes_pinned_host_unregistered", ctypes.c_uint64),
        ("streams_created", ctypes.c_uint64),
        ("streams_destroyed", ctypes.c_uint64),
        ("events_created", ctypes.c_uint64),
        ("events_destroyed", ctypes.c_uint64),
        ("copies_host_to_device", ctypes.c_uint64),
        ("copies_device_to_host", ctypes.c_uint64),
        ("copies_device_to_device", ctypes.c_uint64),
        ("bytes_host_to_device", ctypes.c_uint64),
        ("bytes_device_to_host", ctypes.c_uint64),
        ("bytes_device_to_device", ctypes.c_uint64),
        ("event_queries", ctypes.c_uint64),
        ("stream_waits", ctypes.c_uint64),
        ("stream_synchronizations", ctypes.c_uint64),
        ("provider_activations", ctypes.c_uint64),
    ]


class AdapterStatistics(ctypes.Structure):
    _fields_ = [
        ("allocation_callbacks", ctypes.c_uint64),
        ("zero_size_allocation_callbacks", ctypes.c_uint64),
        ("free_callbacks", ctypes.c_uint64),
        ("record_stream_callbacks", ctypes.c_uint64),
        ("pointer_lookup_failures", ctypes.c_uint64),
        ("callback_failures", ctypes.c_uint64),
        ("physical_checks", ctypes.c_uint64),
        ("peak_process_physical_bytes", ctypes.c_uint64),
        ("observed_external_high_water_bytes", ctypes.c_uint64),
        ("physical_budget_sealed", ctypes.c_uint64),
        ("runtime", RuntimeStatistics),
        ("allocator_pool", MemoryPoolStatistics),
        ("backend", BackendStatistics),
    ]


class Allocation(ctypes.Structure):
    _fields_ = [
        ("pool_id", ctypes.c_uint32),
        ("allocation_id", ctypes.c_uint64),
        ("generation", ctypes.c_uint64),
        ("requested_bytes", ctypes.c_uint64),
        ("charged_bytes", ctypes.c_uint64),
        ("pointer", ctypes.c_void_p),
    ]


class TransferProfile(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("source_pool_id", ctypes.c_uint32),
        ("destination_pool_id", ctypes.c_uint32),
        ("generation", ctypes.c_uint64),
        ("latency_nanoseconds", ctypes.c_uint64),
        ("bandwidth_bytes_per_second", ctypes.c_uint64),
        ("solo_bandwidth_bytes_per_second", ctypes.c_uint64),
        ("concurrent_bandwidth_bytes_per_second", ctypes.c_uint64),
        ("solo_measurement_nanoseconds", ctypes.c_uint64),
        ("concurrent_measurement_nanoseconds", ctypes.c_uint64),
        ("calibrated_timestamp_nanoseconds", ctypes.c_uint64),
        ("small_copy_bytes", ctypes.c_uint64),
        ("large_copy_bytes", ctypes.c_uint64),
        ("measured_copies", ctypes.c_uint32),
        ("available", ctypes.c_uint8),
        ("calibrated", ctypes.c_uint8),
        ("provenance", ctypes.c_uint8),
        ("calibration_mode", ctypes.c_uint8),
        ("concurrent_route_count", ctypes.c_uint8),
    ]
