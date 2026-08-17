"""Declarative ctypes projection of the private PyTorch adapter ABI."""

from __future__ import annotations

import ctypes
from typing import Any, Final

ADAPTER_ABI_VERSION: Final = 39
RUNTIME_ABI_VERSION: Final = 31
FIXED_LAYOUT_ABI_VERSION: Final = 2
TRACE_ABI_VERSION: Final = 1
TRANSFER_PROFILE_ABI_VERSION: Final = 2


class AdapterConfig(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("device_ordinal", ctypes.c_int32),
        ("device_budget_bytes", ctypes.c_uint64),
        ("provider_headroom_bytes", ctypes.c_uint64),
        ("spill_pool_bytes", ctypes.c_uint64),
        ("worker_poll_nanoseconds", ctypes.c_uint64),
    ]


class PhysicalAdmission(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("device_ordinal", ctypes.c_int32),
        ("device_budget_bytes", ctypes.c_uint64),
        ("context_bytes", ctypes.c_uint64),
        ("provider_headroom_bytes", ctypes.c_uint64),
        ("execution_pool_bytes", ctypes.c_uint64),
        ("bootstrap_process_bytes", ctypes.c_uint64),
        ("device_used_bytes", ctypes.c_uint64),
        ("device_total_bytes", ctypes.c_uint64),
        ("spill_pool_bytes", ctypes.c_uint64),
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
        ("debug_task_host_timing", ctypes.c_uint8),
        ("runtime_trace", ctypes.c_uint8),
    ]


class TaskHostTiming(ctypes.Structure):
    _fields_ = [
        ("task_id", ctypes.c_uint64),
        ("before_readiness_waits_timestamp_ns", ctypes.c_uint64),
        ("before_task_compute_timestamp_ns", ctypes.c_uint64),
        ("after_task_compute_timestamp_ns", ctypes.c_uint64),
        ("before_readiness_waits_sequence", ctypes.c_uint64),
        ("before_task_compute_sequence", ctypes.c_uint64),
        ("after_task_compute_sequence", ctypes.c_uint64),
        ("before_task_enter_timestamp_ns", ctypes.c_uint64),
        ("before_task_exit_timestamp_ns", ctypes.c_uint64),
        ("after_task_enter_timestamp_ns", ctypes.c_uint64),
        ("after_task_exit_timestamp_ns", ctypes.c_uint64),
    ]


class RuntimeStatistics(ctypes.Structure):
    _fields_ = [
        ("execution_pool_bytes", ctypes.c_uint64),
        ("requested_allocated_bytes", ctypes.c_uint64),
        ("peak_requested_allocated_bytes", ctypes.c_uint64),
        ("allocated_bytes", ctypes.c_uint64),
        ("free_bytes", ctypes.c_uint64),
        ("free_prefix_bytes", ctypes.c_uint64),
        ("largest_free_range_bytes", ctypes.c_uint64),
        ("external_fragmentation_bytes", ctypes.c_uint64),
        ("peak_allocated_bytes", ctypes.c_uint64),
        ("spill_pool_bytes", ctypes.c_uint64),
        ("spill_allocated_bytes", ctypes.c_uint64),
        ("spill_peak_allocated_bytes", ctypes.c_uint64),
        ("live_allocations", ctypes.c_uint64),
        ("blocked_allocators", ctypes.c_uint64),
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
    ]


class AllocationEvent(ctypes.Structure):
    _fields_ = [
        ("sequence", ctypes.c_uint64),
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
        ("begin_timestamp_ns", ctypes.c_uint64),
        ("end_timestamp_ns", ctypes.c_uint64),
        ("active", ctypes.c_uint8),
        ("event_overflow", ctypes.c_uint8),
        ("allocation_event_overflow", ctypes.c_uint8),
    ]


class Allocation(ctypes.Structure):
    _fields_ = [
        ("allocation_id", ctypes.c_uint64),
        ("generation", ctypes.c_uint64),
        ("requested_bytes", ctypes.c_uint64),
        ("charged_bytes", ctypes.c_uint64),
        ("pointer", ctypes.c_void_p),
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
        ("fetch_copies", ctypes.c_uint64),
        ("evict_copies", ctypes.c_uint64),
        ("bytes_fetched", ctypes.c_uint64),
        ("bytes_evicted", ctypes.c_uint64),
        ("event_queries", ctypes.c_uint64),
        ("stream_waits", ctypes.c_uint64),
        ("stream_synchronizations", ctypes.c_uint64),
        ("context_activations", ctypes.c_uint64),
        ("event_pool_capacity", ctypes.c_uint64),
        ("event_pool_in_use", ctypes.c_uint64),
        ("event_pool_peak_in_use", ctypes.c_uint64),
        ("event_pool_driver_creates", ctypes.c_uint64),
        ("event_pool_growth_rejections", ctypes.c_uint64),
        ("event_pool_sealed", ctypes.c_uint8),
    ]


class RuntimeFailure(ctypes.Structure):
    _fields_ = [
        ("status", ctypes.c_uint32),
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
        ("trace_label", ctypes.c_char_p),
    ]


class TaskAllocationContractStep(ctypes.Structure):
    _fields_ = [
        ("allocation_ordinal", ctypes.c_uint64),
        ("requested_bytes", ctypes.c_uint64),
        ("charged_bytes", ctypes.c_uint64),
        ("alignment_bytes", ctypes.c_uint64),
        ("operation", ctypes.c_uint8),
        ("required", ctypes.c_uint8),
    ]


class ExecutionDescription(ctypes.Structure):
    _fields_ = [
        ("task_id", ctypes.c_uint64),
        ("input_object_ids", ctypes.POINTER(ctypes.c_uint64)),
        ("input_count", ctypes.c_uint32),
        ("updates", ctypes.POINTER(ObjectUpdate)),
        ("update_count", ctypes.c_uint32),
        ("actions", ctypes.POINTER(RuntimeAction)),
        ("action_count", ctypes.c_uint32),
        ("allocation_contract_steps", ctypes.POINTER(TaskAllocationContractStep)),
        ("allocation_contract_step_count", ctypes.c_uint32),
        ("enforce_allocation_contract", ctypes.c_uint8),
        ("maximum_requested_allocation_bytes", ctypes.c_uint64),
        ("maximum_charged_allocation_bytes", ctypes.c_uint64),
        ("live_requested_allocation_limit_bytes", ctypes.c_uint64),
        ("live_charged_allocation_limit_bytes", ctypes.c_uint64),
        ("dynamic_scratch_maximum_allocation_bytes", ctypes.c_uint64),
        ("dynamic_scratch_live_limit_bytes", ctypes.c_uint64),
    ]


class FixedPlacementDescription(ctypes.Structure):
    _fields_ = [
        ("task_id", ctypes.c_uint64),
        ("ordinal", ctypes.c_uint64),
        ("object_id", ctypes.c_uint64),
        ("offset", ctypes.c_uint64),
        ("bytes", ctypes.c_uint64),
        ("alignment_bytes", ctypes.c_uint64),
        ("kind", ctypes.c_uint8),
    ]


class FixedDependencyDescription(ctypes.Structure):
    _fields_ = [
        ("predecessor_task_id", ctypes.c_uint64),
        ("predecessor_action_ordinal", ctypes.c_uint64),
        ("successor_task_id", ctypes.c_uint64),
        ("successor_ordinal", ctypes.c_uint64),
        ("successor_kind", ctypes.c_uint8),
    ]


class FixedLayoutDescription(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("slice_bytes", ctypes.c_uint64),
        ("placements", ctypes.POINTER(FixedPlacementDescription)),
        ("placement_count", ctypes.c_uint64),
        ("dependencies", ctypes.POINTER(FixedDependencyDescription)),
        ("dependency_count", ctypes.c_uint64),
    ]


class ObjectSnapshot(ctypes.Structure):
    _fields_ = [
        ("object_id", ctypes.c_uint64),
        ("size_bytes", ctypes.c_uint64),
        ("generation", ctypes.c_uint64),
        ("allocation_id", ctypes.c_uint64),
        ("authoritative_version", ctypes.c_uint64),
        ("execution_version", ctypes.c_uint64),
        ("spill_version", ctypes.c_uint64),
        ("residency", ctypes.c_uint8),
        ("spill_current", ctypes.c_uint8),
        ("has_spill_lease", ctypes.c_uint8),
        ("execution_pointer", ctypes.c_void_p),
        ("spill_pointer", ctypes.c_void_p),
        ("retired_generation", ctypes.c_uint64),
        ("retired_execution_pointer", ctypes.c_void_p),
    ]


def configure_adapter_library(library: Any) -> None:
    """Assign every non-callback adapter signature in one place."""

    _configure_capabilities(library)
    _configure_profiler(library)
    _configure_physical_memory(library)
    _configure_allocator(library)
    _configure_transfer_calibration(library)
    _configure_debug_timing(library)
    _configure_telemetry(library)
    _configure_trace(library)
    _configure_objects(library)
    _configure_task_boundaries(library)
    _configure_execution(library)


def _signature(
    library: Any,
    name: str,
    argument_types: list[object],
    result_type: object,
) -> None:
    function = getattr(library, name)
    function.argtypes = argument_types
    function.restype = result_type


def _configure_capabilities(library: Any) -> None:
    _signature(
        library,
        "shadowspill_pytorch_adapter_capabilities",
        [ctypes.POINTER(AdapterCapabilities)],
        ctypes.c_uint32,
    )


def _configure_profiler(library: Any) -> None:
    _signature(
        library,
        "shadowspill_pytorch_profile_range_begin",
        [ctypes.c_char_p],
        ctypes.c_uint64,
    )
    _signature(
        library, "shadowspill_pytorch_profile_range_end", [ctypes.c_uint64], None
    )
    _signature(
        library,
        "shadowspill_pytorch_profiler_annotations_set",
        [ctypes.c_uint8],
        ctypes.c_uint32,
    )


def _configure_physical_memory(library: Any) -> None:
    _signature(
        library,
        "shadowspill_pytorch_physical_admission",
        [ctypes.POINTER(PhysicalAdmission)],
        ctypes.c_uint32,
    )
    _signature(
        library,
        "shadowspill_pytorch_physical_memory",
        [ctypes.POINTER(PhysicalMemory)],
        ctypes.c_uint32,
    )
    _signature(
        library,
        "shadowspill_pytorch_seal_physical_budget",
        [ctypes.c_uint64, ctypes.c_uint64],
        ctypes.c_uint32,
    )
    _signature(
        library,
        "shadowspill_pytorch_resize_spill_pool",
        [ctypes.c_uint64],
        ctypes.c_uint32,
    )
    _signature(
        library, "shadowspill_pytorch_check_physical_budget", [], ctypes.c_uint32
    )


def _configure_allocator(library: Any) -> None:
    _signature(
        library,
        "shadowspill_pytorch_allocator_bootstrap",
        [ctypes.POINTER(AdapterConfig)],
        ctypes.c_uint32,
    )
    _signature(
        library,
        "shadowspill_pytorch_allocator_statistics",
        [ctypes.POINTER(AdapterStatistics)],
        ctypes.c_uint32,
    )
    _signature(
        library,
        "shadowspill_pytorch_allocator_failure",
        [ctypes.POINTER(AdapterFailure)],
        ctypes.c_uint32,
    )
    _signature(
        library,
        "shadowspill_pytorch_recover_no_progress",
        [],
        ctypes.c_uint32,
    )
    _signature(library, "shadowspill_pytorch_allocator_wait_idle", [], ctypes.c_uint32)
    _signature(
        library,
        "shadowspill_pytorch_allocation_for_pointer",
        [ctypes.c_uint64, ctypes.POINTER(Allocation)],
        ctypes.c_uint32,
    )


def _configure_transfer_calibration(library: Any) -> None:
    _signature(
        library,
        "shadowspill_pytorch_calibrate_transfer_capabilities",
        [
            ctypes.POINTER(TransferCalibrationConfig),
            ctypes.POINTER(TransferRouteKey),
            ctypes.c_uint32,
        ],
        ctypes.c_uint32,
    )
    _signature(
        library,
        "shadowspill_pytorch_transfer_profiles",
        [
            ctypes.POINTER(TransferProfile),
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint64),
        ],
        ctypes.c_uint32,
    )


def _configure_debug_timing(library: Any) -> None:
    _signature(
        library,
        "shadowspill_pytorch_debug_task_timing_enable",
        [ctypes.c_uint32],
        ctypes.c_uint32,
    )
    _signature(
        library,
        "shadowspill_pytorch_debug_task_timing_read",
        [
            ctypes.POINTER(TaskHostTiming),
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32),
        ],
        ctypes.c_uint32,
    )
    _signature(
        library, "shadowspill_pytorch_debug_task_timing_disable", [], ctypes.c_uint32
    )
    _signature(
        library,
        "shadowspill_pytorch_task_labels_configure",
        [ctypes.POINTER(ctypes.c_char_p), ctypes.c_uint32],
        ctypes.c_uint32,
    )


def _configure_telemetry(library: Any) -> None:
    _signature(
        library,
        "shadowspill_pytorch_allocation_telemetry_start",
        [ctypes.c_uint64],
        ctypes.c_uint32,
    )
    _signature(
        library, "shadowspill_pytorch_allocation_telemetry_stop", [], ctypes.c_uint32
    )
    _signature(
        library,
        "shadowspill_pytorch_allocation_telemetry_read",
        [
            ctypes.POINTER(AllocationEvent),
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_uint64),
        ],
        ctypes.c_uint32,
    )


def _configure_trace(library: Any) -> None:
    _signature(
        library,
        "shadowspill_pytorch_trace_prepare",
        [ctypes.POINTER(TraceConfig)],
        ctypes.c_uint32,
    )
    _signature(
        library, "shadowspill_pytorch_trace_begin", [ctypes.c_uint64], ctypes.c_uint32
    )
    _signature(library, "shadowspill_pytorch_trace_end", [], ctypes.c_uint32)
    _signature(
        library,
        "shadowspill_pytorch_trace_read",
        [
            ctypes.POINTER(TraceSummary),
            ctypes.POINTER(TraceEvent),
            ctypes.c_uint64,
            ctypes.POINTER(AllocationEvent),
            ctypes.c_uint64,
        ],
        ctypes.c_uint32,
    )


def _configure_objects(library: Any) -> None:
    _signature(
        library,
        "shadowspill_pytorch_register_host_object",
        [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint8, ctypes.c_uint64],
        ctypes.c_uint32,
    )
    _signature(
        library,
        "shadowspill_pytorch_register_placeholder_object",
        [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint8],
        ctypes.c_uint32,
    )
    _signature(
        library,
        "shadowspill_pytorch_write_spill_object",
        [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64],
        ctypes.c_uint32,
    )
    _signature(
        library,
        "shadowspill_pytorch_read_spill_object",
        [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64],
        ctypes.c_uint32,
    )
    _signature(
        library,
        "shadowspill_pytorch_unregister_object",
        [ctypes.c_uint64],
        ctypes.c_uint32,
    )
    _signature(
        library,
        "shadowspill_pytorch_rekey_object",
        [ctypes.c_uint64, ctypes.c_uint64],
        ctypes.c_uint32,
    )
    binding = [
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.POINTER(ObjectBinding),
    ]
    _signature(
        library,
        "shadowspill_pytorch_bind_registered_allocation",
        binding,
        ctypes.c_uint32,
    )
    _signature(
        library,
        "shadowspill_pytorch_replace_registered_allocation",
        binding,
        ctypes.c_uint32,
    )
    _signature(
        library,
        "shadowspill_pytorch_transfer_output_to_caller",
        [ctypes.c_uint64, ctypes.c_size_t, ctypes.POINTER(Allocation)],
        ctypes.c_uint32,
    )
    _signature(
        library, "shadowspill_pytorch_promote_allocation", binding, ctypes.c_uint32
    )
    _signature(
        library,
        "shadowspill_pytorch_object_snapshot",
        [ctypes.c_uint64, ctypes.POINTER(ObjectSnapshot)],
        ctypes.c_uint32,
    )
    _signature(
        library,
        "shadowspill_pytorch_validate_spill_binding",
        [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64],
        ctypes.c_uint32,
    )


def _configure_task_boundaries(library: Any) -> None:
    _signature(
        library,
        "shadowspill_pytorch_allocation_scope_begin",
        [ctypes.c_uint64],
        ctypes.c_uint32,
    )
    _signature(
        library,
        "shadowspill_pytorch_allocation_scope_end",
        [ctypes.c_uint64, ctypes.c_size_t],
        ctypes.c_uint32,
    )
    _signature(
        library, "shadowspill_pytorch_allocation_scope_abort", [], None
    )
    _signature(
        library,
        "shadowspill_pytorch_before_task",
        [
            ctypes.c_uint64,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_uint32,
            ctypes.POINTER(ObjectBinding),
            ctypes.c_uint32,
        ],
        ctypes.c_uint32,
    )
    _signature(
        library,
        "shadowspill_pytorch_after_task",
        [
            ctypes.c_uint64,
            ctypes.c_size_t,
            ctypes.POINTER(ObjectUpdate),
            ctypes.c_uint32,
            ctypes.POINTER(RuntimeAction),
            ctypes.c_uint32,
        ],
        ctypes.c_uint32,
    )
    _signature(library, "shadowspill_pytorch_abort_task_range", [], None)


def _configure_execution(library: Any) -> None:
    _signature(
        library,
        "shadowspill_pytorch_plan_create",
        [ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_size_t)],
        ctypes.c_uint32,
    )
    _signature(
        library,
        "shadowspill_pytorch_plan_close",
        [ctypes.c_size_t],
        ctypes.c_uint32,
    )
    _signature(
        library,
        "shadowspill_pytorch_plan_destroy",
        [ctypes.c_size_t],
        None,
    )
    _signature(
        library,
        "shadowspill_pytorch_object_handle_acquire",
        [ctypes.c_uint64, ctypes.POINTER(ctypes.c_size_t)],
        ctypes.c_uint32,
    )
    _signature(
        library,
        "shadowspill_pytorch_object_handle_release",
        [ctypes.c_size_t],
        ctypes.c_uint32,
    )
    _signature(
        library,
        "shadowspill_pytorch_object_release_generation",
        [ctypes.c_size_t, ctypes.c_uint64],
        ctypes.c_uint32,
    )
    _signature(
        library,
        "shadowspill_pytorch_plan_bind_object",
        [ctypes.c_size_t, ctypes.c_uint64, ctypes.c_size_t, ctypes.c_uint8],
        ctypes.c_uint32,
    )
    _signature(
        library,
        "shadowspill_pytorch_plan_admit_execution",
        [ctypes.c_size_t, ctypes.POINTER(ExecutionDescription)],
        ctypes.c_uint32,
    )
    _signature(
        library,
        "shadowspill_pytorch_plan_admit_fixed_layout",
        [ctypes.c_size_t, ctypes.POINTER(FixedLayoutDescription)],
        ctypes.c_uint32,
    )
    _signature(
        library,
        "shadowspill_pytorch_plan_seal_fixed_layout",
        [ctypes.c_size_t],
        ctypes.c_uint32,
    )
    _signature(
        library,
        "shadowspill_pytorch_plan_clear_execution",
        [ctypes.c_size_t],
        ctypes.c_uint32,
    )
    _signature(
        library,
        "shadowspill_pytorch_plan_resolve_execution",
        [ctypes.c_size_t, ctypes.c_uint64, ctypes.POINTER(ctypes.c_size_t)],
        ctypes.c_uint32,
    )
    _signature(
        library,
        "shadowspill_pytorch_admit_execution",
        [ctypes.POINTER(ExecutionDescription)],
        ctypes.c_uint32,
    )
    _signature(
        library,
        "shadowspill_pytorch_admit_fixed_layout",
        [ctypes.POINTER(FixedLayoutDescription)],
        ctypes.c_uint32,
    )
    _signature(
        library,
        "shadowspill_pytorch_seal_fixed_layout",
        [],
        ctypes.c_uint32,
    )
    _signature(library, "shadowspill_pytorch_clear_execution_plan", [], ctypes.c_uint32)
    _signature(
        library,
        "shadowspill_pytorch_resolve_execution",
        [ctypes.c_uint64, ctypes.POINTER(ctypes.c_size_t)],
        ctypes.c_uint32,
    )
    _signature(
        library,
        "shadowspill_pytorch_before_execution",
        [
            ctypes.c_uint64,
            ctypes.c_size_t,
            ctypes.POINTER(ObjectBinding),
            ctypes.c_uint32,
        ],
        ctypes.c_uint32,
    )
    _signature(
        library,
        "shadowspill_pytorch_after_execution",
        [ctypes.c_uint64, ctypes.c_size_t],
        ctypes.c_uint32,
    )
    _signature(
        library,
        "shadowspill_pytorch_before_execution_handle",
        [
            ctypes.c_size_t,
            ctypes.c_uint64,
            ctypes.c_size_t,
            ctypes.POINTER(ObjectBinding),
            ctypes.c_uint32,
        ],
        ctypes.c_uint32,
    )
    _signature(
        library,
        "shadowspill_pytorch_after_execution_handle",
        [ctypes.c_size_t, ctypes.c_uint64, ctypes.c_size_t],
        ctypes.c_uint32,
    )
