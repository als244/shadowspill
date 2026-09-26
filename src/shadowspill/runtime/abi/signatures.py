"""Every adapter and runtime entry point, declared once with its signature."""

from __future__ import annotations

import ctypes
from functools import cache
from typing import Any

from shadowspill.libraries import load_shadowspill_library

from .configuration import (
    AdapterCapabilities,
    AdapterConfig,
    PhysicalAdmission,
    PhysicalMemory,
    TraceConfig,
    TransferCalibrationConfig,
    TransferRouteKey,
)
from .failures import AdapterFailure
from .plans import (
    FixedLayoutDescription,
    ObjectBinding,
    ObjectDescription,
    ObjectLocationSnapshot,
    ObjectSnapshot,
    PlanDescription,
    RuntimeAction,
    TaskDescription,
)
from .statistics import (
    AdapterStatistics,
    Allocation,
    LaneStatistics,
    LiveAllocation,
    MemoryPoolStatistics,
    PlanSliceRecord,
    TransferProfile,
)
from .trace import AllocationEvent, TraceEvent, TraceSummary


def configure_adapter_library(library: Any) -> None:
    """Assign every non-callback adapter signature in one place."""

    _configure_capabilities(library)
    _configure_physical_memory(library)
    _configure_allocator(library)
    _configure_objects(library)
    _configure_task_boundaries(library)
    _configure_execution(library)


#: Plan admission and object handles are the neutral runtime's own API, and
#: the bridge calls them directly. A handle is declared `c_size_t` rather
#: than `c_void_p` because it is pointer-sized either way and reads back as
#: a plain integer.
_RUNTIME_SIGNATURES: tuple[tuple[str, list[object], object], ...] = (
    ("shadowspill_failure_reason_string", [ctypes.c_uint32], ctypes.c_char_p),
    ("shadowspill_plan_close", [ctypes.c_size_t], ctypes.c_uint32),
    ("shadowspill_plan_destroy", [ctypes.c_size_t], None),
    ("shadowspill_plan_wait_idle", [ctypes.c_size_t], ctypes.c_uint32),
    ("shadowspill_plan_require_empty_layout", [ctypes.c_size_t], ctypes.c_uint32),
    (
        "shadowspill_timing_marker_create",
        [ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p)],
        ctypes.c_uint32,
    ),
    (
        "shadowspill_timing_marker_record",
        [ctypes.c_void_p, ctypes.c_uint64],
        ctypes.c_uint32,
    ),
    (
        "shadowspill_timing_marker_query",
        [ctypes.c_void_p, ctypes.c_void_p],
        ctypes.c_uint32,
    ),
    (
        "shadowspill_timing_elapsed",
        [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p],
        ctypes.c_uint32,
    ),
    (
        "shadowspill_profiler_range_begin",
        [ctypes.c_size_t, ctypes.c_char_p],
        ctypes.c_uint64,
    ),
    ("shadowspill_profiler_range_end", [ctypes.c_size_t, ctypes.c_uint64], None),
    (
        "shadowspill_profiler_annotations_set",
        [ctypes.c_size_t, ctypes.c_uint8],
        ctypes.c_uint32,
    ),
    (
        "shadowspill_acquire_objects_handle",
        [
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_uint64,
            ctypes.POINTER(ObjectBinding),
            ctypes.c_uint32,
        ],
        ctypes.c_uint32,
    ),
    (
        "shadowspill_submit_action_batch_handle",
        [ctypes.c_size_t, ctypes.c_size_t, ctypes.c_uint64],
        ctypes.c_uint32,
    ),
    ("shadowspill_timing_marker_wait", [ctypes.c_void_p], ctypes.c_uint32),
    (
        "shadowspill_timing_stream_wait",
        [ctypes.c_size_t, ctypes.c_uint64],
        ctypes.c_uint32,
    ),
    ("shadowspill_timing_marker_release", [ctypes.c_void_p], None),
    ("shadowspill_plan_clear_tasks", [ctypes.c_size_t], ctypes.c_uint32),
    ("shadowspill_plan_seal_fixed_layout", [ctypes.c_size_t], ctypes.c_uint32),
    (
        "shadowspill_plan_bind_object",
        [ctypes.c_size_t, ctypes.c_uint64, ctypes.c_size_t, ctypes.c_uint8],
        ctypes.c_uint32,
    ),
    (
        "shadowspill_plan_admit_task",
        [
            ctypes.c_size_t,
            ctypes.POINTER(TaskDescription),
            ctypes.POINTER(ctypes.c_size_t),
        ],
        ctypes.c_uint32,
    ),
    (
        "shadowspill_plan_publish_initial_allocation",
        [
            ctypes.c_size_t,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.POINTER(ObjectBinding),
        ],
        ctypes.c_uint32,
    ),
    (
        "shadowspill_plan_admit_fixed_layout",
        [ctypes.c_size_t, ctypes.POINTER(FixedLayoutDescription)],
        ctypes.c_uint32,
    ),
    (
        "shadowspill_plan_admit_fixed_layout_in",
        [ctypes.c_size_t, ctypes.POINTER(FixedLayoutDescription), ctypes.c_size_t],
        ctypes.c_uint32,
    ),
    (
        "shadowspill_plan_admit_object_acquisition",
        [
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_size_t),
        ],
        ctypes.c_uint32,
    ),
    (
        "shadowspill_plan_admit_action_batch",
        [
            ctypes.c_size_t,
            ctypes.c_uint64,
            ctypes.POINTER(RuntimeAction),
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_size_t),
        ],
        ctypes.c_uint32,
    ),
    ("shadowspill_object_handle_release", [ctypes.c_size_t], ctypes.c_uint32),
    (
        "shadowspill_object_release_generation",
        [ctypes.c_size_t, ctypes.c_uint64],
        ctypes.c_uint32,
    ),
    ("shadowspill_trace_end", [ctypes.c_size_t], ctypes.c_uint32),
    (
        "shadowspill_trace_prepare",
        [ctypes.c_size_t, ctypes.POINTER(TraceConfig)],
        ctypes.c_uint32,
    ),
    (
        "shadowspill_trace_begin",
        [ctypes.c_size_t, ctypes.c_uint64, ctypes.c_void_p],
        ctypes.c_uint32,
    ),
    (
        "shadowspill_trace_read",
        [
            ctypes.c_size_t,
            ctypes.POINTER(TraceSummary),
            ctypes.POINTER(TraceEvent),
            ctypes.c_uint64,
            ctypes.POINTER(AllocationEvent),
            ctypes.c_uint64,
        ],
        ctypes.c_uint32,
    ),
    (
        "shadowspill_allocation_telemetry_start",
        [ctypes.c_size_t, ctypes.c_uint64],
        ctypes.c_uint32,
    ),
    ("shadowspill_allocation_telemetry_stop", [ctypes.c_size_t], ctypes.c_uint32),
    (
        "shadowspill_allocation_telemetry_read",
        [
            ctypes.c_size_t,
            ctypes.POINTER(AllocationEvent),
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_uint64),
        ],
        ctypes.c_uint32,
    ),
    ("shadowspill_runtime_wait_idle", [ctypes.c_size_t], ctypes.c_uint32),
    ("shadowspill_plan_id", [ctypes.c_size_t], ctypes.c_uint64),
    (
        "shadowspill_plan_reclaim_scoped_leases",
        [ctypes.c_size_t, ctypes.POINTER(ctypes.c_uint64)],
        ctypes.c_uint32,
    ),
    (
        "shadowspill_memory_pool_statistics",
        [ctypes.c_size_t, ctypes.c_uint32, ctypes.POINTER(MemoryPoolStatistics)],
        ctypes.c_uint32,
    ),
    (
        "shadowspill_route_lane_statistics",
        [ctypes.c_size_t, ctypes.c_uint32, ctypes.POINTER(LaneStatistics)],
        ctypes.c_uint32,
    ),
    (
        "shadowspill_runtime_next_plan_id",
        [ctypes.c_size_t, ctypes.POINTER(ctypes.c_uint64)],
        ctypes.c_uint32,
    ),
    (
        "shadowspill_runtime_plan_state",
        [ctypes.c_size_t, ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint32)],
        ctypes.c_uint32,
    ),
    (
        "shadowspill_memory_pool_live_allocations",
        [
            ctypes.c_size_t,
            ctypes.c_uint32,
            ctypes.POINTER(LiveAllocation),
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_uint64),
        ],
        ctypes.c_uint32,
    ),
    (
        "shadowspill_memory_pool_plan_slices",
        [
            ctypes.c_size_t,
            ctypes.c_uint32,
            ctypes.POINTER(PlanSliceRecord),
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_uint64),
        ],
        ctypes.c_uint32,
    ),
    (
        "shadowspill_runtime_calibrate_transfer_capabilities",
        [
            ctypes.c_size_t,
            ctypes.POINTER(TransferCalibrationConfig),
            ctypes.POINTER(TransferRouteKey),
            ctypes.c_uint32,
        ],
        ctypes.c_uint32,
    ),
    (
        "shadowspill_runtime_transfer_profiles",
        [
            ctypes.c_size_t,
            ctypes.POINTER(TransferProfile),
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint64),
        ],
        ctypes.c_uint32,
    ),
    (
        "shadowspill_register_object",
        [ctypes.c_size_t, ctypes.POINTER(ObjectDescription)],
        ctypes.c_uint32,
    ),
    (
        "shadowspill_unregister_object",
        [ctypes.c_size_t, ctypes.c_uint64],
        ctypes.c_uint32,
    ),
    (
        "shadowspill_rekey_object",
        [ctypes.c_size_t, ctypes.c_uint64, ctypes.c_uint64],
        ctypes.c_uint32,
    ),
    (
        "shadowspill_object_snapshot",
        [ctypes.c_size_t, ctypes.c_uint64, ctypes.POINTER(ObjectSnapshot)],
        ctypes.c_uint32,
    ),
    (
        "shadowspill_object_location_snapshot",
        [
            ctypes.c_size_t,
            ctypes.c_uint64,
            ctypes.c_uint32,
            ctypes.POINTER(ObjectLocationSnapshot),
        ],
        ctypes.c_uint32,
    ),
    (
        "shadowspill_read_object",
        [
            ctypes.c_size_t,
            ctypes.c_uint64,
            ctypes.c_uint32,
            ctypes.c_uint64,
            ctypes.c_uint64,
        ],
        ctypes.c_uint32,
    ),
    (
        "shadowspill_write_object",
        [
            ctypes.c_size_t,
            ctypes.c_uint64,
            ctypes.c_uint32,
            ctypes.c_uint64,
            ctypes.c_uint64,
        ],
        ctypes.c_uint32,
    ),
    (
        "shadowspill_object_handle_acquire",
        [ctypes.c_size_t, ctypes.c_uint64, ctypes.POINTER(ctypes.c_size_t)],
        ctypes.c_uint32,
    ),
    (
        "shadowspill_plan_create",
        [
            ctypes.c_size_t,
            ctypes.POINTER(PlanDescription),
            ctypes.POINTER(ctypes.c_size_t),
        ],
        ctypes.c_uint32,
    ),
)


@cache
def runtime_library() -> Any:
    """The neutral runtime library, with the signatures the bridge calls."""

    library = load_shadowspill_library()
    configure_runtime_library(library)
    return library


def configure_runtime_library(library: Any) -> None:
    """Declare the neutral runtime signatures the bridge calls directly."""

    for name, arguments, result in _RUNTIME_SIGNATURES:
        _signature(library, name, arguments, result)


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
    _signature(
        library,
        "shadowspill_pytorch_runtime_handle",
        [ctypes.POINTER(ctypes.c_size_t)],
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
        library, "shadowspill_pytorch_check_physical_budget", [], ctypes.c_uint32
    )


def _configure_allocator(library: Any) -> None:
    _signature(
        library,
        "shadowspill_pytorch_allocator_bootstrap",
        [ctypes.POINTER(AdapterConfig)],
        ctypes.c_uint32,
    )
    _signature(library, "shadowspill_pytorch_allocator_close", [], ctypes.c_uint32)
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
    _signature(
        library,
        "shadowspill_pytorch_allocation_for_pointer",
        [ctypes.c_uint64, ctypes.POINTER(Allocation)],
        ctypes.c_uint32,
    )


def _configure_objects(library: Any) -> None:
    _signature(
        library,
        "shadowspill_pytorch_validate_object_binding",
        [ctypes.c_uint32, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64],
        ctypes.c_uint32,
    )


def _configure_task_boundaries(library: Any) -> None:
    _signature(
        library,
        "shadowspill_pytorch_allocation_scope_begin",
        [ctypes.c_uint64, ctypes.c_uint64],
        ctypes.c_uint32,
    )
    _signature(
        library,
        "shadowspill_pytorch_allocation_scope_end",
        [ctypes.c_uint64, ctypes.c_size_t],
        ctypes.c_uint32,
    )
    _signature(library, "shadowspill_pytorch_allocation_scope_abort", [], None)
    _signature(
        library,
        "shadowspill_pytorch_abort_task_handle",
        [ctypes.c_size_t],
        ctypes.c_uint32,
    )


def _configure_execution(library: Any) -> None:
    _signature(
        library,
        "shadowspill_pytorch_transfer_acquired_object_to_caller",
        [
            ctypes.c_size_t,
            ctypes.c_uint32,
            ctypes.c_size_t,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.POINTER(Allocation),
        ],
        ctypes.c_uint32,
    )
    _signature(
        library,
        "shadowspill_pytorch_before_task_handle",
        [
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.POINTER(ObjectBinding)),
            ctypes.POINTER(ctypes.c_uint32),
        ],
        ctypes.c_uint32,
    )
    _signature(
        library,
        "shadowspill_pytorch_after_task_handle",
        [ctypes.c_size_t, ctypes.c_size_t],
        ctypes.c_uint32,
    )
