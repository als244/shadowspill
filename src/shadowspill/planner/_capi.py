"""Declarative ctypes surface for the versioned planner selection ABI."""

from __future__ import annotations

import ctypes
from functools import cache
from pathlib import Path

from shadowspill._libraries import resolve_library
from shadowspill.simulator._capi import CProgram

ABI_VERSION = 6
NO_INDEX = (1 << 32) - 1


class CPlanCandidate(ctypes.Structure):
    _fields_ = [
        ("program", ctypes.POINTER(CProgram)),
        ("candidate_id", ctypes.c_uint32),
        ("selection_id", ctypes.c_uint32),
    ]


class CCandidateResult(ctypes.Structure):
    _fields_ = [
        ("simulation_status", ctypes.c_uint32),
        ("valid", ctypes.c_uint8),
        ("makespan_ns", ctypes.c_uint64),
    ]


class CSelectionResult(ctypes.Structure):
    _fields_ = [
        ("status", ctypes.c_uint32),
        ("selected_index", ctypes.c_uint32),
        ("selected_candidate_id", ctypes.c_uint32),
        ("selected_selection_id", ctypes.c_uint32),
        ("valid_candidate_count", ctypes.c_uint32),
        ("first_failure_index", ctypes.c_uint32),
        ("first_failure_status", ctypes.c_uint32),
        ("selected_makespan_ns", ctypes.c_uint64),
        ("candidate_results", ctypes.POINTER(CCandidateResult)),
        ("candidate_result_capacity", ctypes.c_uint32),
        ("candidate_result_count", ctypes.c_uint32),
    ]


class CResidencyProblem(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("alias_count", ctypes.c_uint32),
        ("boundary_count", ctypes.c_uint32),
        ("device_count", ctypes.c_uint32),
        ("alias_size_bytes", ctypes.POINTER(ctypes.c_uint64)),
        ("alias_device", ctypes.POINTER(ctypes.c_uint32)),
        ("alias_retain_spill_copy", ctypes.POINTER(ctypes.c_uint8)),
        ("initial_location", ctypes.POINTER(ctypes.c_int8)),
        ("final_location", ctypes.POINTER(ctypes.c_int8)),
        ("anchors", ctypes.POINTER(ctypes.c_uint8)),
        ("productions", ctypes.POINTER(ctypes.c_uint8)),
        ("latest_access_task", ctypes.POINTER(ctypes.c_uint32)),
        ("output_reservations", ctypes.POINTER(ctypes.c_uint8)),
        ("write_prefix", ctypes.POINTER(ctypes.c_uint8)),
        ("first_input_task", ctypes.POINTER(ctypes.c_uint32)),
        ("fetch_runtime_ns", ctypes.POINTER(ctypes.c_uint64)),
        ("evict_runtime_ns", ctypes.POINTER(ctypes.c_uint64)),
        ("task_ideal_end_ns", ctypes.POINTER(ctypes.c_uint64)),
        ("device_capacity_bytes", ctypes.POINTER(ctypes.c_uint64)),
        ("device_priority", ctypes.POINTER(ctypes.c_uint32)),
    ]


class CResidencyOptions(ctypes.Structure):
    _fields_ = [
        ("minimize_transfer", ctypes.c_uint8),
        ("prefetch_headroom", ctypes.c_uint8),
        ("seed_resident", ctypes.POINTER(ctypes.c_uint8)),
        ("seed_breaks", ctypes.POINTER(ctypes.c_uint8)),
        ("extra_pressure_bytes", ctypes.POINTER(ctypes.c_uint64)),
    ]


class CResidencyResult(ctypes.Structure):
    _fields_ = [
        ("status", ctypes.c_uint32),
        ("error_device", ctypes.c_uint32),
        ("error_boundary", ctypes.c_int32),
        ("required_bytes", ctypes.c_uint64),
        ("capacity_bytes", ctypes.c_uint64),
        ("resident", ctypes.POINTER(ctypes.c_uint8)),
        ("resident_capacity", ctypes.c_uint64),
        ("breaks", ctypes.POINTER(ctypes.c_uint8)),
        ("break_capacity", ctypes.c_uint64),
    ]


class CDenseSchedule(ctypes.Structure):
    _fields_ = [
        ("action_count", ctypes.c_uint32),
        ("action_trigger_tasks", ctypes.POINTER(ctypes.c_uint32)),
        ("action_aliases", ctypes.POINTER(ctypes.c_uint32)),
        ("action_kinds", ctypes.POINTER(ctypes.c_uint8)),
        ("initial_count", ctypes.c_uint32),
        ("initial_aliases", ctypes.POINTER(ctypes.c_uint32)),
        ("initial_locations", ctypes.POINTER(ctypes.c_uint8)),
        ("final_count", ctypes.c_uint32),
        ("final_aliases", ctypes.POINTER(ctypes.c_uint32)),
        ("final_locations", ctypes.POINTER(ctypes.c_uint8)),
    ]


class CAdmissionTopology(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("task_count", ctypes.c_uint32),
        ("alias_count", ctypes.c_uint32),
        ("pool_capacity_bytes", ctypes.c_uint64),
        ("object_capacity_bytes", ctypes.c_uint64),
        ("minimum_alignment", ctypes.c_uint64),
        ("task_workspace_bytes", ctypes.POINTER(ctypes.c_uint64)),
        ("fresh_output_offsets", ctypes.POINTER(ctypes.c_uint32)),
        ("fresh_output_aliases", ctypes.POINTER(ctypes.c_uint32)),
        ("replacement_offsets", ctypes.POINTER(ctypes.c_uint32)),
        ("replacement_aliases", ctypes.POINTER(ctypes.c_uint32)),
        ("handoff_offsets", ctypes.POINTER(ctypes.c_uint32)),
        ("handoff_source_aliases", ctypes.POINTER(ctypes.c_uint32)),
        ("handoff_destination_aliases", ctypes.POINTER(ctypes.c_uint32)),
    ]


class CPressureFitContext(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("residency", ctypes.POINTER(CResidencyProblem)),
        ("simulation", ctypes.POINTER(CProgram)),
        ("seed_resident", ctypes.POINTER(ctypes.c_uint8)),
        ("seed_breaks", ctypes.POINTER(ctypes.c_uint8)),
        ("admission", ctypes.POINTER(CAdmissionTopology)),
        ("alias_json_names", ctypes.POINTER(ctypes.c_char_p)),
        ("task_json_names", ctypes.POINTER(ctypes.c_char_p)),
    ]


class CPressureFitContextOptions(ctypes.Structure):
    _fields_ = [
        ("residency_strategies", ctypes.POINTER(ctypes.c_uint8)),
        ("residency_strategy_count", ctypes.c_uint32),
        ("prefetch_rules", ctypes.POINTER(ctypes.c_uint8)),
        ("prefetch_rule_count", ctypes.c_uint32),
        ("evaluate_coalesced", ctypes.c_uint8),
        ("max_repair_attempts", ctypes.c_uint32),
        ("initial_placement", ctypes.c_uint8),
    ]


class CPressureFitProgramContext(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("simulation", ctypes.POINTER(CProgram)),
        ("device_priority", ctypes.POINTER(ctypes.c_uint32)),
        ("admission", ctypes.POINTER(CAdmissionTopology)),
        ("alias_json_names", ctypes.POINTER(ctypes.c_char_p)),
        ("task_json_names", ctypes.POINTER(ctypes.c_char_p)),
    ]


class CPressureFitCandidateDiagnostic(ctypes.Structure):
    _fields_ = [
        ("status", ctypes.c_uint8),
        ("residency_strategy", ctypes.c_uint8),
        ("prefetch_rule", ctypes.c_uint8),
        ("coalesced", ctypes.c_uint8),
        ("repair_attempts", ctypes.c_uint32),
        ("simulation_status", ctypes.c_uint32),
        ("makespan_ns", ctypes.c_uint64),
        ("schedule_digest", ctypes.c_uint8 * 32),
        ("error_task", ctypes.c_uint32),
        ("error_alias", ctypes.c_uint32),
        ("error_device", ctypes.c_uint32),
        ("error_location", ctypes.c_uint8),
        ("error_boundary", ctypes.c_int32),
        ("error_time_ns", ctypes.c_uint64),
        ("error_capacity_bytes", ctypes.c_uint64),
        ("error_used_bytes", ctypes.c_uint64),
        ("error_requested_bytes", ctypes.c_uint64),
        ("error_required_bytes", ctypes.c_uint64),
    ]


class CPressureFitContextResult(ctypes.Structure):
    _fields_ = [
        ("status", ctypes.c_uint32),
        ("selected_candidate_index", ctypes.c_uint32),
        ("selected_makespan_ns", ctypes.c_uint64),
        ("selected_schedule", CDenseSchedule),
        ("candidates", ctypes.POINTER(CPressureFitCandidateDiagnostic)),
        ("candidate_count", ctypes.c_uint32),
        ("residency_cache_hits", ctypes.c_uint64),
        ("residency_cache_misses", ctypes.c_uint64),
        ("schedule_emissions", ctypes.c_uint64),
        ("schedule_cache_hits", ctypes.c_uint64),
        ("simulation_calls", ctypes.c_uint64),
        ("simulation_cache_hits", ctypes.c_uint64),
        ("admission_calls", ctypes.c_uint64),
        ("residency_time_ns", ctypes.c_uint64),
        ("schedule_time_ns", ctypes.c_uint64),
        ("simulation_time_ns", ctypes.c_uint64),
        ("admission_time_ns", ctypes.c_uint64),
        ("digest_time_ns", ctypes.c_uint64),
    ]


class CScheduleAdmissionResult(ctypes.Structure):
    """Caller-owned buffers for one exact dense-schedule admission."""

    _fields_ = [
        ("status", ctypes.c_uint32),
        ("decision_digest", ctypes.c_uint64),
        ("peak_allocated_bytes", ctypes.c_uint64),
        ("peak_reserved_bytes", ctypes.c_uint64),
        ("peak_fragmentation_bytes", ctypes.c_uint64),
        ("error_operation_index", ctypes.c_uint64),
        ("error_requested_bytes", ctypes.c_uint64),
        ("error_free_bytes", ctypes.c_uint64),
        ("error_largest_free_range_bytes", ctypes.c_uint64),
        ("initial_physical_bytes", ctypes.c_uint64),
        ("task_start_deltas", ctypes.POINTER(ctypes.c_int64)),
        ("task_completion_deltas", ctypes.POINTER(ctypes.c_int64)),
        ("task_capacity", ctypes.c_uint32),
        ("action_trigger_deltas", ctypes.POINTER(ctypes.c_int64)),
        ("action_completion_deltas", ctypes.POINTER(ctypes.c_int64)),
        ("action_capacity", ctypes.c_uint32),
        ("reuse_predecessor_actions", ctypes.POINTER(ctypes.c_uint32)),
        ("reuse_successor_tasks", ctypes.POINTER(ctypes.c_uint32)),
        ("reuse_successor_actions", ctypes.POINTER(ctypes.c_uint32)),
        ("reuse_capacity", ctypes.c_uint32),
        ("reuse_count", ctypes.c_uint32),
    ]


def planner_library_path() -> Path | None:
    """Return the selected planner library without loading it."""

    return resolve_library("libshadowspill_planner.so")


@cache
def load_planner_library() -> ctypes.CDLL:
    """Load and validate ``libshadowspill_planner.so`` exactly once."""

    path = planner_library_path()
    if path is None:
        raise RuntimeError(
            "libshadowspill_planner.so was not found; install ShadowSpill or "
            "build the editable checkout at its configured build location"
        )
    library = ctypes.CDLL(str(path))
    library.shadowspill_planner_abi_version.argtypes = []
    library.shadowspill_planner_abi_version.restype = ctypes.c_uint32
    found = int(library.shadowspill_planner_abi_version())
    if found != ABI_VERSION:
        raise RuntimeError(
            f"planner ABI mismatch: Python expects {ABI_VERSION}, library has {found}"
        )
    library.shadowspill_select_plan.argtypes = [
        ctypes.POINTER(CPlanCandidate),
        ctypes.c_uint32,
        ctypes.POINTER(CSelectionResult),
    ]
    library.shadowspill_select_plan.restype = ctypes.c_uint32
    library.shadowspill_reduce_residency.argtypes = [
        ctypes.POINTER(CResidencyProblem),
        ctypes.POINTER(CResidencyOptions),
        ctypes.POINTER(CResidencyResult),
    ]
    library.shadowspill_reduce_residency.restype = ctypes.c_uint32
    library.shadowspill_evaluate_pressurefit_context.argtypes = [
        ctypes.POINTER(CPressureFitContext),
        ctypes.POINTER(CPressureFitContextOptions),
        ctypes.POINTER(CPressureFitContextResult),
    ]
    library.shadowspill_evaluate_pressurefit_context.restype = ctypes.c_uint32
    library.shadowspill_evaluate_pressurefit_program_context.argtypes = [
        ctypes.POINTER(CPressureFitProgramContext),
        ctypes.POINTER(CPressureFitContextOptions),
        ctypes.POINTER(CPressureFitContextResult),
    ]
    library.shadowspill_evaluate_pressurefit_program_context.restype = ctypes.c_uint32
    library.shadowspill_evaluate_schedule_admission.argtypes = [
        ctypes.POINTER(CProgram),
        ctypes.POINTER(CAdmissionTopology),
        ctypes.POINTER(CDenseSchedule),
        ctypes.POINTER(CScheduleAdmissionResult),
    ]
    library.shadowspill_evaluate_schedule_admission.restype = ctypes.c_uint32
    library.shadowspill_pressurefit_context_result_destroy.argtypes = [
        ctypes.POINTER(CPressureFitContextResult),
    ]
    library.shadowspill_pressurefit_context_result_destroy.restype = None
    library.shadowspill_planner_status_string.argtypes = [ctypes.c_uint32]
    library.shadowspill_planner_status_string.restype = ctypes.c_char_p
    return library


__all__ = [
    "ABI_VERSION",
    "NO_INDEX",
    "CAdmissionTopology",
    "CCandidateResult",
    "CDenseSchedule",
    "CPlanCandidate",
    "CPressureFitCandidateDiagnostic",
    "CPressureFitContext",
    "CPressureFitContextOptions",
    "CPressureFitContextResult",
    "CPressureFitProgramContext",
    "CResidencyOptions",
    "CResidencyProblem",
    "CResidencyResult",
    "CScheduleAdmissionResult",
    "CSelectionResult",
    "load_planner_library",
    "planner_library_path",
]
