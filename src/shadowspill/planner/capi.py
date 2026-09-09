"""Declarative ctypes surface for the versioned planner selection ABI."""

from __future__ import annotations

import ctypes
from functools import cache

from shadowspill.libraries import (
    load_shadowspill_library,
)
from shadowspill.simulator.capi import (
    CProgram,
    CTaskInterval,
    CTransferInterval,
)

NO_INDEX = (1 << 32) - 1


class CIndexedSchedule(ctypes.Structure):
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


class CAdmissionFacts(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("task_count", ctypes.c_uint32),
        ("alias_count", ctypes.c_uint32),
        ("pool_capacity_bytes", ctypes.c_uint64),
        ("object_capacity_bytes", ctypes.c_uint64),
        ("minimum_alignment", ctypes.c_uint64),
        ("task_workspace_offsets", ctypes.POINTER(ctypes.c_uint32)),
        ("task_workspace_extent_bytes", ctypes.POINTER(ctypes.c_uint64)),
        ("fresh_output_offsets", ctypes.POINTER(ctypes.c_uint32)),
        ("fresh_output_aliases", ctypes.POINTER(ctypes.c_uint32)),
        ("replacement_offsets", ctypes.POINTER(ctypes.c_uint32)),
        ("replacement_aliases", ctypes.POINTER(ctypes.c_uint32)),
        ("handoff_offsets", ctypes.POINTER(ctypes.c_uint32)),
        ("handoff_source_aliases", ctypes.POINTER(ctypes.c_uint32)),
        ("handoff_destination_aliases", ctypes.POINTER(ctypes.c_uint32)),
        ("allocation_slot_count", ctypes.c_uint32),
        ("task_allocation_offsets", ctypes.POINTER(ctypes.c_uint32)),
        ("task_allocation_slots", ctypes.POINTER(ctypes.c_uint32)),
        ("task_allocation_bytes", ctypes.POINTER(ctypes.c_uint64)),
        ("task_allocation_aliases", ctypes.POINTER(ctypes.c_uint32)),
        ("task_allocation_kinds", ctypes.POINTER(ctypes.c_uint8)),
    ]


class CAdmissionOperations(ctypes.Structure):
    _fields_ = [
        ("lease_ids", ctypes.POINTER(ctypes.c_uint64)),
        ("dependency_ids", ctypes.POINTER(ctypes.c_uint64)),
        ("bytes", ctypes.POINTER(ctypes.c_uint64)),
        ("alignments", ctypes.POINTER(ctypes.c_uint64)),
        ("kinds", ctypes.POINTER(ctypes.c_uint8)),
        ("purposes", ctypes.POINTER(ctypes.c_uint8)),
        ("boundaries", ctypes.POINTER(ctypes.c_uint8)),
        ("indices", ctypes.POINTER(ctypes.c_uint32)),
        ("allocation_offsets", ctypes.POINTER(ctypes.c_uint32)),
        ("operation_capacity", ctypes.c_uint64),
        ("lease_aliases", ctypes.POINTER(ctypes.c_uint32)),
        ("lease_starts", ctypes.POINTER(ctypes.c_uint64)),
        ("lease_retires", ctypes.POINTER(ctypes.c_uint64)),
        ("lease_capacity", ctypes.c_uint64),
        ("operation_count", ctypes.c_uint64),
        ("lease_count", ctypes.c_uint64),
        ("dependency_count", ctypes.c_uint64),
        ("fetch_bytes", ctypes.c_uint64),
        ("evict_bytes", ctypes.c_uint64),
    ]


class CLeaseLifetime(ctypes.Structure):
    """One lease to place. The interval is half-open."""

    _fields_ = [
        ("bytes", ctypes.c_uint64),
        ("alignment", ctypes.c_uint64),
        ("start_ns", ctypes.c_uint64),
        ("end_ns", ctypes.c_uint64),
    ]


class CLeaseIdentity(ctypes.Structure):
    """Everything about a lease except when it is live, all as indices."""

    _fields_ = [
        ("lease_id", ctypes.c_uint64),
        ("causal_start", ctypes.c_uint64),
        ("causal_end", ctypes.c_uint64),
        ("task", ctypes.c_uint32),
        ("alias", ctypes.c_uint32),
        ("action", ctypes.c_uint32),
        ("purpose", ctypes.c_uint8),
    ]


class CLeaseLifetimeProblem(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("operations", ctypes.POINTER(CAdmissionOperations)),
        ("admission", ctypes.POINTER(CAdmissionFacts)),
        ("schedule", ctypes.POINTER(CIndexedSchedule)),
        ("task_intervals", ctypes.POINTER(CTaskInterval)),
        ("task_interval_count", ctypes.c_uint32),
        ("transfer_intervals", ctypes.POINTER(CTransferInterval)),
        ("transfer_interval_count", ctypes.c_uint32),
        ("makespan_ns", ctypes.c_uint64),
        ("dynamic_aliases", ctypes.POINTER(ctypes.c_uint32)),
        ("dynamic_alias_count", ctypes.c_uint32),
    ]


class CLeaseLifetimeResult(ctypes.Structure):
    _fields_ = [
        ("lifetimes", ctypes.POINTER(CLeaseLifetime)),
        ("identities", ctypes.POINTER(CLeaseIdentity)),
        ("allocation_step_leases", ctypes.POINTER(ctypes.c_uint64)),
        ("alias_leases", ctypes.POINTER(ctypes.c_uint64)),
        ("lifetime_count", ctypes.c_uint64),
        ("fixed_count", ctypes.c_uint64),
    ]


class CPlacementProblem(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("lifetime_count", ctypes.c_uint32),
        ("lifetimes", ctypes.POINTER(CLeaseLifetime)),
        ("excluded", ctypes.POINTER(ctypes.c_uint8)),
    ]


class CPlacementResult(ctypes.Structure):
    _fields_ = [
        ("required_bytes", ctypes.c_uint64),
        ("offsets", ctypes.POINTER(ctypes.c_uint64)),
    ]


class CPressureFitBestPlacedRecord(ctypes.Structure):
    _fields_ = [
        ("makespan_ns", ctypes.c_uint64),
        ("object_capacity_bytes", ctypes.c_uint64),
        ("capacity_given_back_bytes", ctypes.c_uint64),
        ("residency_strategy", ctypes.c_uint8),
        ("fetch_rule", ctypes.c_uint8),
        ("coalesced", ctypes.c_uint8),
        ("schedule_digest", ctypes.c_uint8 * 32),
    ]


class CPressureFitOptions(ctypes.Structure):
    _fields_ = [
        ("residency_strategies", ctypes.POINTER(ctypes.c_uint8)),
        ("residency_strategy_count", ctypes.c_uint32),
        ("fetch_rules", ctypes.POINTER(ctypes.c_uint8)),
        ("fetch_rule_count", ctypes.c_uint32),
        ("coalescing_modes", ctypes.POINTER(ctypes.c_uint8)),
        ("coalescing_mode_count", ctypes.c_uint32),
        ("max_repair_attempts", ctypes.c_uint32),
        ("initial_placement", ctypes.c_uint8),
        ("capacity_refinement_bytes", ctypes.c_uint64),
        ("record_reduction_steps", ctypes.c_uint8),
        ("best_placed", ctypes.c_void_p),
        ("workers", ctypes.c_uint32),
        ("deterministic", ctypes.c_uint8),
        ("split_write_backs", ctypes.c_uint8),
        ("minimum_object_bytes_evict_eligible", ctypes.c_uint64),
    ]


class CScheduleContext(ctypes.Structure):
    """The part of a problem that is not about how it is searched."""

    _fields_ = [
        ("simulation", ctypes.POINTER(CProgram)),
        ("admission", ctypes.POINTER(CAdmissionFacts)),
        ("placement", ctypes.POINTER(CAdmissionFacts)),
        ("alias_json_names", ctypes.POINTER(ctypes.c_char_p)),
        ("task_json_names", ctypes.POINTER(ctypes.c_char_p)),
    ]


class CIndexedProblem(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("context", CScheduleContext),
        ("device_priority", ctypes.POINTER(ctypes.c_uint32)),
        ("incumbent", ctypes.POINTER(CIndexedSchedule)),
    ]


class CPressureFitPreflightResult(ctypes.Structure):
    _fields_ = [
        ("status", ctypes.c_uint32),
        ("failure_kind", ctypes.c_uint8),
        ("error_device", ctypes.c_uint32),
        ("error_alias", ctypes.c_uint32),
        ("error_boundary", ctypes.c_int32),
        ("required_bytes", ctypes.c_uint64),
        ("capacity_bytes", ctypes.c_uint64),
    ]


class CPressureFitRepairDiagnostics(ctypes.Structure):
    _fields_ = [
        ("admission_fetch_advance_attempts", ctypes.c_uint64),
        ("admission_fetch_delay_attempts", ctypes.c_uint64),
        ("admission_pressure_boundary_attempts", ctypes.c_uint64),
        ("simulation_fetch_delay_attempts", ctypes.c_uint64),
        ("simulation_pressure_boundary_attempts", ctypes.c_uint64),
    ]


class CPressureFitReductionStep(ctypes.Structure):
    _fields_ = [
        ("makespan_ns", ctypes.c_uint64),
        ("required_bytes", ctypes.c_uint64),
        ("capacity_bytes", ctypes.c_uint64),
        ("cut_offset", ctypes.c_uint32),
        ("cut_count", ctypes.c_uint32),
        ("repairs", ctypes.c_uint32),
        ("simulation_status", ctypes.c_uint32),
        ("capacity_violations", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
    ]


class CPressureFitSectionTiming(ctypes.Structure):
    _fields_ = [
        ("total_ns", ctypes.c_uint64),
        ("prepare_ns", ctypes.c_uint64),
        ("setup_ns", ctypes.c_uint64),
        ("reduce_ns", ctypes.c_uint64),
        ("emit_ns", ctypes.c_uint64),
        ("simulate_ns", ctypes.c_uint64),
        ("repair_ns", ctypes.c_uint64),
        ("digest_ns", ctypes.c_uint64),
        ("place_ns", ctypes.c_uint64),
        ("select_ns", ctypes.c_uint64),
        ("teardown_ns", ctypes.c_uint64),
        ("admit_ns", ctypes.c_uint64),
        ("residual_ns", ctypes.c_uint64),
    ]


class CPressureFitWorkDiagnostics(ctypes.Structure):
    _fields_ = [
        ("schedule_emissions", ctypes.c_uint64),
        ("schedule_cache_hits", ctypes.c_uint64),
        ("simulation_calls", ctypes.c_uint64),
        ("simulation_cache_hits", ctypes.c_uint64),
        ("admission_calls", ctypes.c_uint64),
        ("sections", CPressureFitSectionTiming),
    ]


class CPressureFitCandidateDiagnostic(ctypes.Structure):
    _fields_ = [
        ("status", ctypes.c_uint8),
        ("residency_strategy", ctypes.c_uint8),
        ("fetch_rule", ctypes.c_uint8),
        ("coalesced", ctypes.c_uint8),
        ("repairs", CPressureFitRepairDiagnostics),
        ("work", CPressureFitWorkDiagnostics),
        ("simulation_status", ctypes.c_uint32),
        ("makespan_ns", ctypes.c_uint64),
        ("steps", ctypes.POINTER(CPressureFitReductionStep)),
        ("step_count", ctypes.c_uint32),
        ("step_capacity", ctypes.c_uint32),
        ("cut_aliases", ctypes.POINTER(ctypes.c_uint32)),
        ("cut_count", ctypes.c_uint32),
        ("cut_capacity", ctypes.c_uint32),
        ("capacity_violation_count", ctypes.c_uint32),
        ("placements_attempted", ctypes.c_uint32),
        ("placements_admitted", ctypes.c_uint32),
        ("capacity_refinements", ctypes.c_uint32),
        ("repairs_at_best", ctypes.c_uint32),
        ("pressure_escalations", ctypes.c_uint32),
        ("escalations_taken_back", ctypes.c_uint32),
        ("schedule_digest", ctypes.c_uint8 * 32),
        ("started_ns", ctypes.c_uint64),
        ("finished_ns", ctypes.c_uint64),
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


class CPressureFitResult(ctypes.Structure):
    _fields_ = [
        ("status", ctypes.c_uint32),
        ("selected_candidate_index", ctypes.c_uint32),
        ("selected_makespan_ns", ctypes.c_uint64),
        ("selected_schedule", CIndexedSchedule),
        ("candidates", ctypes.POINTER(CPressureFitCandidateDiagnostic)),
        ("candidate_count", ctypes.c_uint32),
        ("repairs", CPressureFitRepairDiagnostics),
        ("work", CPressureFitWorkDiagnostics),
        ("started_ns", ctypes.c_uint64),
        ("finished_ns", ctypes.c_uint64),
        ("evict_ineligible_aliases", ctypes.c_uint32),
        ("evict_ineligible_bytes", ctypes.c_uint64),
        ("resident_slice_bytes", ctypes.POINTER(ctypes.c_uint64)),
        ("alias_evict_eligible", ctypes.POINTER(ctypes.c_uint8)),
        ("incumbent_given", ctypes.c_uint8),
        ("incumbent_status", ctypes.c_uint8),
        ("incumbent_selected", ctypes.c_uint8),
        ("incumbent_makespan_ns", ctypes.c_uint64),
        ("incumbent_required_bytes", ctypes.c_uint64),
    ]


class CScheduleAdmissionResult(ctypes.Structure):
    """Caller-owned buffers for one exact indexed-schedule admission."""

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


def _check_struct_layout(library: ctypes.CDLL) -> None:
    """Refuse a library whose structures are not the ones mirrored here.

    A mirror that has drifted does not fail loudly: it reads one field where
    the library wrote another, and the result is corrupted counters rather
    than an error. Comparing sizes catches the drift at load, where it can
    still be understood.
    """

    mirrored = (
        (0, "CAdmissionFacts", CAdmissionFacts),
        (1, "CPressureFitOptions", CPressureFitOptions),
        (2, "CPressureFitWorkDiagnostics", CPressureFitWorkDiagnostics),
        (3, "CPressureFitCandidateDiagnostic", CPressureFitCandidateDiagnostic),
        (4, "CPressureFitSectionTiming", CPressureFitSectionTiming),
        (5, "CPressureFitReductionStep", CPressureFitReductionStep),
        (6, "CPressureFitBestPlacedRecord", CPressureFitBestPlacedRecord),
        (7, "CPressureFitResult", CPressureFitResult),
    )
    for which, name, structure in mirrored:
        expected = library.shadowspill_planner_struct_size(which)
        actual = ctypes.sizeof(structure)
        if expected and expected != actual:
            raise RuntimeError(
                f"{name} does not match the compiled planner: "
                f"library {expected} bytes, mirror {actual}"
            )


@cache
def planner_api() -> ctypes.CDLL:
    library = load_shadowspill_library()
    library.shadowspill_pressurefit_search.argtypes = [
        ctypes.POINTER(CIndexedProblem),
        ctypes.c_uint32,
        ctypes.POINTER(CPressureFitOptions),
        ctypes.POINTER(CPressureFitResult),
    ]
    library.shadowspill_pressurefit_search.restype = ctypes.c_uint32
    library.shadowspill_planner_struct_size.argtypes = [ctypes.c_uint32]
    library.shadowspill_planner_struct_size.restype = ctypes.c_uint64
    _check_struct_layout(library)
    library.shadowspill_pressurefit_best_placed_create.argtypes = []
    library.shadowspill_pressurefit_best_placed_create.restype = ctypes.c_void_p
    library.shadowspill_pressurefit_best_placed_destroy.argtypes = [ctypes.c_void_p]
    library.shadowspill_pressurefit_best_placed_destroy.restype = None
    library.shadowspill_pressurefit_best_placed_read.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(CPressureFitBestPlacedRecord),
    ]
    library.shadowspill_pressurefit_best_placed_read.restype = None
    library.shadowspill_pressurefit_preflight.argtypes = [
        ctypes.POINTER(CIndexedProblem),
        ctypes.POINTER(CPressureFitPreflightResult),
    ]
    library.shadowspill_pressurefit_preflight.restype = ctypes.c_uint32
    library.shadowspill_evaluate_schedule_admission.argtypes = [
        ctypes.POINTER(CProgram),
        ctypes.POINTER(CAdmissionFacts),
        ctypes.POINTER(CIndexedSchedule),
        ctypes.POINTER(CScheduleAdmissionResult),
    ]
    library.shadowspill_evaluate_schedule_admission.restype = ctypes.c_uint32
    library.shadowspill_pressurefit_result_destroy.argtypes = [
        ctypes.POINTER(CPressureFitResult),
    ]
    library.shadowspill_pressurefit_result_destroy.restype = None
    library.shadowspill_admission_operation_bounds.argtypes = [
        ctypes.POINTER(CProgram),
        ctypes.POINTER(CAdmissionFacts),
        ctypes.POINTER(CIndexedSchedule),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
    ]
    library.shadowspill_admission_operation_bounds.restype = ctypes.c_uint32
    library.shadowspill_build_admission_operations.argtypes = [
        ctypes.POINTER(CProgram),
        ctypes.POINTER(CAdmissionFacts),
        ctypes.POINTER(CIndexedSchedule),
        ctypes.POINTER(CAdmissionOperations),
    ]
    library.shadowspill_build_admission_operations.restype = ctypes.c_uint32
    library.shadowspill_place_lifetimes.argtypes = [
        ctypes.POINTER(CPlacementProblem),
        ctypes.POINTER(CPlacementResult),
    ]
    library.shadowspill_place_lifetimes.restype = ctypes.c_uint32
    library.shadowspill_build_lease_lifetimes.argtypes = [
        ctypes.POINTER(CLeaseLifetimeProblem),
        ctypes.POINTER(CLeaseLifetimeResult),
    ]
    library.shadowspill_build_lease_lifetimes.restype = ctypes.c_uint32
    library.shadowspill_status_string.argtypes = [ctypes.c_uint32]
    library.shadowspill_status_string.restype = ctypes.c_char_p
    return library


__all__ = [
    "NO_INDEX",
    "CAdmissionFacts",
    "CAdmissionOperations",
    "CIndexedProblem",
    "CIndexedSchedule",
    "CLeaseIdentity",
    "CLeaseLifetime",
    "CLeaseLifetimeProblem",
    "CLeaseLifetimeResult",
    "CPlacementProblem",
    "CPlacementResult",
    "CPressureFitCandidateDiagnostic",
    "CPressureFitOptions",
    "CPressureFitPreflightResult",
    "CPressureFitRepairDiagnostics",
    "CPressureFitResult",
    "CPressureFitWorkDiagnostics",
    "CScheduleAdmissionResult",
    "planner_api",
]
