"""PressureFit's own ctypes surface, mirroring its public C header.

The generic half of the ABI is in :mod:`shadowspill.planner.capi`; this module
mirrors `<shadowspill/pressurefit/pressurefit.h>` and nothing else. A search
implemented elsewhere would have a module of this shape and would reuse the
generic one unchanged.
"""

from __future__ import annotations

import ctypes
from enum import IntEnum
from functools import cache

from shadowspill.libraries import load_shadowspill_library

from ....capi import CIndexedProblem, CIndexedSchedule


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


class CandidateStatus(IntEnum):
    """What became of one candidate, mirroring the C enum of the same shape.

    The values are on the wire in every stored diagnostic, so they are named
    here rather than written as integers at the two places that read them.
    """

    VALID = 0
    ANALYTIC_INFEASIBLE = 1
    SIMULATION_INFEASIBLE = 2
    ADMISSION_INFEASIBLE = 3
    INTERNAL_ERROR = 4
    REPAIR_EXHAUSTED = 5
    UNPLACEABLE = 6


def _check_struct_layout(library: ctypes.CDLL) -> None:
    """Refuse a library whose PressureFit structures are not these ones.

    A mirror that has drifted does not fail loudly: it reads one field where
    the library wrote another, and the result is corrupted counters rather than
    an error. The selector values continue the planner's own enum, so one
    library call answers for both halves.
    """

    mirrored = (
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
                f"{name} does not match the compiled search: "
                f"library {expected} bytes, mirror {actual}"
            )


@cache
def pressurefit_api() -> ctypes.CDLL:
    """The loaded library with PressureFit's own calls bound."""

    library = load_shadowspill_library()
    library.shadowspill_planner_struct_size.argtypes = [ctypes.c_uint32]
    library.shadowspill_planner_struct_size.restype = ctypes.c_uint64
    _check_struct_layout(library)
    library.shadowspill_pressurefit_search.argtypes = [
        ctypes.POINTER(CIndexedProblem),
        ctypes.c_uint32,
        ctypes.POINTER(CPressureFitOptions),
        ctypes.POINTER(CPressureFitResult),
    ]
    library.shadowspill_pressurefit_search.restype = ctypes.c_uint32
    library.shadowspill_pressurefit_preflight.argtypes = [
        ctypes.POINTER(CIndexedProblem),
        ctypes.POINTER(CPressureFitPreflightResult),
    ]
    library.shadowspill_pressurefit_preflight.restype = ctypes.c_uint32
    library.shadowspill_pressurefit_result_destroy.argtypes = [
        ctypes.POINTER(CPressureFitResult),
    ]
    library.shadowspill_pressurefit_result_destroy.restype = None
    library.shadowspill_pressurefit_best_placed_create.argtypes = []
    library.shadowspill_pressurefit_best_placed_create.restype = ctypes.c_void_p
    library.shadowspill_pressurefit_best_placed_destroy.argtypes = [ctypes.c_void_p]
    library.shadowspill_pressurefit_best_placed_destroy.restype = None
    library.shadowspill_pressurefit_best_placed_read.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(CPressureFitBestPlacedRecord),
    ]
    library.shadowspill_pressurefit_best_placed_read.restype = None
    return library


__all__ = [
    "CPressureFitBestPlacedRecord",
    "CPressureFitCandidateDiagnostic",
    "CPressureFitOptions",
    "CPressureFitPreflightResult",
    "CPressureFitReductionStep",
    "CPressureFitRepairDiagnostics",
    "CPressureFitResult",
    "CPressureFitSectionTiming",
    "CPressureFitWorkDiagnostics",
    "CandidateStatus",
    "pressurefit_api",
]
