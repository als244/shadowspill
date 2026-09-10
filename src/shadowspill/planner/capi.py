"""The planner's own ctypes surface, mirroring <shadowspill/planner.h>.

Nothing here belongs to a search. The structures and calls are the ones any
search's answer is certified and placed with; the search that ships mirrors its
own header in its own module beside this one.
"""

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

    mirrored = ((0, "CAdmissionFacts", CAdmissionFacts),)
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
    library.shadowspill_planner_struct_size.argtypes = [ctypes.c_uint32]
    library.shadowspill_planner_struct_size.restype = ctypes.c_uint64
    _check_struct_layout(library)
    library.shadowspill_evaluate_schedule_admission.argtypes = [
        ctypes.POINTER(CProgram),
        ctypes.POINTER(CAdmissionFacts),
        ctypes.POINTER(CIndexedSchedule),
        ctypes.POINTER(CScheduleAdmissionResult),
    ]
    library.shadowspill_evaluate_schedule_admission.restype = ctypes.c_uint32
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
    "CScheduleAdmissionResult",
    "planner_api",
]
