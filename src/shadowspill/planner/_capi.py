"""Declarative ctypes surface for the versioned planner selection ABI."""

from __future__ import annotations

import ctypes
import os
from functools import cache
from pathlib import Path

from shadowspill.simulator._capi import CProgram

ABI_VERSION = 2
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
        ("alias_retain_host", ctypes.POINTER(ctypes.c_uint8)),
        ("initial_location", ctypes.POINTER(ctypes.c_int8)),
        ("final_location", ctypes.POINTER(ctypes.c_int8)),
        ("anchors", ctypes.POINTER(ctypes.c_uint8)),
        ("productions", ctypes.POINTER(ctypes.c_uint8)),
        ("latest_access_task", ctypes.POINTER(ctypes.c_uint32)),
        ("output_reservations", ctypes.POINTER(ctypes.c_uint8)),
        ("write_prefix", ctypes.POINTER(ctypes.c_uint8)),
        ("first_input_task", ctypes.POINTER(ctypes.c_uint32)),
        ("h2d_runtime_ns", ctypes.POINTER(ctypes.c_uint64)),
        ("d2h_runtime_ns", ctypes.POINTER(ctypes.c_uint64)),
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


def _library_candidates() -> tuple[Path, ...]:
    explicit = os.environ.get("SHADOWSPILL_PLANNER_LIBRARY")
    packaged = Path(__file__).resolve().parents[1] / "lib/libshadowspill_planner.so"
    if explicit:
        return (Path(explicit).expanduser().resolve(), packaged)
    return (packaged,)


def planner_library_path() -> Path | None:
    """Return the selected planner library without loading it."""

    for candidate in _library_candidates():
        if candidate.is_file():
            return candidate
    return None


@cache
def load_planner_library() -> ctypes.CDLL:
    """Load and validate ``libshadowspill_planner.so`` exactly once."""

    path = planner_library_path()
    if path is None:
        raise RuntimeError(
            "libshadowspill_planner.so is not installed; install the built "
            "ShadowSpill wheel or set SHADOWSPILL_PLANNER_LIBRARY"
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
    library.shadowspill_planner_status_string.argtypes = [ctypes.c_uint32]
    library.shadowspill_planner_status_string.restype = ctypes.c_char_p
    return library


__all__ = [
    "ABI_VERSION",
    "NO_INDEX",
    "CCandidateResult",
    "CPlanCandidate",
    "CResidencyOptions",
    "CResidencyProblem",
    "CResidencyResult",
    "CSelectionResult",
    "load_planner_library",
    "planner_library_path",
]
