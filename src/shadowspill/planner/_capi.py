"""Declarative ctypes surface for the versioned planner selection ABI."""

from __future__ import annotations

import ctypes
import os
from functools import cache
from pathlib import Path

from shadowspill.simulator._capi import CProgram

ABI_VERSION = 1
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
    library.shadowspill_planner_status_string.argtypes = [ctypes.c_uint32]
    library.shadowspill_planner_status_string.restype = ctypes.c_char_p
    return library


__all__ = [
    "ABI_VERSION",
    "NO_INDEX",
    "CCandidateResult",
    "CPlanCandidate",
    "CSelectionResult",
    "load_planner_library",
    "planner_library_path",
]
