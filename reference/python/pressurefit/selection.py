"""Simulator-backed PressureFit candidate selection."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass

from shadowspill.ir import MemorySchedule, Program, RecomputationSelection
from shadowspill.planner._capi import (
    CCandidateResult,
    CPlanCandidate,
    CSelectionResult,
    load_planner_library,
)
from shadowspill.simulator import SimulationConfig
from shadowspill.simulator._indexed import _project, _Projection


@dataclass(frozen=True, slots=True)
class SelectionCandidate:
    """One immutable schedule submitted to the compiled selector."""

    program: Program
    schedule: MemorySchedule
    selections: tuple[RecomputationSelection, ...]
    config: SimulationConfig
    candidate_id: int
    selection_id: int


@dataclass(frozen=True, slots=True)
class CompiledSelection:
    """Indexed result returned by the compiled selector."""

    selected_index: int
    selected_candidate_id: int
    selected_selection_id: int
    selected_makespan_ns: int
    valid_candidate_count: int
    candidate_results: tuple[tuple[bool, int, int], ...]


def select_compiled(
    candidates: tuple[SelectionCandidate, ...],
) -> CompiledSelection:
    """Simulator-verify candidates and select by makespan then input order."""

    if not candidates:
        raise ValueError("candidates must not be empty")
    projections: tuple[_Projection, ...] = tuple(
        _project(item.program, item.schedule, item.selections, item.config, None)
        for item in candidates
    )
    c_candidates = (CPlanCandidate * len(candidates))(
        *(
            CPlanCandidate(
                ctypes.pointer(projection.program),
                candidate.candidate_id,
                candidate.selection_id,
            )
            for candidate, projection in zip(candidates, projections, strict=True)
        )
    )
    candidate_results = (CCandidateResult * len(candidates))()
    result = CSelectionResult(
        candidate_results=candidate_results,
        candidate_result_capacity=len(candidates),
    )
    library = load_planner_library()
    status = int(
        library.shadowspill_select_plan(
            c_candidates,
            len(candidates),
            ctypes.byref(result),
        )
    )
    if status != 0:
        encoded = library.shadowspill_planner_status_string(status)
        message = encoded.decode("utf-8") if encoded else f"planner status {status}"
        raise ValueError(message)
    return CompiledSelection(
        selected_index=int(result.selected_index),
        selected_candidate_id=int(result.selected_candidate_id),
        selected_selection_id=int(result.selected_selection_id),
        selected_makespan_ns=int(result.selected_makespan_ns),
        valid_candidate_count=int(result.valid_candidate_count),
        candidate_results=tuple(
            (bool(item.valid), int(item.simulation_status), int(item.makespan_ns))
            for item in candidate_results[: result.candidate_result_count]
        ),
    )


__all__ = ["CompiledSelection", "SelectionCandidate", "select_compiled"]
