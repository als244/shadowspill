"""Structured PressureFit search, repair, and work diagnostics.

One module per kind of record: what the search did (``counters``), each
candidate it evaluated (``candidates``), each resolution it
planned (``selections``), each capacity refinement it fell back to
(``refinement``), and all of it together (``summary``).
"""

from __future__ import annotations

from .candidates import CandidateDiagnostic
from .counters import (
    PlanningRepairDiagnostics,
    PlanningSectionTiming,
    PlanningWorkDiagnostics,
    ReductionStep,
)
from .resolved_programs import (
    INCUMBENT_CANDIDATE_ID,
    IncumbentDiagnostic,
    ResolvedProgramDiagnostics,
    TaskAlternativeChoiceDiagnostic,
)
from .summary import PlanningDiagnostics

__all__ = [
    "INCUMBENT_CANDIDATE_ID",
    "CandidateDiagnostic",
    "IncumbentDiagnostic",
    "PlanningDiagnostics",
    "PlanningRepairDiagnostics",
    "PlanningSectionTiming",
    "PlanningWorkDiagnostics",
    "ReductionStep",
    "ResolvedProgramDiagnostics",
    "TaskAlternativeChoiceDiagnostic",
]
