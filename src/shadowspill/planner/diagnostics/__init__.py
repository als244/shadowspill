"""Structured search, repair, and work diagnostics.

One module per kind of record: what the search did (``counters``), each
candidate it evaluated (``candidates``), each resolved program it planned
(``resolved_programs``), all of it together (``summary``), and what each
graph-pair selection cost, read off a finished result (``graph_pairs``).
"""

from __future__ import annotations

from .candidates import CandidateDiagnostic
from .counters import (
    PlanningRepairDiagnostics,
    PlanningSectionTiming,
    PlanningWorkDiagnostics,
    ReductionStep,
)
from .graph_pairs import AlternativeCosts, GraphPairOutcome, graph_pair_outcomes
from .resolved_programs import (
    INCUMBENT_CANDIDATE_ID,
    IncumbentDiagnostic,
    ResolvedProgramDiagnostics,
    TaskAlternativeChoiceDiagnostic,
)
from .summary import PlanningDiagnostics

__all__ = [
    "INCUMBENT_CANDIDATE_ID",
    "AlternativeCosts",
    "CandidateDiagnostic",
    "GraphPairOutcome",
    "IncumbentDiagnostic",
    "PlanningDiagnostics",
    "PlanningRepairDiagnostics",
    "PlanningSectionTiming",
    "PlanningWorkDiagnostics",
    "ReductionStep",
    "ResolvedProgramDiagnostics",
    "TaskAlternativeChoiceDiagnostic",
    "graph_pair_outcomes",
]
