"""Structured PressureFit search, repair, and work diagnostics.

One module per kind of record: what the search did (``counters``), each
candidate it evaluated (``candidates``), each recomputation selection it
planned (``selections``), each capacity refinement it fell back to
(``refinement``), and all of it together (``summary``).
"""

from __future__ import annotations

from .candidates import CandidateDiagnostic
from .counters import (
    PressureFitRepairDiagnostics,
    PressureFitSectionTiming,
    PressureFitWorkDiagnostics,
    ReductionStep,
)
from .selections import (
    RecomputationChoiceDiagnostic,
    RecomputationProblemDiagnostics,
)
from .summary import PressureFitDiagnostics

__all__ = [
    "CandidateDiagnostic",
    "PressureFitDiagnostics",
    "PressureFitRepairDiagnostics",
    "PressureFitSectionTiming",
    "PressureFitWorkDiagnostics",
    "RecomputationChoiceDiagnostic",
    "RecomputationProblemDiagnostics",
    "ReductionStep",
]
