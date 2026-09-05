"""Framework-neutral PressureFit memory and recomputation planning."""

from .admission import (
    AdmissionFacts,
    StorageHandoff,
    TaskAdmissionSpec,
    TaskAllocationStep,
    TaskAllocationStepKind,
)
from .diagnostics import (
    CandidateDiagnostic,
    PressureFitDiagnostics,
    PressureFitRepairDiagnostics,
    PressureFitSectionTiming,
    PressureFitWorkDiagnostics,
    ReductionStep,
    ResolvedProgramDiagnostics,
    TaskAlternativeChoiceDiagnostic,
)
from .plan import (
    plan_program,
    pressurefit,
    pressurefit_program,
    validate_schedule_feasibility,
)
from .request import InitialPlacement, PressureFitOptions
from .result import (
    PressureFitInfeasibleError,
    PressureFitResult,
    PressureFitSearchExhaustedError,
    ResidentSlice,
)

__all__ = [
    "AdmissionFacts",
    "CandidateDiagnostic",
    "InitialPlacement",
    "PressureFitDiagnostics",
    "PressureFitInfeasibleError",
    "PressureFitOptions",
    "PressureFitRepairDiagnostics",
    "PressureFitResult",
    "PressureFitSearchExhaustedError",
    "PressureFitSectionTiming",
    "PressureFitWorkDiagnostics",
    "ReductionStep",
    "ResidentSlice",
    "ResolvedProgramDiagnostics",
    "StorageHandoff",
    "TaskAdmissionSpec",
    "TaskAllocationStep",
    "TaskAllocationStepKind",
    "TaskAlternativeChoiceDiagnostic",
    "plan_program",
    "pressurefit",
    "pressurefit_program",
    "validate_schedule_feasibility",
]
