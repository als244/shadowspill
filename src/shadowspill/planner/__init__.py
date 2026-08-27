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
    RecomputationChoiceDiagnostic,
    RecomputationProblemDiagnostics,
    ReductionStep,
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
    "RecomputationChoiceDiagnostic",
    "RecomputationProblemDiagnostics",
    "ReductionStep",
    "StorageHandoff",
    "TaskAdmissionSpec",
    "TaskAllocationStep",
    "TaskAllocationStepKind",
    "plan_program",
    "pressurefit",
    "pressurefit_program",
    "validate_schedule_feasibility",
]
