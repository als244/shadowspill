"""Framework-neutral memory and recomputation planning."""

from shadowspill.step import StepDataOrdering

from .admission import (
    AdmissionFacts,
    StorageHandoff,
    TaskAdmissionSpec,
    TaskAllocationStep,
    TaskAllocationStepKind,
)
from .diagnostics import (
    CandidateDiagnostic,
    GraphPairOutcome,
    PlanningDiagnostics,
    PlanningRepairDiagnostics,
    PlanningSectionTiming,
    PlanningWorkDiagnostics,
    ReductionStep,
    ResolvedProgramDiagnostics,
    TaskAlternativeChoiceDiagnostic,
)
from .plan import plan_program, summarize_plan, validate_schedule_feasibility
from .request import GenericPlanningOptions, OptionRecord
from .result import (
    ProgramPlanResult,
    ResidentSlice,
)
from .search import SearchAlgorithm, SearchOptions, answer_no_worse_than, toolkit
from .search.algorithms.pressurefit import pressurefit
from .search.toolkit import (
    DEFAULT_RESOLUTION_OPTIONS,
    NAMED_RESOLUTION_OPTIONS,
    CostedAlternatives,
    Resolution,
    named_resolution_options,
    resolutions,
    validate_resolution_options,
    validate_search_inputs,
)

__all__ = [
    "DEFAULT_RESOLUTION_OPTIONS",
    "NAMED_RESOLUTION_OPTIONS",
    "AdmissionFacts",
    "CandidateDiagnostic",
    "CostedAlternatives",
    "GenericPlanningOptions",
    "GraphPairOutcome",
    "OptionRecord",
    "PlanningDiagnostics",
    "PlanningRepairDiagnostics",
    "PlanningSectionTiming",
    "PlanningWorkDiagnostics",
    "ProgramPlanResult",
    "ReductionStep",
    "ResidentSlice",
    "Resolution",
    "ResolvedProgramDiagnostics",
    "SearchAlgorithm",
    "SearchOptions",
    "StepDataOrdering",
    "StorageHandoff",
    "TaskAdmissionSpec",
    "TaskAllocationStep",
    "TaskAllocationStepKind",
    "TaskAlternativeChoiceDiagnostic",
    "answer_no_worse_than",
    "named_resolution_options",
    "plan_program",
    "pressurefit",
    "resolutions",
    "summarize_plan",
    "toolkit",
    "validate_resolution_options",
    "validate_schedule_feasibility",
    "validate_search_inputs",
]
