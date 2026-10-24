"""Phase exports, loaded only when requested during caller migration."""

from importlib import import_module
from typing import Any

_PUBLIC = {
    "StepDataOrdering": "shadowspill.step",
    "AdmissionFacts": ".admission",
    "StorageHandoff": ".admission",
    "TaskAdmissionSpec": ".admission",
    "TaskAllocationStep": ".admission",
    "TaskAllocationStepKind": ".admission",
    "CandidateDiagnostic": ".diagnostics",
    "GraphPairOutcome": ".diagnostics",
    "PlanningDiagnostics": ".diagnostics",
    "PlanningRepairDiagnostics": ".diagnostics",
    "PlanningSectionTiming": ".diagnostics",
    "PlanningWorkDiagnostics": ".diagnostics",
    "ReductionStep": ".diagnostics",
    "ResolvedProgramDiagnostics": ".diagnostics",
    "TaskAlternativeChoiceDiagnostic": ".diagnostics",
    "plan_program": ".plan",
    "summarize_plan": ".plan",
    "validate_schedule_feasibility": ".plan",
    "GenericPlanningOptions": ".request",
    "OptionRecord": ".request",
    "ProgramPlanResult": ".result",
    "ResidentSlice": ".result",
    "SearchAlgorithm": ".search",
    "SearchOptions": ".search",
    "answer_no_worse_than": ".search",
    "toolkit": ".search",
    "pressurefit": ".search.algorithms.pressurefit",
    "DEFAULT_RESOLUTION_OPTIONS": ".search.toolkit",
    "NAMED_RESOLUTION_OPTIONS": ".search.toolkit",
    "CostedAlternatives": ".search.toolkit",
    "Resolution": ".search.toolkit",
    "named_resolution_options": ".search.toolkit",
    "resolutions": ".search.toolkit",
    "validate_resolution_options": ".search.toolkit",
    "validate_search_inputs": ".search.toolkit",
}


def __getattr__(name: str) -> Any:
    if name not in _PUBLIC:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_PUBLIC[name], __name__), name)
    globals()[name] = value
    return value


__all__ = list(_PUBLIC)
