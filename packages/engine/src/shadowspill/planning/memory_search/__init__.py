"""Search schedules and object residency within device and spill budgets."""

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
    "validate_schedule_feasibility": ".plan",
    "GenericPlanningOptions": ".request",
    "OptionRecord": ".request",
    "ProgramPlanResult": ".result",
    "ResidentSlice": ".result",
    "SearchAlgorithm": ".algorithm",
    "SearchOptions": ".algorithm",
    "answer_no_worse_than": ".algorithm",
    "toolkit": ".toolkit",
    "pressurefit": ".algorithms.pressurefit",
    "DEFAULT_RESOLUTION_OPTIONS": ".toolkit",
    "NAMED_RESOLUTION_OPTIONS": ".toolkit",
    "CostedAlternatives": ".toolkit",
    "Resolution": ".toolkit",
    "named_resolution_options": ".toolkit",
    "resolutions": ".toolkit",
    "validate_resolution_options": ".toolkit",
    "validate_tail_save_counts": ".toolkit",
    "validate_search_inputs": ".toolkit",
}


def __getattr__(name: str) -> Any:
    if name not in _PUBLIC:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_PUBLIC[name], __name__)
    value = module if name == "toolkit" else getattr(module, name)
    globals()[name] = value
    return value


__all__ = list(_PUBLIC)
