"""Planning-cache results and format identifiers."""

from dataclasses import dataclass

from shadowspill.errors import PlanInfeasibleError, PlanSearchExhaustedError
from shadowspill.planner.admission.layout.model import FixedLayoutAdmission
from shadowspill.planner.diagnostics import GraphPairOutcome
from shadowspill.planner.diagnostics.plan_summary import PlanSummary
from shadowspill.planner.result import ProgramPlanResult
from shadowspill.schema import artifact_schema

SCHEMA = artifact_schema("plan_selection")
SUMMARY_SCHEMA = artifact_schema("plan_summary")
RESOLUTION_SCHEMA = artifact_schema("plan_resolution")


@dataclass(frozen=True, slots=True)
class PlanLookup:
    """Selected result and its optional fixed-layout certificate."""

    result: ProgramPlanResult
    from_store: bool
    key: str = ""
    certificate: FixedLayoutAdmission | None = None


@dataclass(frozen=True, slots=True)
class PlanSummaryLookup:
    """Certified makespan and selection diagnostics without the full plan."""

    key: str
    makespan_ns: int
    summary: PlanSummary
    graph_pair_outcomes: tuple[GraphPairOutcome, ...]
    answered_with_incumbent: bool


@dataclass(frozen=True, slots=True)
class Verdict:
    """A cached search refusal; environment/runtime failures are not cached."""

    outcome: str
    error: str
    kind: str | None
    message: str

    def exception(self) -> Exception:
        if self.outcome == "exhausted":
            return PlanSearchExhaustedError(self.message)
        return PlanInfeasibleError(
            self.message, kind=self.kind if self.kind is not None else self.error
        )
