"""Planning-cache results and format identifiers."""

from dataclasses import dataclass

from shadowspill.errors import PlanInfeasibleError, PlanSearchExhaustedError
from shadowspill.planning.memory_search.admission.layout.model import (
    FixedLayoutAdmission,
)
from shadowspill.planning.memory_search.result import ProgramPlanResult
from shadowspill.schema import artifact_schema

SCHEMA = artifact_schema("plan_selection")
RESOLUTION_SCHEMA = artifact_schema("plan_resolution")


@dataclass(frozen=True, slots=True)
class PlanLookup:
    """Selected result and its optional fixed-layout certificate."""

    result: ProgramPlanResult
    from_store: bool
    key: str = ""
    certificate: FixedLayoutAdmission | None = None


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
