"""One evaluated candidate: what it was, and how it came out."""

from __future__ import annotations

from dataclasses import dataclass, field

from .counters import (
    PressureFitRepairDiagnostics,
    PressureFitWorkDiagnostics,
)
from .json import (
    _boolean,
    _mapping,
    _optional_integer,
    _optional_string,
    _parse_candidate_id,
    _string,
)


@dataclass(frozen=True, slots=True)
class CandidateDiagnostic:
    """One candidate policy evaluated in one recomputation problem.

    ``candidate_id`` identifies only the reusable policy: residency strategy,
    prefetch rule, and coalescing mode.  ``selection_id`` is the parent-problem
    reference and is deliberately not part of candidate-policy identity.
    """

    candidate_id: str
    selection_id: str
    status: str
    makespan_ns: int | None = None
    #: Places the accepted plan came up short of capacity and waited for
    #: room. Zero means the plan never waited for memory.
    capacity_violation_count: int = 0
    #: Layouts measured for this candidate, how many fitted, and how many
    #: times a plan gave back what it overran and was rebuilt.
    placements_attempted: int = 0
    placements_admitted: int = 0
    capacity_refinements: int = 0
    schedule_digest: str | None = None
    failure_kind: str | None = None
    failure_detail: str | None = None
    repairs: PressureFitRepairDiagnostics = field(
        default_factory=PressureFitRepairDiagnostics
    )
    work: PressureFitWorkDiagnostics = field(default_factory=PressureFitWorkDiagnostics)
    residency_strategy: str = field(init=False)
    prefetch_rule: str = field(init=False)
    coalesced: bool = field(init=False)

    def __post_init__(self) -> None:
        strategy, rule, coalesced = _parse_candidate_id(self.candidate_id)
        object.__setattr__(self, "residency_strategy", strategy)
        object.__setattr__(self, "prefetch_rule", rule)
        object.__setattr__(self, "coalesced", coalesced)

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_policy": {
                "candidate_id": self.candidate_id,
                "residency_strategy": self.residency_strategy,
                "prefetch_rule": self.prefetch_rule,
                "coalesced": self.coalesced,
            },
            "outcome": {
                "status": self.status,
                "makespan_ns": self.makespan_ns,
                "capacity_violation_count": self.capacity_violation_count,
                "placements_attempted": self.placements_attempted,
                "placements_admitted": self.placements_admitted,
                "capacity_refinements": self.capacity_refinements,
                "schedule_digest": self.schedule_digest,
                "failure_kind": self.failure_kind,
                "failure_detail": self.failure_detail,
            },
            "repairs": self.repairs.to_dict(),
            "work": self.work.to_dict(),
        }

    @classmethod
    def from_value(cls, value: object, path: str) -> CandidateDiagnostic:
        data = _mapping(value, path)
        policy = _mapping(data.get("candidate_policy"), f"{path}.candidate_policy")
        outcome = _mapping(data.get("outcome"), f"{path}.outcome")
        result = cls(
            candidate_id=_string(
                policy.get("candidate_id"),
                f"{path}.candidate_policy.candidate_id",
            ),
            selection_id="",
            status=_string(outcome.get("status"), f"{path}.outcome.status"),
            makespan_ns=_optional_integer(
                outcome.get("makespan_ns"), f"{path}.outcome.makespan_ns"
            ),
            capacity_violation_count=_optional_integer(
                outcome.get("capacity_violation_count"),
                f"{path}.outcome.capacity_violation_count",
            )
            or 0,
            placements_attempted=_optional_integer(
                outcome.get("placements_attempted"),
                f"{path}.outcome.placements_attempted",
            )
            or 0,
            placements_admitted=_optional_integer(
                outcome.get("placements_admitted"),
                f"{path}.outcome.placements_admitted",
            )
            or 0,
            capacity_refinements=_optional_integer(
                outcome.get("capacity_refinements"),
                f"{path}.outcome.capacity_refinements",
            )
            or 0,
            schedule_digest=_optional_string(
                outcome.get("schedule_digest"),
                f"{path}.outcome.schedule_digest",
            ),
            failure_kind=_optional_string(
                outcome.get("failure_kind"), f"{path}.outcome.failure_kind"
            ),
            failure_detail=_optional_string(
                outcome.get("failure_detail"), f"{path}.outcome.failure_detail"
            ),
            repairs=PressureFitRepairDiagnostics.from_value(
                data.get("repairs"), f"{path}.repairs"
            ),
            work=PressureFitWorkDiagnostics.from_value(
                data.get("work"), f"{path}.work"
            ),
        )
        declared_policy = {
            "residency_strategy": _string(
                policy.get("residency_strategy"),
                f"{path}.candidate_policy.residency_strategy",
            ),
            "prefetch_rule": _string(
                policy.get("prefetch_rule"),
                f"{path}.candidate_policy.prefetch_rule",
            ),
            "coalesced": _boolean(
                policy.get("coalesced"), f"{path}.candidate_policy.coalesced"
            ),
        }
        expected_policy = {
            "residency_strategy": result.residency_strategy,
            "prefetch_rule": result.prefetch_rule,
            "coalesced": result.coalesced,
        }
        if declared_policy != expected_policy:
            raise ValueError(
                f"{path}.candidate_policy fields do not match candidate_id"
            )
        return result

    def with_selection_id(self, selection_id: str) -> CandidateDiagnostic:
        """Attach the containing recomputation problem after deserialization."""

        return CandidateDiagnostic(
            candidate_id=self.candidate_id,
            selection_id=selection_id,
            status=self.status,
            makespan_ns=self.makespan_ns,
            capacity_violation_count=self.capacity_violation_count,
            placements_attempted=self.placements_attempted,
            placements_admitted=self.placements_admitted,
            capacity_refinements=self.capacity_refinements,
            schedule_digest=self.schedule_digest,
            failure_kind=self.failure_kind,
            failure_detail=self.failure_detail,
            repairs=self.repairs,
            work=self.work,
        )
