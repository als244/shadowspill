"""One evaluated candidate: what it was, and how it came out."""

from __future__ import annotations

from dataclasses import dataclass, field

from .counters import (
    PressureFitRepairDiagnostics,
    PressureFitWorkDiagnostics,
    ReductionStep,
)
from .json import (
    _boolean,
    _mapping,
    _optional_integer,
    _optional_string,
    _parse_candidate_id,
    _span,
    _string,
)


def _steps(value: object, path: str) -> tuple[ReductionStep, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{path} must be a list")
    return tuple(
        ReductionStep.from_value(step, f"{path}[{index}]")
        for index, step in enumerate(value)
    )


@dataclass(frozen=True, slots=True)
class CandidateDiagnostic:
    """One candidate policy evaluated in one recomputation problem.

    ``candidate_id`` identifies only the reusable policy: residency strategy,
    fetch rule, and coalescing mode.  ``selection_id`` is the parent-problem
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
    #: Repairs spent when the plan the candidate answers with was placed;
    #: ``None`` when it placed none.
    repairs_at_best: int | None = None
    #: When this candidate ran, in nanoseconds from the start of the call that
    #: evaluated it. ``work.sections`` is work done; these are wall clock, so
    #: two candidates ran at the same time exactly when their spans overlap.
    started_ns: int = 0
    finished_ns: int = 0
    schedule_digest: str | None = None
    failure_kind: str | None = None
    failure_detail: str | None = None
    repairs: PressureFitRepairDiagnostics = field(
        default_factory=PressureFitRepairDiagnostics
    )
    work: PressureFitWorkDiagnostics = field(default_factory=PressureFitWorkDiagnostics)
    #: Every plan this candidate held, in the order it held them. Empty
    #: unless the caller asked for a trajectory.
    steps: tuple[ReductionStep, ...] = ()
    residency_strategy: str = field(init=False)
    fetch_rule: str = field(init=False)
    coalesced: bool = field(init=False)

    def __post_init__(self) -> None:
        strategy, rule, coalesced = _parse_candidate_id(self.candidate_id)
        object.__setattr__(self, "residency_strategy", strategy)
        object.__setattr__(self, "fetch_rule", rule)
        object.__setattr__(self, "coalesced", coalesced)

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_policy": {
                "candidate_id": self.candidate_id,
                "residency_strategy": self.residency_strategy,
                "fetch_rule": self.fetch_rule,
                "coalesced": self.coalesced,
            },
            "outcome": {
                "status": self.status,
                "makespan_ns": self.makespan_ns,
                "capacity_violation_count": self.capacity_violation_count,
                "placements_attempted": self.placements_attempted,
                "placements_admitted": self.placements_admitted,
                "capacity_refinements": self.capacity_refinements,
                "repairs_at_best": self.repairs_at_best,
                "schedule_digest": self.schedule_digest,
                "failure_kind": self.failure_kind,
                "failure_detail": self.failure_detail,
            },
            "repairs": self.repairs.to_dict(),
            "work": self.work.to_dict(),
            "span": {
                "started_ns": self.started_ns,
                "finished_ns": self.finished_ns,
            },
            "steps": [step.to_dict() for step in self.steps],
        }

    @classmethod
    def from_value(
        cls, value: object, path: str, selection_id: str
    ) -> CandidateDiagnostic:
        """Read one candidate. The parent supplies ``selection_id``, which is
        a reference to the containing problem rather than part of the
        candidate's own record, so it is not in the payload."""

        data = _mapping(value, path)
        policy = _mapping(data.get("candidate_policy"), f"{path}.candidate_policy")
        outcome = _mapping(data.get("outcome"), f"{path}.outcome")
        result = cls(
            candidate_id=_string(
                policy.get("candidate_id"),
                f"{path}.candidate_policy.candidate_id",
            ),
            selection_id=selection_id,
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
            repairs_at_best=_optional_integer(
                outcome.get("repairs_at_best"),
                f"{path}.outcome.repairs_at_best",
            ),
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
            started_ns=_span(data.get("span"), "started_ns", f"{path}.span"),
            finished_ns=_span(data.get("span"), "finished_ns", f"{path}.span"),
            steps=_steps(data.get("steps", []), f"{path}.steps"),
        )
        declared_policy = {
            "residency_strategy": _string(
                policy.get("residency_strategy"),
                f"{path}.candidate_policy.residency_strategy",
            ),
            "fetch_rule": _string(
                policy.get("fetch_rule"),
                f"{path}.candidate_policy.fetch_rule",
            ),
            "coalesced": _boolean(
                policy.get("coalesced"), f"{path}.candidate_policy.coalesced"
            ),
        }
        expected_policy = {
            "residency_strategy": result.residency_strategy,
            "fetch_rule": result.fetch_rule,
            "coalesced": result.coalesced,
        }
        if declared_policy != expected_policy:
            raise ValueError(
                f"{path}.candidate_policy fields do not match candidate_id"
            )
        return result
