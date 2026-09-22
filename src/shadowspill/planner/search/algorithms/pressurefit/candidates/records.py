"""The C records one evaluated candidate and one resolved problem come back in."""

from dataclasses import dataclass

from shadowspill.ir import (
    MemoryActionKind,
    MemoryLocation,
)

from .....admission.indexing import IndexedMemorySchedule
from .....diagnostics import (
    PlanningRepairDiagnostics,
    PlanningSectionTiming,
    PlanningWorkDiagnostics,
    ReductionStep,
)

_STRATEGY_CODE = {
    "headroom-stall": 0,
    "headroom-transfer": 1,
    "tight-stall": 2,
    "tight-transfer": 3,
    "relaxed-stall": 4,
}
_RULE_CODE = {
    "packed-fifo": 0,
    "packed-fit": 1,
    "interval-entry": 2,
    "latest-safe": 3,
    "demand": 4,
}
_STRATEGY_NAME = {code: name for name, code in _STRATEGY_CODE.items()}
_RULE_NAME = {code: name for name, code in _RULE_CODE.items()}
_ACTION_KIND = {
    0: MemoryActionKind.RELEASE,
    1: MemoryActionKind.EVICT,
    2: MemoryActionKind.FETCH,
    3: MemoryActionKind.WRITE_BACK,
}
_LOCATION = {0: MemoryLocation.DEVICE, 1: MemoryLocation.SPILL}
_INITIAL_PLACEMENT = {"required": 0, "greedy": 1}
_PREFLIGHT_WORKSPACE_CAPACITY = 1
_PREFLIGHT_REQUIRED_CAPACITY = 2
_PREFLIGHT_RESIDENT_SLICE_CAPACITY = 4
_PREFLIGHT_MISSING_INITIAL_RESIDENCY = 3


@dataclass(frozen=True, slots=True)
class CCandidateDiagnostic:
    status: int
    strategy: str
    rule: str
    coalesced: bool
    repairs: PlanningRepairDiagnostics
    work: PlanningWorkDiagnostics
    simulation_status: int
    makespan_ns: int
    #: Places the accepted plan came up short of capacity and waited.
    capacity_violation_count: int
    #: What placing this candidate's plans cost, and what it bought.
    placements_attempted: int
    placements_admitted: int
    capacity_refinements: int
    #: Repairs spent when the plan the candidate answers with was placed;
    #: ``None`` when it placed none.
    repairs_at_best: int | None
    #: Pressure repairs that asked for more than the shortfall because the
    #: same failure had repeated, and how many of those no cut could meet.
    pressure_escalations: int
    escalations_taken_back: int
    #: When this candidate ran, in nanoseconds from the start of the call that
    #: evaluated it. ``work.sections`` is work done; these are wall clock, so
    #: two candidates ran at once exactly when their spans overlap. Both are
    #: zero for a candidate no worker reached.
    started_ns: int
    finished_ns: int
    #: Every plan this candidate held, in order. Empty unless the caller asked
    #: for a trajectory.
    steps: tuple[ReductionStep, ...]
    schedule_digest: str | None
    error_task: int
    error_alias: int
    error_device: int
    error_location: int
    error_boundary: int
    error_time_ns: int
    error_capacity_bytes: int
    error_used_bytes: int
    error_requested_bytes: int
    error_required_bytes: int
    #: The fastest plan that simulated but whose layout did not fit the
    #: pool, and how many such plans were measured; ``None`` and 0 when none.
    best_unplaced_makespan_ns: int | None
    unplaced_plans: int

    @property
    def candidate_id(self) -> str:
        suffix = "-coalesced" if self.coalesced else ""
        return f"{self.strategy}/{self.rule}{suffix}"

    @property
    def repair_attempts(self) -> int:
        return self.repairs.total_attempts


@dataclass(frozen=True, slots=True)
class CIncumbentOutcome:
    """What became of the plan to beat, as the library reported it.

    `status` is a candidate status code: valid, unplaceable, or the
    infeasibility that stopped it. `makespan_ns` is what it simulated to at
    this problem's capacity, zero if it never simulated; `required_bytes`
    the pool its layout needed, zero if it was never measured.
    """

    status: int
    makespan_ns: int
    required_bytes: int
    selected: bool


@dataclass(frozen=True, slots=True)
class CProblemResult:
    selected_candidate_index: int | None
    selected_makespan_ns: int | None
    selected_schedule: IndexedMemorySchedule | None
    candidates: tuple[CCandidateDiagnostic, ...]
    repairs: PlanningRepairDiagnostics
    work: PlanningWorkDiagnostics
    #: This problem's span on the same clock its candidates use: from the first
    #: candidate a worker started to the last one it finished. With several
    #: problems in one call these overlap, because workers take whatever task
    #: is next rather than finishing a problem first.
    started_ns: int
    finished_ns: int
    #: The objects `minimum_object_bytes_evict_eligible` kept resident: how
    #: many, their bytes, the resident slice reserved for them, and which
    #: they are, by alias index.
    evict_ineligible_aliases: int
    evict_ineligible_bytes: int
    resident_slice_bytes: int
    resident_aliases: tuple[int, ...]
    #: The plan to beat's outcome, when the problem carried one. A selected
    #: incumbent is the answer with no candidate index: `selected_makespan_ns`
    #: and `selected_schedule` are its own.
    incumbent: CIncumbentOutcome | None = None

    def __post_init__(self) -> None:
        repairs = PlanningRepairDiagnostics()
        candidate_work = PlanningWorkDiagnostics()
        for candidate in self.candidates:
            repairs += candidate.repairs
            candidate_work += candidate.work
        if repairs != self.repairs:
            raise RuntimeError("PressureFit problem repair counters do not reconcile")
        for name in candidate_work.__dataclass_fields__:
            if name == "sections":
                continue
            if getattr(candidate_work, name) > getattr(self.work, name):
                raise RuntimeError(
                    f"PressureFit candidate work exceeds problem work: {name}"
                )
        # Every candidate section is a delta of the same workspace counter the
        # problem totals, so a candidate can never hold more of one than the
        # problem it ran inside.
        for name in PlanningSectionTiming.__dataclass_fields__:
            if getattr(candidate_work.sections, name) > getattr(
                self.work.sections, name
            ):
                raise RuntimeError(
                    f"PressureFit candidate sections exceed problem sections: {name}"
                )


@dataclass(frozen=True, slots=True)
class CPreflightResult:
    """Structured semantic-feasibility result from the planner."""

    failure_kind: str | None
    error_device: int | None
    error_alias: int | None
    error_boundary: int | None
    required_bytes: int | None
    capacity_bytes: int | None

    @property
    def valid(self) -> bool:
        return self.failure_kind is None


class ProblemPreparationError(RuntimeError):
    """The facts could not be normalized into planner facts."""
