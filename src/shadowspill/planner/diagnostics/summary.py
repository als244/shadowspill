"""Everything one PressureFit call did, as one record."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
from typing import ClassVar

from shadowspill.schema import artifact_schema

from .candidates import CandidateDiagnostic
from .counters import (
    PlanningRepairDiagnostics,
    PlanningWorkDiagnostics,
    _nonnegative,
)
from .json import (
    _integer,
    _list,
    _mapping,
    _optional_integer,
    _string,
    without_measurements,
)
from .resolved_programs import (
    INCUMBENT_CANDIDATE_ID,
    ResolvedProgramDiagnostics,
)


@dataclass(frozen=True, slots=True)
class PlanningDiagnostics:
    """What a search did, and what it answered with.

    The record has two halves, and the split is what lets a second search
    arrive without this type changing. The **generic** half is what any
    search must say: which plan won, and at what cost. The **search** half
    is whatever that search has to report about how it got there, written
    under its own name and read by nothing here.

    Everything below `search` today is PressureFit's -- resolved programs,
    repairs, capacity refinement, section timings -- because PressureFit is
    the only search. The contract is deliberately loose: a search reports
    what it has, and a reader that does not recognise the name reads the
    generic half and stops.
    """

    SCHEMA: ClassVar[str] = artifact_schema("search_diagnostics")

    selected_candidate_id: str
    selected_selection_id: str
    selected_makespan_ns: int
    resolved_programs: tuple[ResolvedProgramDiagnostics, ...]
    work: PlanningWorkDiagnostics = field(default_factory=PlanningWorkDiagnostics)
    effective_object_capacity_bytes: int | None = None
    #: Which search produced this. The key its own half is written under.
    search: str = "pressurefit"
    #: How many threads the search was given. Recorded for visibility and
    #: excluded from anything compared across runs, because it changes how
    #: long an answer took and not which answer was right.
    workers: int = 0

    def __post_init__(self) -> None:
        problem_ids = tuple(item.selection_id for item in self.resolved_programs)
        if len(problem_ids) != len(set(problem_ids)):
            raise ValueError("resolution appears more than once")
        selected_problem = tuple(
            problem
            for problem in self.resolved_programs
            if problem.selection_id == self.selected_selection_id
        )
        if len(selected_problem) != 1:
            raise ValueError("selected resolved program is not unique")
        problem = selected_problem[0]
        if problem.selected_candidate_id != self.selected_candidate_id:
            raise ValueError("global and problem candidate selections disagree")
        if problem.selected_makespan_ns != self.selected_makespan_ns:
            raise ValueError("global and problem selected makespans disagree")

    def _candidate_evaluations(self) -> tuple[CandidateDiagnostic, ...]:
        return tuple(
            candidate
            for problem in self.resolved_programs
            for candidate in problem.candidate_evaluations
        )

    @property
    def resolved_program_count(self) -> int:
        return len(self.resolved_programs)

    @property
    def candidate_policy_count(self) -> int:
        return len({item.candidate_id for item in self._candidate_evaluations()})

    @property
    def candidate_evaluation_count(self) -> int:
        return len(self._candidate_evaluations())

    @property
    def valid_candidate_evaluation_count(self) -> int:
        return sum(item.status == "valid" for item in self._candidate_evaluations())

    @property
    def valid_resolved_program_count(self) -> int:
        return sum(
            problem.selected_candidate_id is not None
            for problem in self.resolved_programs
        )

    @property
    def repairs(self) -> PlanningRepairDiagnostics:
        result = PlanningRepairDiagnostics()
        for candidate in self._candidate_evaluations():
            result += candidate.repairs
        return result

    @property
    def candidate_status_counts(self) -> dict[str, int]:
        return dict(
            sorted(
                Counter(item.status for item in self._candidate_evaluations()).items()
            )
        )

    def replace_selected_makespan(self, makespan_ns: int) -> PlanningDiagnostics:
        """Replace the selected policy's admission-aware timing consistently."""

        _nonnegative("makespan_ns", makespan_ns)
        problems = tuple(
            replace(
                problem,
                selected_makespan_ns=makespan_ns,
                candidate_evaluations=tuple(
                    replace(candidate, makespan_ns=makespan_ns)
                    if candidate.candidate_id == self.selected_candidate_id
                    else candidate
                    for candidate in problem.candidate_evaluations
                ),
                incumbent=(
                    replace(problem.incumbent, makespan_ns=makespan_ns)
                    if problem.incumbent is not None
                    and self.selected_candidate_id == INCUMBENT_CANDIDATE_ID
                    else problem.incumbent
                ),
            )
            if problem.selection_id == self.selected_selection_id
            else problem
            for problem in self.resolved_programs
        )
        return replace(
            self,
            selected_makespan_ns=makespan_ns,
            resolved_programs=problems,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA,
            # The generic half: which plan won, and which search found it.
            # A reader that knows no search at all can read this much.
            "search_name": self.search,
            "workers": self.workers,
            "selection": {
                "candidate_id": self.selected_candidate_id,
                "selection_id": self.selected_selection_id,
                "makespan_ns": self.selected_makespan_ns,
            },
            # The search's own half, under its name. Nothing generic reads
            # inside it, so a search may put whatever it has here.
            "search": {"summary": {
                "resolved_program_count": self.resolved_program_count,
                "valid_resolved_program_count": (
                    self.valid_resolved_program_count
                ),
                "candidate_policy_count": self.candidate_policy_count,
                "candidate_evaluation_count": self.candidate_evaluation_count,
                "valid_candidate_evaluation_count": (
                    self.valid_candidate_evaluation_count
                ),
                "candidate_status_counts": self.candidate_status_counts,
            },
            "work": self.work.to_dict(),
            "repairs": self.repairs.to_dict(),
            "capacity_refinement": {
                "effective_object_capacity_bytes": (
                    self.effective_object_capacity_bytes
                ),
            },
            "resolved_programs": [
                item.to_dict() for item in self.resolved_programs
            ],
            },
        }

    def stable_dict(self) -> dict[str, object]:
        """Return deterministic search evidence without measured work times."""

        value = without_measurements(self.to_dict())
        assert isinstance(value, dict)
        return value

    @classmethod
    def from_value(cls, value: object, path: str) -> PlanningDiagnostics:
        data = _mapping(value, path)
        if data.get("schema") != cls.SCHEMA:
            raise ValueError(f"{path}.schema: unsupported schema")
        selection = _mapping(data.get("selection"), f"{path}.selection")
        search = _mapping(data.get("search"), f"{path}.search")
        summary = _mapping(search.get("summary"), f"{path}.search.summary")
        refinement = _mapping(
            search.get("capacity_refinement"), f"{path}.search.capacity_refinement"
        )
        problems = tuple(
            ResolvedProgramDiagnostics.from_value(
                item, f"{path}.search.resolved_programs[{index}]"
            )
            for index, item in enumerate(
                _list(
                    search.get("resolved_programs"),
                    f"{path}.search.resolved_programs",
                )
            )
        )
        result = cls(
            selected_candidate_id=_string(
                selection.get("candidate_id"), f"{path}.selection.candidate_id"
            ),
            selected_selection_id=_string(
                selection.get("selection_id"), f"{path}.selection.selection_id"
            ),
            selected_makespan_ns=_integer(
                selection.get("makespan_ns"), f"{path}.selection.makespan_ns"
            ),
            resolved_programs=problems,
            work=PlanningWorkDiagnostics.from_value(
                search.get("work"), f"{path}.search.work"
            ),
            search=_string(data.get("search_name"), f"{path}.search_name"),
            workers=_integer(data.get("workers", 0), f"{path}.workers"),
            effective_object_capacity_bytes=_optional_integer(
                refinement.get("effective_object_capacity_bytes"),
                f"{path}.search.capacity_refinement"
                ".effective_object_capacity_bytes",
            ),
        )
        declared_repairs = PlanningRepairDiagnostics.from_value(
            search.get("repairs"), f"{path}.search.repairs"
        )
        if declared_repairs != result.repairs:
            raise ValueError(f"{path}.search.repairs does not match candidate repairs")
        expected_summary = {
            "resolved_program_count": result.resolved_program_count,
            "valid_resolved_program_count": (result.valid_resolved_program_count),
            "candidate_policy_count": result.candidate_policy_count,
            "candidate_evaluation_count": result.candidate_evaluation_count,
            "valid_candidate_evaluation_count": (
                result.valid_candidate_evaluation_count
            ),
            "candidate_status_counts": result.candidate_status_counts,
        }
        for name, expected in expected_summary.items():
            if summary.get(name) != expected:
                raise ValueError(f"{path}.summary.{name} does not reconcile")
        return result
