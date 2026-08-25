"""Everything one PressureFit call did, as one record."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
from typing import ClassVar

from .candidates import CandidateDiagnostic
from .counters import (
    PressureFitRepairDiagnostics,
    PressureFitWorkDiagnostics,
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
from .selections import (
    RecomputationProblemDiagnostics,
)


@dataclass(frozen=True, slots=True)
class PressureFitDiagnostics:
    """Complete problem, policy-evaluation, and aggregate PressureFit evidence."""

    SCHEMA: ClassVar[str] = "shadowspill.pressurefit_diagnostics/v2"

    selected_candidate_id: str
    selected_selection_id: str
    selected_makespan_ns: int
    recomputation_problems: tuple[RecomputationProblemDiagnostics, ...]
    work: PressureFitWorkDiagnostics = field(default_factory=PressureFitWorkDiagnostics)
    effective_object_capacity_bytes: int | None = None

    def __post_init__(self) -> None:
        problem_ids = tuple(item.selection_id for item in self.recomputation_problems)
        if len(problem_ids) != len(set(problem_ids)):
            raise ValueError("recomputation selection appears more than once")
        selected_problem = tuple(
            problem
            for problem in self.recomputation_problems
            if problem.selection_id == self.selected_selection_id
        )
        if len(selected_problem) != 1:
            raise ValueError("selected recomputation problem is not unique")
        problem = selected_problem[0]
        if problem.selected_candidate_id != self.selected_candidate_id:
            raise ValueError("global and problem candidate selections disagree")
        if problem.selected_makespan_ns != self.selected_makespan_ns:
            raise ValueError("global and problem selected makespans disagree")

    def _candidate_evaluations(self) -> tuple[CandidateDiagnostic, ...]:
        return tuple(
            candidate
            for problem in self.recomputation_problems
            for candidate in problem.candidate_evaluations
        )

    @property
    def recomputation_problem_count(self) -> int:
        return len(self.recomputation_problems)

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
    def valid_recomputation_problem_count(self) -> int:
        return sum(
            problem.selected_candidate_id is not None
            for problem in self.recomputation_problems
        )

    @property
    def repairs(self) -> PressureFitRepairDiagnostics:
        result = PressureFitRepairDiagnostics()
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

    def with_selected_makespan(self, makespan_ns: int) -> PressureFitDiagnostics:
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
            )
            if problem.selection_id == self.selected_selection_id
            else problem
            for problem in self.recomputation_problems
        )
        return replace(
            self,
            selected_makespan_ns=makespan_ns,
            recomputation_problems=problems,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA,
            "selection": {
                "candidate_id": self.selected_candidate_id,
                "selection_id": self.selected_selection_id,
                "makespan_ns": self.selected_makespan_ns,
            },
            "summary": {
                "recomputation_problem_count": self.recomputation_problem_count,
                "valid_recomputation_problem_count": (
                    self.valid_recomputation_problem_count
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
            "recomputation_problems": [
                item.to_dict() for item in self.recomputation_problems
            ],
        }

    def stable_dict(self) -> dict[str, object]:
        """Return deterministic search evidence without measured work times."""

        value = without_measurements(self.to_dict())
        assert isinstance(value, dict)
        return value

    @classmethod
    def from_value(cls, value: object, path: str) -> PressureFitDiagnostics:
        data = _mapping(value, path)
        if data.get("schema") != cls.SCHEMA:
            raise ValueError(f"{path}.schema: unsupported schema")
        selection = _mapping(data.get("selection"), f"{path}.selection")
        summary = _mapping(data.get("summary"), f"{path}.summary")
        refinement = _mapping(
            data.get("capacity_refinement"), f"{path}.capacity_refinement"
        )
        problems = tuple(
            RecomputationProblemDiagnostics.from_value(
                item, f"{path}.recomputation_problems[{index}]"
            )
            for index, item in enumerate(
                _list(
                    data.get("recomputation_problems"),
                    f"{path}.recomputation_problems",
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
            recomputation_problems=problems,
            work=PressureFitWorkDiagnostics.from_value(
                data.get("work"), f"{path}.work"
            ),
            effective_object_capacity_bytes=_optional_integer(
                refinement.get("effective_object_capacity_bytes"),
                f"{path}.capacity_refinement.effective_object_capacity_bytes",
            ),
        )
        declared_repairs = PressureFitRepairDiagnostics.from_value(
            data.get("repairs"), f"{path}.repairs"
        )
        if declared_repairs != result.repairs:
            raise ValueError(f"{path}.repairs does not match candidate repairs")
        expected_summary = {
            "recomputation_problem_count": result.recomputation_problem_count,
            "valid_recomputation_problem_count": (
                result.valid_recomputation_problem_count
            ),
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
