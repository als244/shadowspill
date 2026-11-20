"""Everything one search call did, as one record."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
from typing import ClassVar

from shadowspill.planning.memory_search.diagnostics.candidates import (
    CandidateDiagnostic,
)
from shadowspill.planning.memory_search.diagnostics.counters import (
    PlanningRepairDiagnostics,
    PlanningWorkDiagnostics,
    _nonnegative,
)
from shadowspill.planning.memory_search.diagnostics.json import (
    _integer,
    _list,
    _mapping,
    _optional_integer,
    _string,
    without_measurements,
)
from shadowspill.planning.memory_search.diagnostics.resolved_programs import (
    INCUMBENT_CANDIDATE_ID,
    ResolvedProgramDiagnostics,
)
from shadowspill.schema import artifact_schema


@dataclass(frozen=True, slots=True)
class PlanningDiagnostics:
    """What a search did, and what it answered with.

    The record has two halves. The generic half is what any search must
    say: which plan won, at what cost, and which search found it. The rest
    is how this search got there, which the written record keeps under
    ``search``; a reader that knows no search reads only the generic half.

    Parameters
    ----------
    selected_candidate_id : str
        The candidate policy the search answered with, or ``"incumbent"``
        when the plan to beat won.
    selected_selection_id : str
        The resolved program the answer came from.
    selected_makespan_ns : int
        The answer's makespan, in nanoseconds; once the layout is
        certified, the certificate's.
    resolved_programs : tuple of ResolvedProgramDiagnostics
        Every resolved program the search evaluated, each once; the selected
        one agrees with the three fields above.
    work : PlanningWorkDiagnostics, default PlanningWorkDiagnostics()
        The call's operation counts and section times.
    effective_object_capacity_bytes : int, optional
        The object capacity the search finally planned against, when it
        reports one.
    search : str, default ""
        The search's own ``name``; empty only on a record built by hand.
    workers : int, default 0
        How many threads the search was given. Recorded for visibility and
        left out of anything compared across runs.

    Attributes
    ----------
    SCHEMA : str
        The schema the record is written under.

    Raises
    ------
    ValueError
        If a resolved program appears twice, or the selected one is not unique
        or disagrees with the selection.
    """

    SCHEMA: ClassVar[str] = artifact_schema("search_diagnostics")

    selected_candidate_id: str
    selected_selection_id: str
    selected_makespan_ns: int
    resolved_programs: tuple[ResolvedProgramDiagnostics, ...]
    work: PlanningWorkDiagnostics = field(default_factory=PlanningWorkDiagnostics)
    effective_object_capacity_bytes: int | None = None
    #: Which search produced this, as that search names itself; serialized
    #: as ``search_name``. Empty only on a record built by hand.
    search: str = ""
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
        """How many resolved programs the search evaluated."""
        return len(self.resolved_programs)

    @property
    def candidate_policy_count(self) -> int:
        """How many distinct candidate policies it evaluated."""
        return len({item.candidate_id for item in self._candidate_evaluations()})

    @property
    def candidate_evaluation_count(self) -> int:
        """How many candidate evaluations it made, over every resolved program."""
        return len(self._candidate_evaluations())

    @property
    def valid_candidate_evaluation_count(self) -> int:
        """How many of those evaluations reached a plan."""
        return sum(item.status == "valid" for item in self._candidate_evaluations())

    @property
    def valid_resolved_program_count(self) -> int:
        """How many resolved programs have an answer."""
        return sum(
            problem.selected_candidate_id is not None
            for problem in self.resolved_programs
        )

    @property
    def repairs(self) -> PlanningRepairDiagnostics:
        """Every candidate's repairs, summed."""
        result = PlanningRepairDiagnostics()
        for candidate in self._candidate_evaluations():
            result += candidate.repairs
        return result

    @property
    def candidate_status_counts(self) -> dict[str, int]:
        """How many evaluations ended in each status, by status name."""
        return dict(
            sorted(
                Counter(item.status for item in self._candidate_evaluations()).items()
            )
        )

    def replace_selected_makespan(self, makespan_ns: int) -> PlanningDiagnostics:
        """Copy the record with the answer's makespan replaced everywhere it appears.

        The selected resolved program, its selected candidate, and its plan
        to beat when that won, all take the new makespan; the planner uses
        it to record the certificate's makespan.

        Parameters
        ----------
        makespan_ns : int
            The new makespan, in nanoseconds. At least zero.

        Returns
        -------
        PlanningDiagnostics
            The copy.

        Raises
        ------
        ValueError
            If ``makespan_ns`` is negative or not an integer.
        """

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
        """Write the record as a JSON-compatible mapping.

        Returns
        -------
        dict
            ``schema``, ``search_name``, ``workers`` and the ``selection``,
            then the search's own half under ``search``: the derived counts,
            work, repairs, capacity refinement and resolved programs.
        """
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
            "search": {
                "summary": {
                    "resolved_program_count": self.resolved_program_count,
                    "valid_resolved_program_count": (self.valid_resolved_program_count),
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
        """Write the record without anything that measures the run.

        Section times, spans, wall times and ``workers`` are left out, so two
        runs that found the same plan write the same mapping.

        Returns
        -------
        dict
            `PlanningDiagnostics.to_dict` without those fields.
        """

        value = without_measurements(self.to_dict())
        assert isinstance(value, dict)
        return value

    @classmethod
    def from_value(cls, value: object, path: str) -> PlanningDiagnostics:
        """Read a record back from what `PlanningDiagnostics.to_dict` wrote.

        Parameters
        ----------
        value : object
            The mapping.
        path : str
            Where the value sits in its record, for error messages.

        Returns
        -------
        PlanningDiagnostics
            The record.

        Raises
        ------
        ValueError
            If the schema is another one, a field is missing or malformed, or a
            written count or the written repairs disagree with the record.
        """
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
                f"{path}.search.capacity_refinement.effective_object_capacity_bytes",
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
