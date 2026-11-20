"""One resolved program: what it chose, and what it cost."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from shadowspill.planning.memory_search.diagnostics.candidates import (
    CandidateDiagnostic,
)
from shadowspill.planning.memory_search.diagnostics.counters import (
    PlanningRepairDiagnostics,
    PlanningWorkDiagnostics,
)
from shadowspill.planning.memory_search.diagnostics.json import (
    _list,
    _mapping,
    _optional_integer,
    _optional_string,
    _span,
    _string,
)

#: The candidate id a resolved program answers with when the plan to beat
#: won: the plan it was handed, not one its candidates reached.
INCUMBENT_CANDIDATE_ID = "incumbent"


@dataclass(frozen=True, slots=True)
class IncumbentDiagnostic:
    """What became of the plan to beat, measured at one resolved program's capacity.

    `ResolvedProgramDiagnostics.incumbent` holds it.

    Parameters
    ----------
    status : str
        ``"valid"`` when it simulated and its layout fit the pool,
        ``"unplaceable"`` when its layout did not fit, ``"infeasible"`` when
        it did not simulate or admit here, and ``"error"`` when it could not
        be measured.
    makespan_ns : int or None
        Its makespan at this capacity, in nanoseconds, if measured.
    required_bytes : int or None
        Pool bytes its layout needed, if measured against a pool.
    selected : bool
        Whether it is the resolved program's answer; only a valid plan with
        a makespan can be.
    schedule_digest : str, optional
        Which plan it is.
    found_by : str, optional
        The candidate policy that first found it, read through any chain of
        hand-offs between searches.
    found_at_capacity_bytes : int, optional
        The object capacity it was first found at.

    Raises
    ------
    ValueError
        If the status is unknown, or a plan that is not valid or has no
        makespan is selected.
    """

    status: str
    makespan_ns: int | None
    required_bytes: int | None
    selected: bool
    schedule_digest: str | None = None
    found_by: str | None = None
    found_at_capacity_bytes: int | None = None

    def __post_init__(self) -> None:
        if self.status not in ("valid", "unplaceable", "infeasible", "error"):
            raise ValueError(f"unknown incumbent status {self.status!r}")
        if self.selected and self.status != "valid":
            raise ValueError("a plan to beat that is not valid cannot be selected")
        if self.selected and self.makespan_ns is None:
            raise ValueError("a selected plan to beat has a makespan")

    def to_dict(self) -> dict[str, object]:
        """Write the outcome as a JSON-compatible mapping.

        Returns
        -------
        dict
            Every field by name.
        """
        return {
            "status": self.status,
            "makespan_ns": self.makespan_ns,
            "required_bytes": self.required_bytes,
            "selected": self.selected,
            "schedule_digest": self.schedule_digest,
            "found_by": self.found_by,
            "found_at_capacity_bytes": self.found_at_capacity_bytes,
        }

    @classmethod
    def from_value(cls, value: object, path: str) -> IncumbentDiagnostic:
        """Read an outcome back from what `IncumbentDiagnostic.to_dict` wrote.

        Parameters
        ----------
        value : object
            The mapping.
        path : str
            Where the value sits in its record, for error messages.

        Returns
        -------
        IncumbentDiagnostic
            The outcome.

        Raises
        ------
        ValueError
            If a field is missing or malformed.
        """
        data = _mapping(value, path)
        selected = data.get("selected")
        if not isinstance(selected, bool):
            raise ValueError(f"{path}.selected must be a boolean")
        return cls(
            status=_string(data.get("status"), f"{path}.status"),
            makespan_ns=_optional_integer(
                data.get("makespan_ns"), f"{path}.makespan_ns"
            ),
            required_bytes=_optional_integer(
                data.get("required_bytes"), f"{path}.required_bytes"
            ),
            selected=selected,
            schedule_digest=_optional_string(
                data.get("schedule_digest"), f"{path}.schedule_digest"
            ),
            found_by=_optional_string(data.get("found_by"), f"{path}.found_by"),
            found_at_capacity_bytes=_optional_integer(
                data.get("found_at_capacity_bytes"), f"{path}.found_at_capacity_bytes"
            ),
        )


@dataclass(frozen=True, slots=True)
class TaskAlternativeChoiceDiagnostic:
    """One task-alternative choice a resolved program fixes.

    Parameters
    ----------
    group_id : str
        The task-alternative group.
    option_id : str
        The option chosen for it.
    """

    group_id: str
    option_id: str

    def to_dict(self) -> dict[str, str]:
        """Write the choice as a JSON-compatible mapping.

        Returns
        -------
        dict
            ``group_id`` and ``option_id``.
        """
        return {"group_id": self.group_id, "option_id": self.option_id}

    @classmethod
    def from_value(cls, value: object, path: str) -> TaskAlternativeChoiceDiagnostic:
        """Read a choice back from what its ``to_dict`` wrote.

        Parameters
        ----------
        value : object
            The mapping.
        path : str
            Where the value sits in its record, for error messages.

        Returns
        -------
        TaskAlternativeChoiceDiagnostic
            The choice.

        Raises
        ------
        ValueError
            If a field is missing or malformed.
        """
        data = _mapping(value, path)
        return cls(
            group_id=_string(data.get("group_id"), f"{path}.group_id"),
            option_id=_string(data.get("option_id"), f"{path}.option_id"),
        )


@dataclass(frozen=True, slots=True)
class ResolvedProgramDiagnostics:
    """One resolved program a search planned: its choices, candidates and answer.

    `PlanningDiagnostics.resolved_programs` holds these.

    Parameters
    ----------
    selection_id : str
        The resolved program's identity.
    choices : tuple of TaskAlternativeChoiceDiagnostic
        The option it fixes for each task-alternative group.
    selected_candidate_id : str or None
        The candidate policy it answers with, ``"incumbent"`` when the plan
        to beat won, or None when no candidate reached a plan.
    selected_makespan_ns : int or None
        That answer's makespan, in nanoseconds; None without an answer.
    candidate_evaluations : tuple of CandidateDiagnostic
        Every candidate policy evaluated for it, each once.
    work : PlanningWorkDiagnostics, default PlanningWorkDiagnostics()
        Its operation counts and section times.
    started_ns : int, default 0
        When its first candidate started, in nanoseconds of wall time from
        the start of the call.
    finished_ns : int, default 0
        When its last candidate finished, on the same clock. Resolved
        programs of one call may overlap.
    evict_ineligible_aliases : int, default 0
        How many objects
        `GenericPlanningOptions.minimum_object_bytes_evict_eligible` kept
        resident.
    evict_ineligible_bytes : int, default 0
        Their bytes.
    fetched_bytes : int, default 0
        Bytes its own best plan fetches; zero when it placed nothing.
    evicted_bytes : int, default 0
        Bytes its own best plan evicts; zero when it placed nothing.
    incumbent : IncumbentDiagnostic, optional
        What became of the plan to beat, when it was handed one.

    Raises
    ------
    ValueError
        If a candidate names another resolved program or appears twice, or the
        selected candidate and makespan do not match a valid evaluation or the
        plan to beat.
    """

    selection_id: str
    choices: tuple[TaskAlternativeChoiceDiagnostic, ...]
    selected_candidate_id: str | None
    selected_makespan_ns: int | None
    candidate_evaluations: tuple[CandidateDiagnostic, ...]
    work: PlanningWorkDiagnostics = field(default_factory=PlanningWorkDiagnostics)
    #: This problem's span on the same clock its candidates use: from the first
    #: candidate a worker started to the last one it finished. Several problems
    #: evaluated in one call overlap, because workers take whatever task is
    #: next rather than finishing one problem before starting another.
    started_ns: int = 0
    finished_ns: int = 0
    #: The objects `minimum_object_bytes_evict_eligible` kept resident for
    #: this problem: how many, and their bytes.
    evict_ineligible_aliases: int = 0
    evict_ineligible_bytes: int = 0
    #: What this problem's own best plan moves. The winner's traffic is on
    #: the plan summary; these are the alternatives', which is what says
    #: whether a cheaper-compute selection pays for it on the lanes. Zero on
    #: a problem that placed nothing.
    fetched_bytes: int = 0
    evicted_bytes: int = 0
    #: What became of the plan to beat, when this problem was handed one;
    #: absent on a problem that carried none.
    incumbent: IncumbentDiagnostic | None = None

    def __post_init__(self) -> None:
        if any(
            candidate.selection_id != self.selection_id
            for candidate in self.candidate_evaluations
        ):
            raise ValueError(
                "candidate evaluation does not reference its resolved program"
            )
        candidate_ids = tuple(
            candidate.candidate_id for candidate in self.candidate_evaluations
        )
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError(
                "candidate policy appears more than once in a resolved program"
            )
        valid = tuple(
            candidate
            for candidate in self.candidate_evaluations
            if candidate.status == "valid"
        )
        if self.selected_candidate_id is None:
            if self.selected_makespan_ns is not None:
                raise ValueError(
                    "selected_makespan_ns requires a selected candidate policy"
                )
            if self.incumbent is not None and self.incumbent.selected:
                raise ValueError("a selected plan to beat is the selected policy")
        elif self.selected_candidate_id == INCUMBENT_CANDIDATE_ID:
            if self.incumbent is None or not self.incumbent.selected:
                raise ValueError(
                    "the plan to beat is selected only when it was given and won"
                )
            if self.selected_makespan_ns != self.incumbent.makespan_ns:
                raise ValueError(
                    "problem selected makespan does not match the plan to beat"
                )
        else:
            if self.incumbent is not None and self.incumbent.selected:
                raise ValueError("a selected plan to beat is the selected policy")
            selected = tuple(
                candidate
                for candidate in valid
                if candidate.candidate_id == self.selected_candidate_id
            )
            if len(selected) != 1:
                raise ValueError(
                    "selected candidate policy is not a unique valid evaluation"
                )
            if self.selected_makespan_ns != selected[0].makespan_ns:
                raise ValueError(
                    "problem selected makespan does not match candidate evaluation"
                )

    @property
    def candidate_policy_count(self) -> int:
        """How many candidate policies were evaluated."""
        return len(self.candidate_evaluations)

    @property
    def valid_candidate_policy_count(self) -> int:
        """How many of them reached a plan."""
        return sum(item.status == "valid" for item in self.candidate_evaluations)

    @property
    def candidate_status_counts(self) -> dict[str, int]:
        """How many candidates ended in each status, by status name."""
        return dict(
            sorted(Counter(item.status for item in self.candidate_evaluations).items())
        )

    @property
    def repairs(self) -> PlanningRepairDiagnostics:
        """The candidates' repairs, summed."""
        result = PlanningRepairDiagnostics()
        for candidate in self.candidate_evaluations:
            result += candidate.repairs
        return result

    def to_dict(self) -> dict[str, object]:
        """Write the resolved program's record as a JSON-compatible mapping.

        Returns
        -------
        dict
            Its identity and choices, its answer, the derived counts, work,
            span, evict-ineligible objects, traffic, repairs, incumbent and
            candidate evaluations.
        """
        return {
            "resolved_program": {
                "selection_id": self.selection_id,
                "task_alternative_choices": [item.to_dict() for item in self.choices],
            },
            "selected_candidate_policy": {
                "candidate_id": self.selected_candidate_id,
                "makespan_ns": self.selected_makespan_ns,
            },
            "summary": {
                "candidate_policy_count": self.candidate_policy_count,
                "valid_candidate_policy_count": (self.valid_candidate_policy_count),
                "candidate_status_counts": self.candidate_status_counts,
            },
            "work": self.work.to_dict(),
            "span": {
                "started_ns": self.started_ns,
                "finished_ns": self.finished_ns,
            },
            "evict_ineligible": {
                "aliases": self.evict_ineligible_aliases,
                "bytes": self.evict_ineligible_bytes,
            },
            "transfer": {
                "fetched_bytes": self.fetched_bytes,
                "evicted_bytes": self.evicted_bytes,
            },
            "repairs": self.repairs.to_dict(),
            "incumbent": None if self.incumbent is None else self.incumbent.to_dict(),
            "candidate_policy_evaluations": [
                item.to_dict() for item in self.candidate_evaluations
            ],
        }

    @classmethod
    def from_value(cls, value: object, path: str) -> ResolvedProgramDiagnostics:
        """Read a record back from what `ResolvedProgramDiagnostics.to_dict` wrote.

        Parameters
        ----------
        value : object
            The mapping.
        path : str
            Where the value sits in its record, for error messages.

        Returns
        -------
        ResolvedProgramDiagnostics
            The record.

        Raises
        ------
        ValueError
            If a field is missing or malformed, or a written count or the
            written repairs disagree with the candidates.
        """
        data = _mapping(value, path)
        selection = _mapping(
            data.get("resolved_program"),
            f"{path}.resolved_program",
        )
        selected = _mapping(
            data.get("selected_candidate_policy"),
            f"{path}.selected_candidate_policy",
        )
        summary = _mapping(data.get("summary"), f"{path}.summary")
        selection_id = _string(
            selection.get("selection_id"),
            f"{path}.resolved_program.selection_id",
        )
        raw_candidates = _list(
            data.get("candidate_policy_evaluations"),
            f"{path}.candidate_policy_evaluations",
        )
        candidates = tuple(
            CandidateDiagnostic.from_value(
                item,
                f"{path}.candidate_policy_evaluations[{index}]",
                selection_id,
            )
            for index, item in enumerate(raw_candidates)
        )
        result = cls(
            selection_id=selection_id,
            choices=tuple(
                TaskAlternativeChoiceDiagnostic.from_value(
                    item,
                    f"{path}.resolved_program.task_alternative_choices[{index}]",
                )
                for index, item in enumerate(
                    _list(
                        selection.get("task_alternative_choices"),
                        f"{path}.resolved_program.task_alternative_choices",
                    )
                )
            ),
            selected_candidate_id=_optional_string(
                selected.get("candidate_id"),
                f"{path}.selected_candidate_policy.candidate_id",
            ),
            selected_makespan_ns=_optional_integer(
                selected.get("makespan_ns"),
                f"{path}.selected_candidate_policy.makespan_ns",
            ),
            candidate_evaluations=candidates,
            work=PlanningWorkDiagnostics.from_value(data.get("work"), f"{path}.work"),
            started_ns=_span(data.get("span"), "started_ns", f"{path}.span"),
            finished_ns=_span(data.get("span"), "finished_ns", f"{path}.span"),
            evict_ineligible_aliases=_span(
                data.get("evict_ineligible"), "aliases", f"{path}.evict_ineligible"
            ),
            evict_ineligible_bytes=_span(
                data.get("evict_ineligible"), "bytes", f"{path}.evict_ineligible"
            ),
            fetched_bytes=_span(
                data.get("transfer"), "fetched_bytes", f"{path}.transfer"
            ),
            evicted_bytes=_span(
                data.get("transfer"), "evicted_bytes", f"{path}.transfer"
            ),
            incumbent=(
                None
                if data.get("incumbent") is None
                else IncumbentDiagnostic.from_value(
                    data.get("incumbent"), f"{path}.incumbent"
                )
            ),
        )
        expected_summary = {
            "candidate_policy_count": result.candidate_policy_count,
            "valid_candidate_policy_count": result.valid_candidate_policy_count,
            "candidate_status_counts": result.candidate_status_counts,
        }
        for name, expected in expected_summary.items():
            if summary.get(name) != expected:
                raise ValueError(f"{path}.summary.{name} does not reconcile")
        if (
            PlanningRepairDiagnostics.from_value(data.get("repairs"), f"{path}.repairs")
            != result.repairs
        ):
            raise ValueError(f"{path}.repairs does not reconcile")
        return result
