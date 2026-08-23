"""Structured PressureFit search, repair, and work diagnostics."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
from typing import ClassVar


def _nonnegative(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class PressureFitRepairDiagnostics:
    """Categorized monotonic changes made while repairing one search path."""

    unclassified_attempts: int = 0
    admission_prefetch_advance_attempts: int = 0
    admission_prefetch_delay_attempts: int = 0
    admission_pressure_boundary_attempts: int = 0
    simulation_prefetch_delay_attempts: int = 0
    simulation_pressure_boundary_attempts: int = 0

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            _nonnegative(name, getattr(self, name))

    @property
    def total_attempts(self) -> int:
        return sum(getattr(self, name) for name in self.__dataclass_fields__)

    @property
    def pressure_boundary_attempts(self) -> int:
        return (
            self.admission_pressure_boundary_attempts
            + self.simulation_pressure_boundary_attempts
        )

    def __add__(
        self, other: PressureFitRepairDiagnostics
    ) -> PressureFitRepairDiagnostics:
        if not isinstance(other, PressureFitRepairDiagnostics):
            return NotImplemented
        return PressureFitRepairDiagnostics(
            **{
                name: getattr(self, name) + getattr(other, name)
                for name in self.__dataclass_fields__
            }
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "total_attempts": self.total_attempts,
            "pressure_boundary_attempts": self.pressure_boundary_attempts,
            "unclassified_attempts": self.unclassified_attempts,
            "admission_failure": {
                "prefetch_advance_attempts": (self.admission_prefetch_advance_attempts),
                "prefetch_delay_attempts": self.admission_prefetch_delay_attempts,
                "pressure_boundary_attempts": (
                    self.admission_pressure_boundary_attempts
                ),
            },
            "simulation_failure": {
                "prefetch_delay_attempts": self.simulation_prefetch_delay_attempts,
                "pressure_boundary_attempts": (
                    self.simulation_pressure_boundary_attempts
                ),
            },
        }

    @classmethod
    def from_value(
        cls, value: object, path: str = "pressurefit_repairs"
    ) -> PressureFitRepairDiagnostics:
        data = _mapping(value, path)
        admission = _mapping(data.get("admission_failure"), f"{path}.admission_failure")
        simulation = _mapping(
            data.get("simulation_failure"), f"{path}.simulation_failure"
        )
        result = cls(
            unclassified_attempts=_integer(
                data.get("unclassified_attempts", 0),
                f"{path}.unclassified_attempts",
            ),
            admission_prefetch_advance_attempts=_integer(
                admission.get("prefetch_advance_attempts", 0),
                f"{path}.admission_failure.prefetch_advance_attempts",
            ),
            admission_prefetch_delay_attempts=_integer(
                admission.get("prefetch_delay_attempts", 0),
                f"{path}.admission_failure.prefetch_delay_attempts",
            ),
            admission_pressure_boundary_attempts=_integer(
                admission.get("pressure_boundary_attempts", 0),
                f"{path}.admission_failure.pressure_boundary_attempts",
            ),
            simulation_prefetch_delay_attempts=_integer(
                simulation.get("prefetch_delay_attempts", 0),
                f"{path}.simulation_failure.prefetch_delay_attempts",
            ),
            simulation_pressure_boundary_attempts=_integer(
                simulation.get("pressure_boundary_attempts", 0),
                f"{path}.simulation_failure.pressure_boundary_attempts",
            ),
        )
        declared = data.get("total_attempts")
        if declared is not None and _integer(declared, f"{path}.total_attempts") != (
            result.total_attempts
        ):
            raise ValueError(f"{path}.total_attempts does not reconcile")
        declared_pressure = data.get("pressure_boundary_attempts")
        if (
            declared_pressure is not None
            and _integer(declared_pressure, f"{path}.pressure_boundary_attempts")
            != result.pressure_boundary_attempts
        ):
            raise ValueError(f"{path}.pressure_boundary_attempts does not reconcile")
        return result


@dataclass(frozen=True, slots=True)
class PressureFitWorkDiagnostics:
    """Exact search operations and summed component work time.

    Invocation-level values include shared work performed before or across
    candidates. Consequently they need not equal the sum of candidate values.
    Times are summed component work, not necessarily elapsed wall time when
    independent recomputation problems are evaluated concurrently.
    """

    evaluation_time_ns: int = 0
    residency_cache_hits: int = 0
    residency_cache_misses: int = 0
    schedule_emissions: int = 0
    schedule_cache_hits: int = 0
    simulation_calls: int = 0
    simulation_cache_hits: int = 0
    result_simulation_calls: int = 0
    admission_calls: int = 0
    result_admission_calls: int = 0
    residency_time_ns: int = 0
    schedule_time_ns: int = 0
    simulation_time_ns: int = 0
    result_simulation_time_ns: int = 0
    admission_time_ns: int = 0
    result_admission_time_ns: int = 0
    digest_time_ns: int = 0

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            _nonnegative(name, getattr(self, name))

    @property
    def simulation_requests(self) -> int:
        return self.simulation_calls + self.simulation_cache_hits

    @property
    def total_simulation_calls(self) -> int:
        return self.simulation_calls + self.result_simulation_calls

    @property
    def total_admission_calls(self) -> int:
        return self.admission_calls + self.result_admission_calls

    def __add__(self, other: PressureFitWorkDiagnostics) -> PressureFitWorkDiagnostics:
        if not isinstance(other, PressureFitWorkDiagnostics):
            return NotImplemented
        return PressureFitWorkDiagnostics(
            **{
                name: getattr(self, name) + getattr(other, name)
                for name in self.__dataclass_fields__
            }
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "evaluation": {"summed_wall_time_ns": self.evaluation_time_ns},
            "residency": {
                "cache_hits": self.residency_cache_hits,
                "evaluations": self.residency_cache_misses,
                "summed_work_time_ns": self.residency_time_ns,
            },
            "schedule": {
                "cache_hits": self.schedule_cache_hits,
                "emissions": self.schedule_emissions,
                "summed_work_time_ns": self.schedule_time_ns,
            },
            "simulation": {
                "requests": self.simulation_requests,
                "search_calls": self.simulation_calls,
                "cache_hits": self.simulation_cache_hits,
                "result_materialization_calls": self.result_simulation_calls,
                "total_calls": self.total_simulation_calls,
                "summed_work_time_ns": self.simulation_time_ns,
                "result_materialization_time_ns": (self.result_simulation_time_ns),
            },
            "admission": {
                "search_calls": self.admission_calls,
                "result_materialization_calls": self.result_admission_calls,
                "total_calls": self.total_admission_calls,
                "summed_work_time_ns": self.admission_time_ns,
                "result_materialization_time_ns": self.result_admission_time_ns,
            },
            "digest": {"summed_work_time_ns": self.digest_time_ns},
        }

    @classmethod
    def from_value(
        cls, value: object, path: str = "pressurefit_work"
    ) -> PressureFitWorkDiagnostics:
        data = _mapping(value, path)
        evaluation = _mapping(data.get("evaluation"), f"{path}.evaluation")
        residency = _mapping(data.get("residency"), f"{path}.residency")
        schedule = _mapping(data.get("schedule"), f"{path}.schedule")
        simulation = _mapping(data.get("simulation"), f"{path}.simulation")
        admission = _mapping(data.get("admission"), f"{path}.admission")
        digest = _mapping(data.get("digest"), f"{path}.digest")
        result = cls(
            evaluation_time_ns=_integer(
                evaluation.get("summed_wall_time_ns", 0),
                f"{path}.evaluation.summed_wall_time_ns",
            ),
            residency_cache_hits=_integer(
                residency.get("cache_hits", 0), f"{path}.residency.cache_hits"
            ),
            residency_cache_misses=_integer(
                residency.get("evaluations", 0), f"{path}.residency.evaluations"
            ),
            schedule_emissions=_integer(
                schedule.get("emissions", 0), f"{path}.schedule.emissions"
            ),
            schedule_cache_hits=_integer(
                schedule.get("cache_hits", 0), f"{path}.schedule.cache_hits"
            ),
            simulation_calls=_integer(
                simulation.get("search_calls", simulation.get("calls", 0)),
                f"{path}.simulation.search_calls",
            ),
            simulation_cache_hits=_integer(
                simulation.get("cache_hits", 0), f"{path}.simulation.cache_hits"
            ),
            result_simulation_calls=_integer(
                simulation.get("result_materialization_calls", 0),
                f"{path}.simulation.result_materialization_calls",
            ),
            admission_calls=_integer(
                admission.get("search_calls", admission.get("calls", 0)),
                f"{path}.admission.search_calls",
            ),
            result_admission_calls=_integer(
                admission.get("result_materialization_calls", 0),
                f"{path}.admission.result_materialization_calls",
            ),
            residency_time_ns=_integer(
                residency.get("summed_work_time_ns", 0),
                f"{path}.residency.summed_work_time_ns",
            ),
            schedule_time_ns=_integer(
                schedule.get("summed_work_time_ns", 0),
                f"{path}.schedule.summed_work_time_ns",
            ),
            simulation_time_ns=_integer(
                simulation.get("summed_work_time_ns", 0),
                f"{path}.simulation.summed_work_time_ns",
            ),
            result_simulation_time_ns=_integer(
                simulation.get("result_materialization_time_ns", 0),
                f"{path}.simulation.result_materialization_time_ns",
            ),
            admission_time_ns=_integer(
                admission.get("summed_work_time_ns", 0),
                f"{path}.admission.summed_work_time_ns",
            ),
            result_admission_time_ns=_integer(
                admission.get("result_materialization_time_ns", 0),
                f"{path}.admission.result_materialization_time_ns",
            ),
            digest_time_ns=_integer(
                digest.get("summed_work_time_ns", 0),
                f"{path}.digest.summed_work_time_ns",
            ),
        )
        expected = {
            "requests": result.simulation_requests,
            "total_calls": result.total_simulation_calls,
        }
        for name, expected_value in expected.items():
            declared = simulation.get(name)
            if (
                declared is not None
                and _integer(declared, f"{path}.simulation.{name}") != expected_value
            ):
                raise ValueError(f"{path}.simulation.{name} does not reconcile")
        declared_admission_calls = admission.get("total_calls")
        if (
            declared_admission_calls is not None
            and _integer(declared_admission_calls, f"{path}.admission.total_calls")
            != result.total_admission_calls
        ):
            raise ValueError(f"{path}.admission.total_calls does not reconcile")
        return result


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
            schedule_digest=self.schedule_digest,
            failure_kind=self.failure_kind,
            failure_detail=self.failure_detail,
            repairs=self.repairs,
            work=self.work,
        )


@dataclass(frozen=True, slots=True)
class RecomputationChoiceDiagnostic:
    """One graph-pair choice in a recomputation selection."""

    group_id: str
    option_id: str

    def to_dict(self) -> dict[str, str]:
        return {"group_id": self.group_id, "option_id": self.option_id}

    @classmethod
    def from_value(cls, value: object, path: str) -> RecomputationChoiceDiagnostic:
        data = _mapping(value, path)
        return cls(
            group_id=_string(data.get("group_id"), f"{path}.group_id"),
            option_id=_string(data.get("option_id"), f"{path}.option_id"),
        )


@dataclass(frozen=True, slots=True)
class RecomputationProblemDiagnostics:
    """Shared problem work and policy evaluations for one concrete selection."""

    selection_id: str
    choices: tuple[RecomputationChoiceDiagnostic, ...]
    selected_candidate_id: str | None
    selected_makespan_ns: int | None
    candidate_evaluations: tuple[CandidateDiagnostic, ...]
    work: PressureFitWorkDiagnostics = field(default_factory=PressureFitWorkDiagnostics)

    def __post_init__(self) -> None:
        if any(
            candidate.selection_id != self.selection_id
            for candidate in self.candidate_evaluations
        ):
            raise ValueError(
                "candidate evaluation does not reference its recomputation problem"
            )
        candidate_ids = tuple(
            candidate.candidate_id for candidate in self.candidate_evaluations
        )
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError(
                "candidate policy appears more than once in a recomputation problem"
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
        else:
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
        return len(self.candidate_evaluations)

    @property
    def valid_candidate_policy_count(self) -> int:
        return sum(item.status == "valid" for item in self.candidate_evaluations)

    @property
    def candidate_status_counts(self) -> dict[str, int]:
        return dict(
            sorted(Counter(item.status for item in self.candidate_evaluations).items())
        )

    @property
    def repairs(self) -> PressureFitRepairDiagnostics:
        result = PressureFitRepairDiagnostics()
        for candidate in self.candidate_evaluations:
            result += candidate.repairs
        return result

    def to_dict(self) -> dict[str, object]:
        return {
            "recomputation_selection": {
                "selection_id": self.selection_id,
                "graph_pair_choices": [item.to_dict() for item in self.choices],
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
            "repairs": self.repairs.to_dict(),
            "candidate_policy_evaluations": [
                item.to_dict() for item in self.candidate_evaluations
            ],
        }

    @classmethod
    def from_value(cls, value: object, path: str) -> RecomputationProblemDiagnostics:
        data = _mapping(value, path)
        selection = _mapping(
            data.get("recomputation_selection"),
            f"{path}.recomputation_selection",
        )
        selected = _mapping(
            data.get("selected_candidate_policy"),
            f"{path}.selected_candidate_policy",
        )
        summary = _mapping(data.get("summary"), f"{path}.summary")
        selection_id = _string(
            selection.get("selection_id"),
            f"{path}.recomputation_selection.selection_id",
        )
        raw_candidates = _list(
            data.get("candidate_policy_evaluations"),
            f"{path}.candidate_policy_evaluations",
        )
        candidates = tuple(
            CandidateDiagnostic.from_value(
                item,
                f"{path}.candidate_policy_evaluations[{index}]",
            ).with_selection_id(selection_id)
            for index, item in enumerate(raw_candidates)
        )
        result = cls(
            selection_id=selection_id,
            choices=tuple(
                RecomputationChoiceDiagnostic.from_value(
                    item,
                    f"{path}.recomputation_selection.graph_pair_choices[{index}]",
                )
                for index, item in enumerate(
                    _list(
                        selection.get("graph_pair_choices"),
                        f"{path}.recomputation_selection.graph_pair_choices",
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
            work=PressureFitWorkDiagnostics.from_value(
                data.get("work"), f"{path}.work"
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
            PressureFitRepairDiagnostics.from_value(
                data.get("repairs"), f"{path}.repairs"
            )
            != result.repairs
        ):
            raise ValueError(f"{path}.repairs does not reconcile")
        return result


@dataclass(frozen=True, slots=True)
class AdmissionRefinement:
    """One monotonic reduction of logical object capacity after slab replay."""

    attempt: int
    previous_object_capacity_bytes: int
    required_additional_slack_bytes: int
    reserve_increment_bytes: int
    object_capacity_bytes: int


@dataclass(frozen=True, slots=True)
class PressureFitDiagnostics:
    """Complete problem, policy-evaluation, and aggregate PressureFit evidence."""

    SCHEMA: ClassVar[str] = "shadowspill.pressurefit_diagnostics/v2"

    selected_candidate_id: str
    selected_selection_id: str
    selected_makespan_ns: int
    recomputation_problems: tuple[RecomputationProblemDiagnostics, ...]
    work: PressureFitWorkDiagnostics = field(default_factory=PressureFitWorkDiagnostics)
    admission_refinements: tuple[AdmissionRefinement, ...] = ()
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
                "attempts": [
                    {
                        "attempt": item.attempt,
                        "previous_object_capacity_bytes": (
                            item.previous_object_capacity_bytes
                        ),
                        "required_additional_slack_bytes": (
                            item.required_additional_slack_bytes
                        ),
                        "reserve_increment_bytes": item.reserve_increment_bytes,
                        "object_capacity_bytes": item.object_capacity_bytes,
                    }
                    for item in self.admission_refinements
                ],
            },
            "recomputation_problems": [
                item.to_dict() for item in self.recomputation_problems
            ],
        }

    def stable_dict(self) -> dict[str, object]:
        """Return deterministic search evidence without measured work times."""

        value = _without_work_times(self.to_dict())
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
            admission_refinements=tuple(
                AdmissionRefinement(
                    attempt=_integer(
                        item.get("attempt"),
                        f"{path}.capacity_refinement.attempts[{index}].attempt",
                    ),
                    previous_object_capacity_bytes=_integer(
                        item.get("previous_object_capacity_bytes"),
                        f"{path}.capacity_refinement.attempts[{index}].previous_object_capacity_bytes",
                    ),
                    required_additional_slack_bytes=_integer(
                        item.get("required_additional_slack_bytes"),
                        f"{path}.capacity_refinement.attempts[{index}].required_additional_slack_bytes",
                    ),
                    reserve_increment_bytes=_integer(
                        item.get("reserve_increment_bytes"),
                        f"{path}.capacity_refinement.attempts[{index}].reserve_increment_bytes",
                    ),
                    object_capacity_bytes=_integer(
                        item.get("object_capacity_bytes"),
                        f"{path}.capacity_refinement.attempts[{index}].object_capacity_bytes",
                    ),
                )
                for index, raw in enumerate(
                    _list(
                        refinement.get("attempts"),
                        f"{path}.capacity_refinement.attempts",
                    )
                )
                for item in (
                    _mapping(raw, f"{path}.capacity_refinement.attempts[{index}]"),
                )
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


def _parse_candidate_id(value: str) -> tuple[str, str, bool]:
    coalesced = value.endswith("-coalesced")
    base = value[: -len("-coalesced")] if coalesced else value
    strategy, separator, rule = base.partition("/")
    if not separator:
        return "unknown", "unknown", coalesced
    return strategy, rule, coalesced


def _mapping(value: object, path: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{path} must be an object")
    return value


def _without_work_times(value: object) -> object:
    if isinstance(value, list):
        return [_without_work_times(item) for item in value]
    if not isinstance(value, dict):
        return value
    return {
        key: _without_work_times(item)
        for key, item in value.items()
        if not key.endswith("_time_ns")
    }


def _list(value: object, path: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{path} must be a list")
    return value


def _integer(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path} must be an integer")
    return value


def _string(value: object, path: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{path} must be a string")
    return value


def _boolean(value: object, path: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{path} must be a boolean")
    return value


def _optional_integer(value: object, path: str) -> int | None:
    return None if value is None else _integer(value, path)


def _optional_string(value: object, path: str) -> str | None:
    return None if value is None else _string(value, path)


__all__ = [
    "AdmissionRefinement",
    "CandidateDiagnostic",
    "PressureFitDiagnostics",
    "PressureFitRepairDiagnostics",
    "PressureFitWorkDiagnostics",
    "RecomputationChoiceDiagnostic",
    "RecomputationProblemDiagnostics",
]
