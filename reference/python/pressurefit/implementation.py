"""Deterministic, simulator-verified PressureFit orchestration."""

from __future__ import annotations

import time
from bisect import bisect_right
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace

from reference.python.simulator import simulate_python
from shadowspill.errors import PlanInfeasibleError, PlanSearchExhaustedError
from shadowspill.ir import (
    MemorySchedule,
    ResidencySpec,
    ShadowSpillProgram,
    TaskAlternativeChoice,
)
from shadowspill.ir.validation import ValidationError
from shadowspill.planner.admission import AdmissionFacts
from shadowspill.planner.diagnostics import (
    CandidateDiagnostic,
    PlanningDiagnostics,
    PlanningRepairDiagnostics,
    PlanningSectionTiming,
    PlanningWorkDiagnostics,
    ResolvedProgramDiagnostics,
    TaskAlternativeChoiceDiagnostic,
)
from shadowspill.planner.request import GenericPlanningOptions
from shadowspill.planner.result import ProgramPlanResult
from shadowspill.planner.search import SearchOptions
from shadowspill.planner.search.algorithms.pressurefit import PressureFit
from shadowspill.planner.search.algorithms.pressurefit.options import PressureFitOptions
from shadowspill.planner.search.toolkit.resolution import resolutions
from shadowspill.simulator import (
    SimulationConfig,
    SimulationInfeasibleError,
    SimulationResult,
)

from .actions import emit_schedule
from .facts import PlanningFacts, build_facts
from .residency import (
    Cut,
    ResidencyPlan,
    assert_required_floor,
    extend_interval_entries,
    reduce_pressure,
    seed_residency,
)

_ADMISSION_RESERVE_GRANULARITY_BYTES = 2 << 20
_ADMISSION_INITIAL_REFINEMENT_BYTES = 128 << 20
_ADMISSION_DOUBLING_LIMIT_BYTES = 1 << 30
_ADMISSION_LINEAR_REFINEMENT_BYTES = 512 << 20


@dataclass(frozen=True, slots=True)
class _SelectionProblem:
    selections: tuple[TaskAlternativeChoice, ...]
    selection_id: str
    facts: PlanningFacts
    seed: ResidencyPlan
    cut_scores: dict[tuple[Cut, str], tuple[int, ...]] = field(
        default_factory=dict,
        compare=False,
        repr=False,
    )
    interval_plans: dict[
        tuple[str, tuple[tuple[str, int, int], ...]], ResidencyPlan
    ] = field(
        default_factory=dict,
        compare=False,
        repr=False,
    )
    schedule_cache: dict[tuple[ResidencyPlan, str, bool, bool], MemorySchedule] = field(
        default_factory=dict, compare=False, repr=False
    )
    simulation_cache: dict[
        MemorySchedule,
        SimulationResult | _CachedSimulationFailure,
    ] = field(default_factory=dict, compare=False, repr=False)


@dataclass(frozen=True, slots=True)
class _CachedSimulationFailure:
    message: str
    kind: str
    time_ns: int
    task_id: str | None
    alias_group_ids: tuple[str, ...]
    location: str | None
    capacity_bytes: int | None
    used_bytes: int | None
    requested_bytes: int | None

    @classmethod
    def from_error(cls, error: SimulationInfeasibleError) -> _CachedSimulationFailure:
        return cls(
            str(error),
            error.kind,
            error.time_ns,
            error.task_id,
            error.alias_group_ids,
            error.location,
            error.capacity_bytes,
            error.used_bytes,
            error.requested_bytes,
        )

    def to_error(self) -> SimulationInfeasibleError:
        return SimulationInfeasibleError(
            self.message,
            kind=self.kind,
            time_ns=self.time_ns,
            task_id=self.task_id,
            alias_group_ids=self.alias_group_ids,
            location=self.location,
            capacity_bytes=self.capacity_bytes,
            used_bytes=self.used_bytes,
            requested_bytes=self.requested_bytes,
        )


@dataclass(frozen=True, slots=True)
class _CandidateSpec:
    ordinal: int
    problem: _SelectionProblem
    strategy: str
    fetch_rule: str
    coalesced: bool

    @property
    def candidate_id(self) -> str:
        suffix = "-coalesced" if self.coalesced else ""
        return f"{self.strategy}/{self.fetch_rule}{suffix}"


@dataclass(frozen=True, slots=True)
class _CandidateOutcome:
    spec: _CandidateSpec
    diagnostic: CandidateDiagnostic
    schedule: MemorySchedule | None = None
    simulation: SimulationResult | None = None


def _selection_id(selections: tuple[TaskAlternativeChoice, ...]) -> str:
    if not selections:
        return "none"
    return ",".join(f"{item.group_id}={item.option_id}" for item in selections)


def validate_schedule_feasibility(
    program: ShadowSpillProgram,
    *,
    initial_residency: tuple[ResidencySpec, ...],
    final_residency: tuple[ResidencySpec, ...] = (),
    config: SimulationConfig,
    admission: AdmissionFacts | None = None,
    algorithm_options: PressureFitOptions | None = None,
) -> None:
    """Reject irreducible capacity failures before schedule search.

    This is a necessary-condition preflight, not a PressureFit candidate
    search. It accepts when at least one legal resolution has a
    task-by-task residency floor that fits the declared capacity. PressureFit
    retains the same checks internally as defensive invariants.
    """

    if not isinstance(program, ShadowSpillProgram):
        raise TypeError("program must be a ShadowSpillProgram")
    if not isinstance(initial_residency, tuple):
        raise TypeError("initial_residency must be a tuple")
    if not isinstance(final_residency, tuple):
        raise TypeError("final_residency must be a tuple")
    if not isinstance(config, SimulationConfig):
        raise TypeError("config must be a SimulationConfig")
    if admission is not None:
        if not isinstance(admission, AdmissionFacts):
            raise TypeError("admission must be an AdmissionFacts")
        admission.validate(program)
        configured = {item.device_id: item for item in config.devices}
        if (
            admission.device_id not in configured
            or configured[admission.device_id].capacity_bytes
            != admission.object_capacity_bytes
        ):
            raise ValueError(
                "feasibility capacity must equal AdmissionFacts object capacity"
            )

    failures: list[PlanInfeasibleError] = []
    chosen = algorithm_options or PressureFitOptions()
    for selections in resolutions(program, chosen.resolution_options):
        try:
            facts = build_facts(
                program,
                selections,
                initial_residency,
                final_residency,
                config,
            )
            assert_required_floor(facts)
        except PlanInfeasibleError as error:
            failures.append(error)
        else:
            return

    if failures:
        raise failures[0]
    raise PlanInfeasibleError(
        "no resolution could be constructed",
        kind="graph_pair_selection",
    )


def _repair_pressure(
    facts: PlanningFacts,
    error: SimulationInfeasibleError,
) -> tuple[tuple[str, int], int] | None:
    if error.kind not in {
        "initial-device-capacity",
        "fetch-device-capacity",
        "task-device-capacity",
    }:
        return None
    if error.location is None or not error.location.startswith("device:"):
        return None
    device_id = error.location.removeprefix("device:")
    if error.task_id is None:
        boundary = -1
    else:
        task = facts.task_index.get(error.task_id)
        if task is None:
            return None
        boundary = task - 1 if error.kind == "task-device-capacity" else task
    used = error.used_bytes or 0
    requested = error.requested_bytes or 0
    capacity = (
        error.capacity_bytes
        or facts.object_capacity_by_boundary[device_id][boundary + 1]
    )
    excess = max(used + requested - capacity, 1)
    return (device_id, boundary), excess


def _delay_fetch(
    facts: PlanningFacts,
    schedule: MemorySchedule,
    error: SimulationInfeasibleError,
) -> MemorySchedule | None:
    if error.kind not in {"fetch-device-capacity", "task-device-capacity"}:
        return None
    failing_task = (
        facts.task_index.get(error.task_id) if error.task_id is not None else None
    )
    requested_alias = error.alias_group_ids[0] if error.alias_group_ids else None
    candidates: list[tuple[int, int, int, int]] = []
    for action_index, action in enumerate(schedule.actions):
        if action.kind.value != "fetch":
            continue
        if requested_alias is not None and action.alias_group_id != requested_alias:
            continue
        if (
            error.kind == "fetch-device-capacity"
            and error.task_id is not None
            and action.trigger_task_id != error.task_id
        ):
            continue
        trigger = facts.task_index[action.trigger_task_id]
        # An action submitted by the failing task runs after that task has
        # completed.  It cannot contribute to admission pressure at the task
        # boundary and delaying it would only perturb unrelated future work.
        if (
            error.kind == "task-device-capacity"
            and failing_task is not None
            and trigger >= failing_task
        ):
            continue
        alias = facts.alias_index[action.alias_group_id]
        consumers = facts.input_tasks[alias]
        next_consumer = bisect_right(consumers, trigger)
        latest = (
            consumers[next_consumer] - 1
            if next_consumer < len(consumers)
            else facts.last_boundary
        )
        target = trigger + 1
        if failing_task is not None and error.kind == "task-device-capacity":
            target = max(target, failing_task)
        if target > latest:
            continue
        candidates.append((-facts.alias_sizes[alias], -trigger, action_index, target))
    if not candidates:
        return None
    _size, _trigger, selected_index, target = min(candidates)
    actions = list(schedule.actions)
    actions[selected_index] = replace(
        actions[selected_index],
        trigger_task_id=facts.tasks[target].task_id,
    )
    kind_order = {"release": 0, "evict": 1, "fetch": 2}
    actions.sort(
        key=lambda action: (
            facts.task_index[action.trigger_task_id],
            kind_order[action.kind.value],
            facts.alias_index[action.alias_group_id],
        )
    )
    repaired = replace(schedule, actions=tuple(actions))
    repaired._validate_selected(facts.program, facts.tasks)
    return repaired


def _failure_diagnostic(
    spec: _CandidateSpec,
    *,
    status: str,
    kind: str,
    detail: str,
    repairs: PlanningRepairDiagnostics | None = None,
    work: PlanningWorkDiagnostics | None = None,
) -> CandidateDiagnostic:
    return CandidateDiagnostic(
        candidate_id=spec.candidate_id,
        selection_id=spec.problem.selection_id,
        status=status,
        failure_kind=kind,
        failure_detail=detail,
        repairs=repairs or PlanningRepairDiagnostics(),
        work=work or PlanningWorkDiagnostics(),
    )


def _repair_exhausted_diagnostic(
    spec: _CandidateSpec,
    error: SimulationInfeasibleError,
    repairs: PlanningRepairDiagnostics,
    work: PlanningWorkDiagnostics,
) -> CandidateDiagnostic:
    return _failure_diagnostic(
        spec,
        status="exhausted",
        kind="repair_budget_exhausted",
        detail=(
            "candidate repair budget exhausted after "
            f"{repairs.total_attempts} monotonic "
            f"repairs; last simulator result: {error}"
        ),
        repairs=repairs,
        work=work,
    )


def _evaluate_candidate(
    spec: _CandidateSpec,
    config: SimulationConfig,
    generic: GenericPlanningOptions,
    algorithm_options: PressureFitOptions,
) -> _CandidateOutcome:
    facts = spec.problem.facts
    seed = spec.problem.seed
    extra_pressure: dict[tuple[str, int], int] = {}
    candidate_started = time.perf_counter_ns()
    schedule_emissions = 0
    schedule_cache_hits = 0
    simulation_calls = 0
    simulation_cache_hits = 0
    reduce_ns = 0
    emit_ns = 0
    simulate_ns = 0
    digest_ns = 0
    simulation_fetch_delay_attempts = 0
    simulation_pressure_boundary_attempts = 0

    def repairs_value() -> PlanningRepairDiagnostics:
        return PlanningRepairDiagnostics(
            simulation_fetch_delay_attempts=(simulation_fetch_delay_attempts),
            simulation_pressure_boundary_attempts=(
                simulation_pressure_boundary_attempts
            ),
        )

    def work_value() -> PlanningWorkDiagnostics:
        total_ns = time.perf_counter_ns() - candidate_started
        named_ns = reduce_ns + emit_ns + simulate_ns + digest_ns
        return PlanningWorkDiagnostics(
            schedule_emissions=schedule_emissions,
            schedule_cache_hits=schedule_cache_hits,
            simulation_calls=simulation_calls,
            simulation_cache_hits=simulation_cache_hits,
            sections=PlanningSectionTiming(
                total_ns=total_ns,
                reduce_ns=reduce_ns,
                emit_ns=emit_ns,
                simulate_ns=simulate_ns,
                digest_ns=digest_ns,
                residual_ns=max(total_ns - named_ns, 0),
            ),
        )

    while True:
        try:
            pressure_key = tuple(
                sorted(
                    (device_id, boundary, value)
                    for (device_id, boundary), value in extra_pressure.items()
                )
            )
            residency_key = (spec.strategy, pressure_key)
            residency_started = time.perf_counter_ns()
            residency = reduce_pressure(
                facts,
                config,
                seed,
                spec.strategy,
                extra_pressure=extra_pressure,
                score_cache=spec.problem.cut_scores,
            )
            reduce_ns += time.perf_counter_ns() - residency_started
            if spec.fetch_rule == "interval-entry":
                extended = spec.problem.interval_plans.get(residency_key)
                if extended is None:
                    extended = extend_interval_entries(facts, residency)
                    spec.problem.interval_plans[residency_key] = extended
                residency = extended
            fetch_headroom = spec.strategy.startswith("headroom")
            schedule_key = (
                residency,
                spec.fetch_rule,
                spec.coalesced,
                fetch_headroom,
            )
            schedule = spec.problem.schedule_cache.get(schedule_key)
            if schedule is None:
                schedule_started = time.perf_counter_ns()
                schedule = emit_schedule(
                    facts,
                    config,
                    residency,
                    spec.fetch_rule,
                    coalesced=spec.coalesced,
                    fetch_headroom=fetch_headroom,
                )
                emit_ns += time.perf_counter_ns() - schedule_started
                schedule_emissions += 1
                spec.problem.schedule_cache[schedule_key] = schedule
            else:
                schedule_cache_hits += 1
        except PlanInfeasibleError as error:
            return _CandidateOutcome(
                spec,
                _failure_diagnostic(
                    spec,
                    status="infeasible",
                    kind=error.kind,
                    detail=str(error),
                    repairs=repairs_value(),
                    work=work_value(),
                ),
            )
        except ValidationError as error:
            return _CandidateOutcome(
                spec,
                _failure_diagnostic(
                    spec,
                    status="invalid",
                    kind="schedule_validation",
                    detail=str(error),
                    repairs=repairs_value(),
                    work=work_value(),
                ),
            )
        restart_reduction = False
        while True:
            try:
                cached_simulation = spec.problem.simulation_cache.get(schedule)
                if isinstance(cached_simulation, _CachedSimulationFailure):
                    simulation_cache_hits += 1
                    raise cached_simulation.to_error()
                if cached_simulation is None:
                    simulation_started = time.perf_counter_ns()
                    try:
                        cached_simulation = simulate_python(
                            facts.program,
                            schedule,
                            selections=facts.selections,
                            config=config,
                        )
                    except SimulationInfeasibleError as error:
                        simulate_ns += time.perf_counter_ns() - simulation_started
                        simulation_calls += 1
                        spec.problem.simulation_cache[schedule] = (
                            _CachedSimulationFailure.from_error(error)
                        )
                        raise
                    simulate_ns += time.perf_counter_ns() - simulation_started
                    simulation_calls += 1
                    spec.problem.simulation_cache[schedule] = cached_simulation
                else:
                    simulation_cache_hits += 1
                simulation = cached_simulation
            except SimulationInfeasibleError as error:
                attempts = repairs_value().total_attempts
                if attempts < algorithm_options.max_repair_attempts:
                    delayed = _delay_fetch(facts, schedule, error)
                    if delayed is not None and delayed != schedule:
                        schedule = delayed
                        simulation_fetch_delay_attempts += 1
                        continue
                    repair = _repair_pressure(facts, error)
                    if repair is not None:
                        boundary, extra = repair
                        # The analytic plan already satisfied any pressure
                        # previously recorded at this boundary.  A new
                        # simulator failure therefore describes additional,
                        # not replacement, overlap pressure (normally an
                        # admitted fetch destination).  Accumulate it so a
                        # restarted reduction cannot reproduce the same plan
                        # forever.
                        extra_pressure[boundary] = (
                            extra_pressure.get(boundary, 0) + extra
                        )
                        simulation_pressure_boundary_attempts += 1
                        restart_reduction = True
                        break
                elif (
                    _delay_fetch(facts, schedule, error) is not None
                    or _repair_pressure(facts, error) is not None
                ):
                    return _CandidateOutcome(
                        spec,
                        _repair_exhausted_diagnostic(
                            spec, error, repairs_value(), work_value()
                        ),
                    )
                return _CandidateOutcome(
                    spec,
                    _failure_diagnostic(
                        spec,
                        status="infeasible",
                        kind=error.kind,
                        detail=str(error),
                        repairs=repairs_value(),
                        work=work_value(),
                    ),
                )
            break
        if restart_reduction:
            continue
        digest_started = time.perf_counter_ns()
        schedule_digest = schedule.digest
        digest_ns += time.perf_counter_ns() - digest_started
        return _CandidateOutcome(
            spec,
            CandidateDiagnostic(
                candidate_id=spec.candidate_id,
                selection_id=spec.problem.selection_id,
                status="valid",
                makespan_ns=simulation.makespan_ns,
                schedule_digest=schedule_digest,
                repairs=repairs_value(),
                work=work_value(),
            ),
            schedule,
            simulation,
        )


def _build_problems(
    program: ShadowSpillProgram,
    initial_residency: tuple[ResidencySpec, ...],
    final_residency: tuple[ResidencySpec, ...],
    config: SimulationConfig,
    generic: GenericPlanningOptions,
    algorithm_options: PressureFitOptions,
    *,
    resolved: tuple[tuple[TaskAlternativeChoice, ...], ...],
    progress: Callable[[str], None] | None,
) -> tuple[_SelectionProblem, ...]:
    problems: list[_SelectionProblem] = []
    failures: list[PlanInfeasibleError] = []
    started = time.perf_counter_ns()
    for selection_index, selections in enumerate(resolved, start=1):
        try:
            facts = build_facts(
                program,
                selections,
                initial_residency,
                final_residency,
                config,
            )
            assert_required_floor(facts)
        except PlanInfeasibleError as error:
            failures.append(error)
            continue
        seed = seed_residency(
            facts,
            config,
            algorithm_options.initial_placement,
            # Initial placement is a property of the program and public
            # capacity, not a later strategy's speculative headroom.
            initial_capacity_by_device=facts.object_capacity_by_device,
        )
        problems.append(
            _SelectionProblem(
                selections,
                _selection_id(selections),
                facts,
                seed,
            )
        )
        if progress is not None:
            progress(
                "PressureFit problem "
                f"{selection_index}/{len(resolved)}: "
                f"tasks={len(facts.tasks)}, aliases={len(facts.alias_ids)}, "
                f"elapsed={(time.perf_counter_ns() - started) / 1e9:.3f}s"
            )
    if problems:
        return tuple(problems)
    if failures:
        raise failures[0]
    raise PlanInfeasibleError(
        "no resolution could be constructed",
        kind="graph_pair_selection",
    )


def _candidate_specs(
    problems: tuple[_SelectionProblem, ...],
    generic: GenericPlanningOptions,
    algorithm_options: PressureFitOptions,
) -> tuple[_CandidateSpec, ...]:
    specs: list[_CandidateSpec] = []
    ordinal = 0
    coalescing = (False, True) if algorithm_options.evaluate_coalesced else (False,)
    for problem in problems:
        for strategy in algorithm_options.residency_strategies:
            for rule in algorithm_options.fetch_rules:
                for coalesced in coalescing:
                    specs.append(
                        _CandidateSpec(
                            ordinal,
                            problem,
                            strategy,
                            rule,
                            coalesced,
                        )
                    )
                    ordinal += 1
    return tuple(specs)


def _run_candidates(
    specs: tuple[_CandidateSpec, ...],
    config: SimulationConfig,
    generic: GenericPlanningOptions,
    algorithm_options: PressureFitOptions,
    workers: int,
) -> tuple[_CandidateOutcome, ...]:
    if workers == 1 or len(specs) <= 1:
        return tuple(
            _evaluate_candidate(spec, config, generic, algorithm_options)
            for spec in specs
        )
    batches: list[list[_CandidateSpec]] = []
    for spec in specs:
        if not batches or batches[-1][0].problem is not spec.problem:
            batches.append([])
        batches[-1].append(spec)

    def evaluate_batch(batch: list[_CandidateSpec]) -> tuple[_CandidateOutcome, ...]:
        return tuple(
            _evaluate_candidate(spec, config, generic, algorithm_options)
            for spec in batch
        )

    thread_count = None if workers == 0 else workers
    with ThreadPoolExecutor(max_workers=thread_count) as executor:
        results = tuple(
            outcome
            for batch_results in executor.map(evaluate_batch, batches)
            for outcome in batch_results
        )
    return tuple(sorted(results, key=lambda item: item.spec.ordinal))


def _pressurefit_once(
    program: ShadowSpillProgram,
    *,
    initial_residency: tuple[ResidencySpec, ...],
    final_residency: tuple[ResidencySpec, ...] = (),
    config: SimulationConfig,
    generic: GenericPlanningOptions | None = None,
    algorithm_options: PressureFitOptions | None = None,
    workers: int = 0,
    admission: AdmissionFacts | None = None,
    progress: Callable[[str], None] | None = None,
) -> ProgramPlanResult:
    """Plan residency, movement, and recomputation for a validated program.

    Every returned schedule has been accepted by the public simulator. Planner
    heuristics may reject candidates, but they never weaken simulator checks or
    move an action after candidate selection.
    """

    if not isinstance(program, ShadowSpillProgram):
        raise TypeError("program must be a ShadowSpillProgram")
    if not isinstance(initial_residency, tuple):
        raise TypeError("initial_residency must be a tuple")
    if not isinstance(final_residency, tuple):
        raise TypeError("final_residency must be a tuple")
    if not isinstance(config, SimulationConfig):
        raise TypeError("config must be a SimulationConfig")
    if admission is not None:
        if not isinstance(admission, AdmissionFacts):
            raise TypeError("admission must be an AdmissionFacts")
        admission.validate(program)
    selected_options = generic or GenericPlanningOptions()
    selected_search = algorithm_options or PressureFitOptions()
    resolved = resolutions(program, selected_search.resolution_options)
    if progress is not None:
        progress(
            "PressureFit resolved: "
            f"groups={len(program.task_alternative_groups)}, "
            f"selections={len(resolved)}"
        )
    problems_started = time.perf_counter_ns()
    problems = _build_problems(
        program,
        initial_residency,
        final_residency,
        config,
        selected_options,
        selected_search,
        resolved=resolved,
        progress=progress,
    )
    if progress is not None:
        progress(
            "PressureFit problems ready: "
            f"valid={len(problems)}/{len(resolved)}, "
            f"elapsed={(time.perf_counter_ns() - problems_started) / 1e9:.3f}s"
        )
    specs = _candidate_specs(problems, selected_options, selected_search)
    if progress is not None:
        progress(
            "PressureFit candidates: "
            f"count={len(specs)}, per_problem={len(specs) // len(problems)}"
        )
    candidates_started = time.perf_counter_ns()
    if workers == 1 or len(specs) <= 1:
        outcomes_list: list[_CandidateOutcome] = []
        per_problem = len(specs) // len(problems)
        for index, spec in enumerate(specs, start=1):
            outcomes_list.append(
                _evaluate_candidate(spec, config, selected_options, selected_search)
            )
            if progress is not None and (
                index % per_problem == 0 or index == len(specs)
            ):
                batch = outcomes_list[-per_problem:]
                progress(
                    "PressureFit candidate problem "
                    f"{index // per_problem}/{len(problems)}: "
                    f"valid={sum(item.schedule is not None for item in batch)}, "
                    "repairs="
                    f"{sum(item.diagnostic.repairs.total_attempts for item in batch)}, "
                    "elapsed="
                    f"{(time.perf_counter_ns() - candidates_started) / 1e9:.3f}s"
                )
        outcomes = tuple(outcomes_list)
    else:
        outcomes = _run_candidates(
            specs, config, selected_options, selected_search, workers
        )
        if progress is not None:
            progress(
                "PressureFit parallel candidates finished: "
                f"elapsed={(time.perf_counter_ns() - candidates_started) / 1e9:.3f}s"
            )
    valid = tuple(
        outcome
        for outcome in outcomes
        if outcome.schedule is not None and outcome.simulation is not None
    )
    if not valid:
        failure_diagnostics = tuple(outcome.diagnostic for outcome in outcomes)
        if any(item.status == "exhausted" for item in failure_diagnostics):
            raise PlanSearchExhaustedError(
                "PressureFit exhausted its bounded candidate-repair budget "
                "before proving a feasible schedule",
                diagnostics=failure_diagnostics,
            )
        first = failure_diagnostics[0] if failure_diagnostics else None
        raise PlanInfeasibleError(
            "no simulator-valid PressureFit candidate satisfied the declared "
            "capacity and residency constraints",
            kind=first.failure_kind if first and first.failure_kind else "no_candidate",
            diagnostics=failure_diagnostics,
        )
    best = min(
        valid,
        key=lambda outcome: (
            outcome.simulation.makespan_ns,  # type: ignore[union-attr]
            outcome.spec.ordinal,
        ),
    )
    assert best.schedule is not None
    assert best.simulation is not None
    final_simulation = best.simulation
    problem_diagnostics: list[ResolvedProgramDiagnostics] = []
    aggregate_work = PlanningWorkDiagnostics()
    for problem in problems:
        problem_outcomes = tuple(
            outcome for outcome in outcomes if outcome.spec.problem is problem
        )
        problem_candidates = tuple(outcome.diagnostic for outcome in problem_outcomes)
        problem_valid = tuple(
            outcome
            for outcome in problem_outcomes
            if outcome.schedule is not None and outcome.simulation is not None
        )
        problem_best = (
            None
            if not problem_valid
            else min(
                problem_valid,
                key=lambda outcome: (
                    outcome.simulation.makespan_ns,  # type: ignore[union-attr]
                    outcome.spec.ordinal,
                ),
            )
        )
        problem_work = PlanningWorkDiagnostics()
        for candidate in problem_candidates:
            problem_work += candidate.work
        aggregate_work += problem_work
        problem_diagnostics.append(
            ResolvedProgramDiagnostics(
                selection_id=problem.selection_id,
                choices=tuple(
                    TaskAlternativeChoiceDiagnostic(item.group_id, item.option_id)
                    for item in problem.selections
                ),
                selected_candidate_id=(
                    None if problem_best is None else problem_best.spec.candidate_id
                ),
                selected_makespan_ns=(
                    None
                    if problem_best is None
                    else problem_best.simulation.makespan_ns  # type: ignore[union-attr]
                ),
                candidate_evaluations=problem_candidates,
                work=problem_work,
            )
        )
    diagnostics = PlanningDiagnostics(
        selected_candidate_id=best.spec.candidate_id,
        selected_selection_id=best.spec.problem.selection_id,
        selected_makespan_ns=final_simulation.makespan_ns,
        resolved_programs=tuple(problem_diagnostics),
        work=aggregate_work,
    )
    return ProgramPlanResult(
        program=program,
        search_options=SearchOptions(
            generic=selected_options, algorithm=PressureFit(selected_search)
        ),
        initial_residency=initial_residency,
        final_residency=final_residency,
        simulation_config=config,
        schedule=best.schedule,
        selections=best.spec.problem.selections,
        simulation=final_simulation,
        diagnostics=diagnostics,
        admission_facts=admission,
    )


def pressurefit(
    program: ShadowSpillProgram,
    *,
    initial_residency: tuple[ResidencySpec, ...],
    final_residency: tuple[ResidencySpec, ...] = (),
    config: SimulationConfig,
    generic: GenericPlanningOptions | None = None,
    algorithm_options: PressureFitOptions | None = None,
    workers: int = 0,
    admission: AdmissionFacts | None = None,
    progress: Callable[[str], None] | None = None,
) -> ProgramPlanResult:
    """Select a schedule at the caller's object capacity.

    Selection runs once.  The global capacity ladder this used to describe -
    reducing object capacity and repeating the whole selection - is retired;
    a candidate that misses exact dynamic ``MemoryPool`` admission now refines
    within the search, so physical pool capacity, task semantics, and action
    rules are unchanged either way.
    """

    # Preserve the framework-neutral semantic diagnostics before entering the
    # required compiled search. This validates caller input; it is not an
    # alternate planner or simulator execution path.
    generic = generic or GenericPlanningOptions()
    algorithm_options = algorithm_options or PressureFitOptions()
    validate_schedule_feasibility(
        program,
        initial_residency=initial_residency,
        final_residency=final_residency,
        config=config,
        admission=admission,
        algorithm_options=algorithm_options,
    )
    result = _pressurefit_once(
        program,
        initial_residency=initial_residency,
        final_residency=final_residency,
        config=config,
        generic=generic,
        algorithm_options=algorithm_options,
        workers=workers,
        admission=admission,
        progress=progress,
    )
    return replace(
        result,
        diagnostics=replace(
            result.diagnostics,
            effective_object_capacity_bytes=(
                None if admission is None else admission.object_capacity_bytes
            ),
        ),
        admission_facts=admission,
    )


__all__ = ["pressurefit", "validate_schedule_feasibility"]
