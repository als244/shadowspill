"""Deterministic, simulator-verified PressureFit orchestration."""

from __future__ import annotations

import time
from bisect import bisect_right
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace

from reference.python.simulator import simulate_python
from shadowspill.ir import (
    MemorySchedule,
    Program,
    RecomputationSelection,
    ResidencySpec,
)
from shadowspill.ir.validation import ValidationError
from shadowspill.planner.admission import AdmissionFacts
from shadowspill.planner.diagnostics import (
    AdmissionRefinement,
    CandidateDiagnostic,
    PressureFitDiagnostics,
    PressureFitRepairDiagnostics,
    PressureFitWorkDiagnostics,
    RecomputationChoiceDiagnostic,
    RecomputationProblemDiagnostics,
)
from shadowspill.planner.recomputation import build_recomputation_portfolio
from shadowspill.planner.request import PressureFitOptions
from shadowspill.planner.result import (
    PressureFitInfeasibleError,
    PressureFitResult,
    PressureFitSearchExhaustedError,
)
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


def _scheduled_admission_refinement(attempt: int) -> int:
    """Return the deterministic reserve increment for one failed admission.

    Early attempts double from 128 MiB through 1 GiB so high-pressure plans
    converge quickly.  Later attempts grow linearly by 512 MiB, avoiding the
    multi-GiB jumps that would discard too much otherwise usable capacity.
    ``attempt`` is zero-based.
    """

    doubled = _ADMISSION_INITIAL_REFINEMENT_BYTES << attempt
    if doubled <= _ADMISSION_DOUBLING_LIMIT_BYTES:
        return doubled
    attempts_after_limit = attempt - 3
    return (
        _ADMISSION_DOUBLING_LIMIT_BYTES
        + attempts_after_limit * _ADMISSION_LINEAR_REFINEMENT_BYTES
    )


@dataclass(frozen=True, slots=True)
class _SelectionProblem:
    selections: tuple[RecomputationSelection, ...]
    selection_id: str
    facts: PlanningFacts
    seed: ResidencyPlan
    cut_scores: dict[tuple[Cut, str], tuple[int, ...]] = field(
        default_factory=dict,
        compare=False,
        repr=False,
    )
    residency_plans: dict[
        tuple[str, tuple[tuple[str, int, int], ...]], ResidencyPlan
    ] = field(
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
    prefetch_rule: str
    coalesced: bool

    @property
    def candidate_id(self) -> str:
        suffix = "-coalesced" if self.coalesced else ""
        return f"{self.strategy}/{self.prefetch_rule}{suffix}"


@dataclass(frozen=True, slots=True)
class _CandidateOutcome:
    spec: _CandidateSpec
    diagnostic: CandidateDiagnostic
    schedule: MemorySchedule | None = None
    simulation: SimulationResult | None = None


def _selection_id(selections: tuple[RecomputationSelection, ...]) -> str:
    if not selections:
        return "none"
    return ",".join(f"{item.group_id}={item.option_id}" for item in selections)


def validate_schedule_feasibility(
    program: Program,
    *,
    initial_residency: tuple[ResidencySpec, ...],
    final_residency: tuple[ResidencySpec, ...] = (),
    config: SimulationConfig,
    admission: AdmissionFacts | None = None,
) -> None:
    """Reject irreducible capacity failures before schedule search.

    This is a necessary-condition preflight, not a PressureFit candidate
    search. It accepts when at least one legal recomputation selection has a
    task-by-task residency floor that fits the declared capacity. PressureFit
    retains the same checks internally as defensive invariants.
    """

    if not isinstance(program, Program):
        raise TypeError("program must be a Program")
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

    failures: list[PressureFitInfeasibleError] = []
    for selections in build_recomputation_portfolio(program):
        try:
            facts = build_facts(
                program,
                selections,
                initial_residency,
                final_residency,
                config,
            )
            assert_required_floor(facts)
        except PressureFitInfeasibleError as error:
            failures.append(error)
        else:
            return

    if failures:
        raise failures[0]
    raise PressureFitInfeasibleError(
        "no recomputation selection could be constructed",
        kind="recomputation_selection",
    )


def _repair_pressure(
    facts: PlanningFacts,
    error: SimulationInfeasibleError,
) -> tuple[tuple[str, int], int] | None:
    if error.kind not in {
        "initial-device-capacity",
        "prefetch-device-capacity",
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


def _delay_prefetch(
    facts: PlanningFacts,
    schedule: MemorySchedule,
    error: SimulationInfeasibleError,
) -> MemorySchedule | None:
    if error.kind not in {"prefetch-device-capacity", "task-device-capacity"}:
        return None
    failing_task = (
        facts.task_index.get(error.task_id) if error.task_id is not None else None
    )
    requested_alias = error.alias_group_ids[0] if error.alias_group_ids else None
    candidates: list[tuple[int, int, int, int]] = []
    for action_index, action in enumerate(schedule.actions):
        if action.kind.value != "prefetch":
            continue
        if requested_alias is not None and action.alias_group_id != requested_alias:
            continue
        if (
            error.kind == "prefetch-device-capacity"
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
    kind_order = {"release": 0, "offload": 1, "prefetch": 2}
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
    repairs: PressureFitRepairDiagnostics | None = None,
    work: PressureFitWorkDiagnostics | None = None,
) -> CandidateDiagnostic:
    return CandidateDiagnostic(
        candidate_id=spec.candidate_id,
        selection_id=spec.problem.selection_id,
        status=status,
        failure_kind=kind,
        failure_detail=detail,
        repairs=repairs or PressureFitRepairDiagnostics(),
        work=work or PressureFitWorkDiagnostics(),
    )


def _repair_exhausted_diagnostic(
    spec: _CandidateSpec,
    error: SimulationInfeasibleError,
    repairs: PressureFitRepairDiagnostics,
    work: PressureFitWorkDiagnostics,
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
    options: PressureFitOptions,
) -> _CandidateOutcome:
    facts = spec.problem.facts
    seed = spec.problem.seed
    extra_pressure: dict[tuple[str, int], int] = {}
    candidate_started = time.perf_counter_ns()
    residency_cache_hits = 0
    residency_cache_misses = 0
    schedule_emissions = 0
    schedule_cache_hits = 0
    simulation_calls = 0
    simulation_cache_hits = 0
    residency_time_ns = 0
    schedule_time_ns = 0
    simulation_time_ns = 0
    digest_time_ns = 0
    simulation_prefetch_delay_attempts = 0
    simulation_pressure_boundary_attempts = 0

    def repairs_value() -> PressureFitRepairDiagnostics:
        return PressureFitRepairDiagnostics(
            simulation_prefetch_delay_attempts=(simulation_prefetch_delay_attempts),
            simulation_pressure_boundary_attempts=(
                simulation_pressure_boundary_attempts
            ),
        )

    def work_value() -> PressureFitWorkDiagnostics:
        return PressureFitWorkDiagnostics(
            evaluation_time_ns=time.perf_counter_ns() - candidate_started,
            residency_cache_hits=residency_cache_hits,
            residency_cache_misses=residency_cache_misses,
            schedule_emissions=schedule_emissions,
            schedule_cache_hits=schedule_cache_hits,
            simulation_calls=simulation_calls,
            simulation_cache_hits=simulation_cache_hits,
            residency_time_ns=residency_time_ns,
            schedule_time_ns=schedule_time_ns,
            simulation_time_ns=simulation_time_ns,
            digest_time_ns=digest_time_ns,
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
            residency = spec.problem.residency_plans.get(residency_key)
            if residency is None:
                residency_cache_misses += 1
                residency_started = time.perf_counter_ns()
                residency = reduce_pressure(
                    facts,
                    config,
                    seed,
                    spec.strategy,
                    extra_pressure=extra_pressure,
                    score_cache=spec.problem.cut_scores,
                )
                residency_time_ns += time.perf_counter_ns() - residency_started
                spec.problem.residency_plans[residency_key] = residency
            else:
                residency_cache_hits += 1
            if spec.prefetch_rule == "interval-entry":
                extended = spec.problem.interval_plans.get(residency_key)
                if extended is None:
                    extended = extend_interval_entries(facts, residency)
                    spec.problem.interval_plans[residency_key] = extended
                residency = extended
            prefetch_headroom = spec.strategy.startswith("headroom")
            schedule_key = (
                residency,
                spec.prefetch_rule,
                spec.coalesced,
                prefetch_headroom,
            )
            schedule = spec.problem.schedule_cache.get(schedule_key)
            if schedule is None:
                schedule_started = time.perf_counter_ns()
                schedule = emit_schedule(
                    facts,
                    config,
                    residency,
                    spec.prefetch_rule,
                    coalesced=spec.coalesced,
                    prefetch_headroom=prefetch_headroom,
                )
                schedule_time_ns += time.perf_counter_ns() - schedule_started
                schedule_emissions += 1
                spec.problem.schedule_cache[schedule_key] = schedule
            else:
                schedule_cache_hits += 1
        except PressureFitInfeasibleError as error:
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
                        simulation_time_ns += (
                            time.perf_counter_ns() - simulation_started
                        )
                        simulation_calls += 1
                        spec.problem.simulation_cache[schedule] = (
                            _CachedSimulationFailure.from_error(error)
                        )
                        raise
                    simulation_time_ns += time.perf_counter_ns() - simulation_started
                    simulation_calls += 1
                    spec.problem.simulation_cache[schedule] = cached_simulation
                else:
                    simulation_cache_hits += 1
                simulation = cached_simulation
            except SimulationInfeasibleError as error:
                if repairs_value().total_attempts < options.max_repair_attempts:
                    delayed = _delay_prefetch(facts, schedule, error)
                    if delayed is not None and delayed != schedule:
                        schedule = delayed
                        simulation_prefetch_delay_attempts += 1
                        continue
                    repair = _repair_pressure(facts, error)
                    if repair is not None:
                        boundary, extra = repair
                        # The analytic plan already satisfied any pressure
                        # previously recorded at this boundary.  A new
                        # simulator failure therefore describes additional,
                        # not replacement, overlap pressure (normally an
                        # admitted prefetch destination).  Accumulate it so a
                        # restarted reduction cannot reproduce the same plan
                        # forever.
                        extra_pressure[boundary] = (
                            extra_pressure.get(boundary, 0) + extra
                        )
                        simulation_pressure_boundary_attempts += 1
                        restart_reduction = True
                        break
                elif (
                    _delay_prefetch(facts, schedule, error) is not None
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
        digest_time_ns += time.perf_counter_ns() - digest_started
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
    program: Program,
    initial_residency: tuple[ResidencySpec, ...],
    final_residency: tuple[ResidencySpec, ...],
    config: SimulationConfig,
    options: PressureFitOptions,
    *,
    portfolio: tuple[tuple[RecomputationSelection, ...], ...],
    progress: Callable[[str], None] | None,
) -> tuple[_SelectionProblem, ...]:
    problems: list[_SelectionProblem] = []
    failures: list[PressureFitInfeasibleError] = []
    started = time.perf_counter_ns()
    for selection_index, selections in enumerate(portfolio, start=1):
        try:
            facts = build_facts(
                program,
                selections,
                initial_residency,
                final_residency,
                config,
            )
            assert_required_floor(facts)
        except PressureFitInfeasibleError as error:
            failures.append(error)
            continue
        seed = seed_residency(
            facts,
            config,
            options.initial_placement,
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
                f"{selection_index}/{len(portfolio)}: "
                f"tasks={len(facts.tasks)}, aliases={len(facts.alias_ids)}, "
                f"elapsed={(time.perf_counter_ns() - started) / 1e9:.3f}s"
            )
    if problems:
        return tuple(problems)
    if failures:
        raise failures[0]
    raise PressureFitInfeasibleError(
        "no recomputation selection could be constructed",
        kind="recomputation_selection",
    )


def _candidate_specs(
    problems: tuple[_SelectionProblem, ...],
    options: PressureFitOptions,
) -> tuple[_CandidateSpec, ...]:
    specs: list[_CandidateSpec] = []
    ordinal = 0
    coalescing = (False, True) if options.evaluate_coalesced else (False,)
    for problem in problems:
        for strategy in options.residency_strategies:
            for rule in options.prefetch_rules:
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
    options: PressureFitOptions,
) -> tuple[_CandidateOutcome, ...]:
    if options.workers == 1 or len(specs) <= 1:
        return tuple(_evaluate_candidate(spec, config, options) for spec in specs)
    batches: list[list[_CandidateSpec]] = []
    for spec in specs:
        if not batches or batches[-1][0].problem is not spec.problem:
            batches.append([])
        batches[-1].append(spec)

    def evaluate_batch(batch: list[_CandidateSpec]) -> tuple[_CandidateOutcome, ...]:
        return tuple(_evaluate_candidate(spec, config, options) for spec in batch)

    workers = None if options.workers == 0 else options.workers
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = tuple(
            outcome
            for batch_results in executor.map(evaluate_batch, batches)
            for outcome in batch_results
        )
    return tuple(sorted(results, key=lambda item: item.spec.ordinal))


def _pressurefit_once(
    program: Program,
    *,
    initial_residency: tuple[ResidencySpec, ...],
    final_residency: tuple[ResidencySpec, ...] = (),
    config: SimulationConfig,
    options: PressureFitOptions | None = None,
    admission: AdmissionFacts | None = None,
    progress: Callable[[str], None] | None = None,
) -> PressureFitResult:
    """Plan residency, movement, and recomputation for a validated program.

    Every returned schedule has been accepted by the public simulator. Planner
    heuristics may reject candidates, but they never weaken simulator checks or
    move an action after candidate selection.
    """

    if not isinstance(program, Program):
        raise TypeError("program must be a Program")
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
    selected_options = options or PressureFitOptions()
    portfolio = build_recomputation_portfolio(program)
    if progress is not None:
        progress(
            "PressureFit portfolio: "
            f"groups={len(program.recomputation_groups)}, "
            f"selections={len(portfolio)}"
        )
    problems_started = time.perf_counter_ns()
    problems = _build_problems(
        program,
        initial_residency,
        final_residency,
        config,
        selected_options,
        portfolio=portfolio,
        progress=progress,
    )
    if progress is not None:
        progress(
            "PressureFit problems ready: "
            f"valid={len(problems)}/{len(portfolio)}, "
            f"elapsed={(time.perf_counter_ns() - problems_started) / 1e9:.3f}s"
        )
    specs = _candidate_specs(problems, selected_options)
    if progress is not None:
        progress(
            "PressureFit candidates: "
            f"count={len(specs)}, per_problem={len(specs) // len(problems)}"
        )
    candidates_started = time.perf_counter_ns()
    if selected_options.workers == 1 or len(specs) <= 1:
        outcomes_list: list[_CandidateOutcome] = []
        per_problem = len(specs) // len(problems)
        for index, spec in enumerate(specs, start=1):
            outcomes_list.append(_evaluate_candidate(spec, config, selected_options))
            if progress is not None and (
                index % per_problem == 0 or index == len(specs)
            ):
                batch = outcomes_list[-per_problem:]
                progress(
                    "PressureFit candidate problem "
                    f"{index // per_problem}/{len(problems)}: "
                    f"valid={sum(item.schedule is not None for item in batch)}, "
                    "repairs="
                    f"{sum(item.diagnostic.repair_attempts for item in batch)}, "
                    "elapsed="
                    f"{(time.perf_counter_ns() - candidates_started) / 1e9:.3f}s"
                )
        outcomes = tuple(outcomes_list)
    else:
        outcomes = _run_candidates(specs, config, selected_options)
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
            raise PressureFitSearchExhaustedError(
                "PressureFit exhausted its bounded candidate-repair budget "
                "before proving a feasible schedule",
                diagnostics=failure_diagnostics,
            )
        first = failure_diagnostics[0] if failure_diagnostics else None
        raise PressureFitInfeasibleError(
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
    problem_diagnostics: list[RecomputationProblemDiagnostics] = []
    aggregate_work = PressureFitWorkDiagnostics()
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
        problem_work = PressureFitWorkDiagnostics()
        for candidate in problem_candidates:
            problem_work += candidate.work
        aggregate_work += problem_work
        problem_diagnostics.append(
            RecomputationProblemDiagnostics(
                selection_id=problem.selection_id,
                choices=tuple(
                    RecomputationChoiceDiagnostic(item.group_id, item.option_id)
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
    diagnostics = PressureFitDiagnostics(
        selected_candidate_id=best.spec.candidate_id,
        selected_selection_id=best.spec.problem.selection_id,
        selected_makespan_ns=final_simulation.makespan_ns,
        recomputation_problems=tuple(problem_diagnostics),
        work=aggregate_work,
    )
    return PressureFitResult(
        program=program,
        options=selected_options,
        initial_residency=initial_residency,
        final_residency=final_residency,
        simulation_config=config,
        schedule=best.schedule,
        selections=best.spec.problem.selections,
        simulation=final_simulation,
        diagnostics=diagnostics,
        admission_facts=admission,
    )


def _round_up_admission_reserve(value: int) -> int:
    granularity = _ADMISSION_RESERVE_GRANULARITY_BYTES
    return ((value + granularity - 1) // granularity) * granularity


def _with_object_capacity(
    config: SimulationConfig,
    admission: AdmissionFacts,
    capacity_bytes: int,
) -> tuple[SimulationConfig, AdmissionFacts]:
    devices = tuple(
        replace(device, capacity_bytes=capacity_bytes)
        if device.device_id == admission.device_id
        else device
        for device in config.devices
    )
    if devices == config.devices:
        raise ValueError(
            f"admission device {admission.device_id!r} is absent from simulation"
        )
    return (
        replace(config, devices=devices),
        replace(admission, object_capacity_bytes=capacity_bytes),
    )


def pressurefit(
    program: Program,
    *,
    initial_residency: tuple[ResidencySpec, ...],
    final_residency: tuple[ResidencySpec, ...] = (),
    config: SimulationConfig,
    options: PressureFitOptions | None = None,
    admission: AdmissionFacts | None = None,
    progress: Callable[[str], None] | None = None,
) -> PressureFitResult:
    """Select a schedule and monotonically refine dynamic-slab headroom.

    PressureFit first uses the caller's conservative object capacity.  If all
    logically valid schedules fail exact dynamic ``MemoryPool`` admission,
    object capacity is reduced by at least 128 MiB.  The increment doubles on
    every subsequent failure and is never smaller than the measured contiguous
    deficit rounded to allocator granularity.  Increments double through 1 GiB
    and then grow by 512 MiB per attempt. Selection then repeats without
    changing physical pool capacity, task semantics, or action rules.
    """

    # Preserve the framework-neutral semantic diagnostics before entering the
    # required compiled search. This validates caller input; it is not an
    # alternate planner or simulator execution path.
    validate_schedule_feasibility(
        program,
        initial_residency=initial_residency,
        final_residency=final_residency,
        config=config,
        admission=admission,
    )
    original_config = config
    current_config = config
    current_admission = admission
    refinements: list[AdmissionRefinement] = []
    while True:
        try:
            result = _pressurefit_once(
                program,
                initial_residency=initial_residency,
                final_residency=final_residency,
                config=current_config,
                options=options,
                admission=current_admission,
                progress=progress,
            )
            effective_capacity = (
                None
                if current_admission is None
                else current_admission.object_capacity_bytes
            )
            return replace(
                result,
                # Preserve the public call boundary for cache identity.  The
                # exact effective capacity is recorded below, while physical
                # simulation uses AdmissionFacts.pool_capacity_bytes.
                simulation_config=original_config,
                diagnostics=replace(
                    result.diagnostics,
                    admission_refinements=tuple(refinements),
                    effective_object_capacity_bytes=effective_capacity,
                ),
                admission_facts=admission,
            )
        except PressureFitInfeasibleError as error:
            if (
                current_admission is None
                or error.kind != "physical_admission"
                or error.required_bytes is None
                or error.required_bytes <= 0
            ):
                raise
            previous = current_admission.object_capacity_bytes
            scheduled_increment = _scheduled_admission_refinement(len(refinements))
            increment = max(
                _round_up_admission_reserve(error.required_bytes),
                scheduled_increment,
            )
            capacity = previous - increment
            if capacity <= 0:
                raise
            refinements.append(
                AdmissionRefinement(
                    attempt=len(refinements) + 1,
                    previous_object_capacity_bytes=previous,
                    required_additional_slack_bytes=error.required_bytes,
                    reserve_increment_bytes=increment,
                    object_capacity_bytes=capacity,
                )
            )
            if progress is not None:
                progress(
                    "PressureFit physical admission refinement "
                    f"{len(refinements)}: required_slack={error.required_bytes}, "
                    f"reserve_increment={increment}, "
                    f"object_capacity={capacity}"
                )
            current_config, current_admission = _with_object_capacity(
                current_config,
                current_admission,
                capacity,
            )


__all__ = ["pressurefit", "validate_schedule_feasibility"]
