"""Deterministic, simulator-verified PressureFit orchestration."""

from __future__ import annotations

from bisect import bisect_right
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from itertools import product

from shadowspill.ir import (
    MemorySchedule,
    Program,
    RecomputationSelection,
    ResidencySpec,
)
from shadowspill.ir._validation import ValidationError
from shadowspill.simulator import (
    SimulationConfig,
    SimulationInfeasibleError,
    SimulationResult,
    simulate,
)
from shadowspill.simulator._capi import simulator_library_path
from shadowspill.simulator._compiled import (
    CompiledSimulationSummary,
    CompiledSimulationTemplate,
    compile_simulation_template,
    simulate_compiled_template,
    simulate_compiled_template_summary,
)

from ._actions import emit_schedule
from ._capi import planner_library_path
from ._dense_residency import (
    CompiledResidencyTemplate,
    compile_residency_template,
    reduce_residency_compiled,
)
from ._facts import PlanningFacts, build_facts
from ._residency import (
    Cut,
    ResidencyPlan,
    assert_required_floor,
    extend_interval_entries,
    reduce_pressure,
    seed_residency,
)
from .model import (
    CandidateDiagnostic,
    PressureFitDiagnostics,
    PressureFitInfeasibleError,
    PressureFitOptions,
    PressureFitResult,
)


@dataclass(frozen=True, slots=True)
class _SelectionContext:
    selections: tuple[RecomputationSelection, ...]
    selection_id: str
    facts: PlanningFacts
    seed: ResidencyPlan
    compiled_template: CompiledSimulationTemplate | None
    compiled_residency: CompiledResidencyTemplate | None
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
        SimulationResult | CompiledSimulationSummary | _CachedSimulationFailure,
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
    context: _SelectionContext
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
    simulation: SimulationResult | CompiledSimulationSummary | None = None


def _selection_id(selections: tuple[RecomputationSelection, ...]) -> str:
    if not selections:
        return "none"
    return ",".join(f"{item.group_id}={item.option_id}" for item in selections)


def _selection_portfolio(
    program: Program,
) -> tuple[tuple[RecomputationSelection, ...], ...]:
    groups = program.recomputation_groups
    if not groups:
        return ((),)
    option_counts = [len(group.options) for group in groups]
    combination_count = 1
    for count in option_counts:
        combination_count *= count
    raw: list[tuple[int, ...]] = []
    if combination_count <= 64:
        raw.extend(product(*(range(count) for count in option_counts)))
    else:
        first = tuple(0 for _ in groups)
        last = tuple(count - 1 for count in option_counts)
        raw.extend((first, last))
        for split in range(len(groups) + 1):
            raw.append(
                tuple(
                    0 if index < split else last[index] for index in range(len(groups))
                )
            )
            raw.append(
                tuple(
                    last[index] if index < split else 0 for index in range(len(groups))
                )
            )
        for group_index, count in enumerate(option_counts):
            for option_index in range(count):
                changed_first = list(first)
                changed_first[group_index] = option_index
                raw.append(tuple(changed_first))
                changed_last = list(last)
                changed_last[group_index] = option_index
                raw.append(tuple(changed_last))

    unique: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for value in raw:
        if value not in seen:
            seen.add(value)
            unique.append(value)
    return tuple(
        tuple(
            RecomputationSelection(
                group.group_id,
                group.options[option_index].option_id,
            )
            for group, option_index in zip(groups, indices, strict=True)
        )
        for indices in unique
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
    capacity = error.capacity_bytes or facts.object_capacity_by_device[device_id]
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
    repairs: int = 0,
) -> CandidateDiagnostic:
    return CandidateDiagnostic(
        candidate_id=spec.candidate_id,
        selection_id=spec.context.selection_id,
        status=status,
        failure_kind=kind,
        failure_detail=detail,
        repair_attempts=repairs,
    )


def _evaluate_candidate(
    spec: _CandidateSpec,
    config: SimulationConfig,
    options: PressureFitOptions,
) -> _CandidateOutcome:
    facts = spec.context.facts
    seed = spec.context.seed
    extra_pressure: dict[tuple[str, int], int] = {}
    repairs = 0
    while True:
        try:
            pressure_key = tuple(
                sorted(
                    (device_id, boundary, value)
                    for (device_id, boundary), value in extra_pressure.items()
                )
            )
            residency_key = (spec.strategy, pressure_key)
            residency = spec.context.residency_plans.get(residency_key)
            if residency is None:
                if spec.context.compiled_residency is not None:
                    residency = reduce_residency_compiled(
                        spec.context.compiled_residency,
                        seed,
                        spec.strategy,
                        extra_pressure=extra_pressure,
                    )
                else:
                    residency = reduce_pressure(
                        facts,
                        config,
                        seed,
                        spec.strategy,
                        extra_pressure=extra_pressure,
                        score_cache=spec.context.cut_scores,
                    )
                spec.context.residency_plans[residency_key] = residency
            if spec.prefetch_rule == "interval-entry":
                extended = spec.context.interval_plans.get(residency_key)
                if extended is None:
                    extended = extend_interval_entries(facts, residency)
                    spec.context.interval_plans[residency_key] = extended
                residency = extended
            prefetch_headroom = spec.strategy.startswith("headroom")
            schedule_key = (
                residency,
                spec.prefetch_rule,
                spec.coalesced,
                prefetch_headroom,
            )
            schedule = spec.context.schedule_cache.get(schedule_key)
            if schedule is None:
                schedule = emit_schedule(
                    facts,
                    config,
                    residency,
                    spec.prefetch_rule,
                    coalesced=spec.coalesced,
                    prefetch_headroom=prefetch_headroom,
                )
                spec.context.schedule_cache[schedule_key] = schedule
        except PressureFitInfeasibleError as error:
            return _CandidateOutcome(
                spec,
                _failure_diagnostic(
                    spec,
                    status="infeasible",
                    kind=error.kind,
                    detail=str(error),
                    repairs=repairs,
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
                    repairs=repairs,
                ),
            )
        restart_reduction = False
        while True:
            try:
                cached_simulation = spec.context.simulation_cache.get(schedule)
                if isinstance(cached_simulation, _CachedSimulationFailure):
                    raise cached_simulation.to_error()
                if cached_simulation is None:
                    try:
                        cached_simulation = (
                            simulate_compiled_template_summary(
                                spec.context.compiled_template,
                                schedule,
                            )
                            if spec.context.compiled_template is not None
                            else simulate(
                                facts.program,
                                schedule,
                                selections=facts.selections,
                                config=config,
                            )
                        )
                    except SimulationInfeasibleError as error:
                        spec.context.simulation_cache[schedule] = (
                            _CachedSimulationFailure.from_error(error)
                        )
                        raise
                    spec.context.simulation_cache[schedule] = cached_simulation
                simulation = cached_simulation
            except SimulationInfeasibleError as error:
                if repairs < options.max_repair_attempts:
                    delayed = _delay_prefetch(facts, schedule, error)
                    if delayed is not None and delayed != schedule:
                        schedule = delayed
                        repairs += 1
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
                        repairs += 1
                        restart_reduction = True
                        break
                return _CandidateOutcome(
                    spec,
                    _failure_diagnostic(
                        spec,
                        status="infeasible",
                        kind=error.kind,
                        detail=str(error),
                        repairs=repairs,
                    ),
                )
            break
        if restart_reduction:
            continue
        return _CandidateOutcome(
            spec,
            CandidateDiagnostic(
                candidate_id=spec.candidate_id,
                selection_id=spec.context.selection_id,
                status="valid",
                makespan_ns=simulation.makespan_ns,
                schedule_digest=schedule.digest,
                repair_attempts=repairs,
            ),
            schedule,
            simulation,
        )


def _build_contexts(
    program: Program,
    initial_residency: tuple[ResidencySpec, ...],
    final_residency: tuple[ResidencySpec, ...],
    config: SimulationConfig,
    options: PressureFitOptions,
) -> tuple[_SelectionContext, ...]:
    contexts: list[_SelectionContext] = []
    failures: list[PressureFitInfeasibleError] = []
    compiled_simulator_available = simulator_library_path() is not None
    compiled_planner_available = planner_library_path() is not None
    for selections in _selection_portfolio(program):
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
        contexts.append(
            _SelectionContext(
                selections,
                _selection_id(selections),
                facts,
                seed,
                (
                    compile_simulation_template(
                        program,
                        selections,
                        config,
                        selected_tasks=facts.tasks,
                    )
                    if compiled_simulator_available
                    else None
                ),
                (
                    compile_residency_template(facts, config, seed)
                    if compiled_planner_available
                    else None
                ),
            )
        )
    if contexts:
        return tuple(contexts)
    if failures:
        raise failures[0]
    raise PressureFitInfeasibleError(
        "no recomputation selection could be constructed",
        kind="recomputation_selection",
    )


def _candidate_specs(
    contexts: tuple[_SelectionContext, ...],
    options: PressureFitOptions,
) -> tuple[_CandidateSpec, ...]:
    specs: list[_CandidateSpec] = []
    ordinal = 0
    coalescing = (False, True) if options.evaluate_coalesced else (False,)
    for context in contexts:
        for strategy in options.residency_strategies:
            for rule in options.prefetch_rules:
                for coalesced in coalescing:
                    specs.append(
                        _CandidateSpec(
                            ordinal,
                            context,
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
        if not batches or batches[-1][0].context is not spec.context:
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


def pressurefit(
    program: Program,
    *,
    initial_residency: tuple[ResidencySpec, ...],
    final_residency: tuple[ResidencySpec, ...] = (),
    config: SimulationConfig,
    options: PressureFitOptions | None = None,
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
    selected_options = options or PressureFitOptions()
    contexts = _build_contexts(
        program,
        initial_residency,
        final_residency,
        config,
        selected_options,
    )
    specs = _candidate_specs(contexts, selected_options)
    outcomes = _run_candidates(specs, config, selected_options)
    valid = tuple(
        outcome
        for outcome in outcomes
        if outcome.schedule is not None and outcome.simulation is not None
    )
    if not valid:
        failure_diagnostics = tuple(outcome.diagnostic for outcome in outcomes)
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
    final_simulation = (
        simulate_compiled_template(
            best.spec.context.compiled_template,
            best.schedule,
        )
        if isinstance(best.simulation, CompiledSimulationSummary)
        and best.spec.context.compiled_template is not None
        else best.simulation
    )
    assert isinstance(final_simulation, SimulationResult)
    diagnostics = PressureFitDiagnostics(
        selected_candidate_id=best.spec.candidate_id,
        selected_selection_id=best.spec.context.selection_id,
        candidate_count=len(outcomes),
        valid_candidate_count=len(valid),
        selected_makespan_ns=final_simulation.makespan_ns,
        candidates=tuple(outcome.diagnostic for outcome in outcomes),
    )
    return PressureFitResult(
        program=program,
        options=selected_options,
        initial_residency=initial_residency,
        final_residency=final_residency,
        simulation_config=config,
        schedule=best.schedule,
        selections=best.spec.context.selections,
        simulation=final_simulation,
        diagnostics=diagnostics,
    )


__all__ = ["pressurefit"]
