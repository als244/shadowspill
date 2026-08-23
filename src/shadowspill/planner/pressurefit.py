"""Compiled, simulator-verified PressureFit orchestration."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from functools import cache

from shadowspill.ir import (
    Program,
    RecomputationSelection,
    ResidencySpec,
    shared_residency_footprint,
)
from shadowspill.simulator import SimulationConfig
from shadowspill.simulator._capi import simulator_api
from shadowspill.simulator._indexed import (
    IndexedSimulationTemplate,
    index_simulation_template,
    simulate_template,
)

from ._admission import (
    IndexedAdmissionFacts,
    evaluate_schedule_admission,
    index_admission_facts,
)
from ._capi import planner_api
from ._portfolio import (
    CCandidateDiagnostic,
    CPreflightResult,
    CProblemResult,
    decode_candidate_diagnostic,
    decode_schedule,
    evaluate_program_problem,
    validate_program_problem,
)
from ._recomputation import build_recomputation_portfolio
from .admission import AdmissionFacts
from .diagnostics import (
    PressureFitRepairDiagnostics,
    PressureFitWorkDiagnostics,
    RecomputationChoiceDiagnostic,
    RecomputationProblemDiagnostics,
)
from .model import (
    AdmissionRefinement,
    PressureFitDiagnostics,
    PressureFitInfeasibleError,
    PressureFitOptions,
    PressureFitResult,
    PressureFitSearchExhaustedError,
)

_ADMISSION_RESERVE_GRANULARITY_BYTES = 2 << 20
_ADMISSION_INITIAL_REFINEMENT_BYTES = 128 << 20
_ADMISSION_DOUBLING_LIMIT_BYTES = 1 << 30
_ADMISSION_LINEAR_REFINEMENT_BYTES = 512 << 20


def _scheduled_admission_refinement(attempt: int) -> int:
    """Return the deterministic reserve increment for one failed admission."""

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
    """One recomputation selection projected into the planner ABI."""

    selections: tuple[RecomputationSelection, ...]
    selection_id: str
    indexed_template: IndexedSimulationTemplate
    indexed_admission: IndexedAdmissionFacts | None


def _selection_id(selections: tuple[RecomputationSelection, ...]) -> str:
    if not selections:
        return "none"
    return ",".join(f"{item.group_id}={item.option_id}" for item in selections)


def _validate_pressurefit_inputs(
    program: Program,
    initial_residency: tuple[ResidencySpec, ...],
    final_residency: tuple[ResidencySpec, ...],
    config: SimulationConfig,
    admission: AdmissionFacts | None,
) -> None:
    """Validate public types and the logical/physical capacity relationship."""

    if not isinstance(program, Program):
        raise TypeError("program must be a Program")
    if not isinstance(initial_residency, tuple):
        raise TypeError("initial_residency must be a tuple")
    if not isinstance(final_residency, tuple):
        raise TypeError("final_residency must be a tuple")
    if not isinstance(config, SimulationConfig):
        raise TypeError("config must be a SimulationConfig")
    if admission is None:
        return
    if not isinstance(admission, AdmissionFacts):
        raise TypeError("admission must be an AdmissionFacts")
    admission.validate(program)
    configured = {item.device_id: item for item in config.devices}
    shared = shared_residency_footprint(program)
    movable_capacity = (
        configured[admission.device_id].capacity_bytes
        - shared.for_device(admission.device_id)
        if admission.device_id in configured
        else None
    )
    if (
        admission.device_id not in configured
        or movable_capacity != admission.object_capacity_bytes
    ):
        raise ValueError(
            "feasibility capacity after shared residency must equal "
            "AdmissionFacts object capacity"
        )


def validate_schedule_feasibility(
    program: Program,
    *,
    initial_residency: tuple[ResidencySpec, ...],
    final_residency: tuple[ResidencySpec, ...] = (),
    config: SimulationConfig,
    admission: AdmissionFacts | None = None,
) -> None:
    """Reject irreducible capacity failures using the planner."""

    _validate_pressurefit_inputs(
        program,
        initial_residency,
        final_residency,
        config,
        admission,
    )
    simulator_api()
    planner_api()
    problems = _build_problems(
        program,
        initial_residency,
        final_residency,
        config,
        admission,
        portfolio=build_recomputation_portfolio(program),
        progress=None,
    )
    _preflight_problems(problems)


def _build_problems(
    program: Program,
    initial_residency: tuple[ResidencySpec, ...],
    final_residency: tuple[ResidencySpec, ...],
    config: SimulationConfig,
    admission: AdmissionFacts | None,
    *,
    portfolio: tuple[tuple[RecomputationSelection, ...], ...],
    progress: Callable[[str], None] | None,
) -> tuple[_SelectionProblem, ...]:
    """Project each recomputation selection without Python residency matrices."""

    problems: list[_SelectionProblem] = []
    started = time.perf_counter_ns()
    for selection_index, selections in enumerate(portfolio, start=1):
        tasks = program.selected_tasks(selections)
        indexed_template = index_simulation_template(
            program,
            selections,
            config,
            selected_tasks=tasks,
            initial_residency=initial_residency,
            final_residency=final_residency,
        )
        problems.append(
            _SelectionProblem(
                selections=selections,
                selection_id=_selection_id(selections),
                indexed_template=indexed_template,
                indexed_admission=(
                    index_admission_facts(admission, indexed_template)
                    if admission is not None
                    else None
                ),
            )
        )
        if progress is not None:
            progress(
                "PressureFit compiled problem "
                f"{selection_index}/{len(portfolio)}: "
                f"tasks={len(tasks)}, aliases={len(program.alias_groups)}, "
                f"elapsed={(time.perf_counter_ns() - started) / 1e9:.3f}s"
            )
    return tuple(problems)


def _preflight_error(
    problem: _SelectionProblem,
    result: CPreflightResult,
) -> ValueError:
    """Decode one compiled preflight failure into the public exception model."""

    if result.failure_kind == "missing_initial_residency":
        if result.error_alias is None:
            raise RuntimeError("compiled preflight omitted its failing alias")
        alias_id = problem.indexed_template.alias_ids[result.error_alias]
        return ValueError(f"input alias {alias_id!r} has no initial residency")

    device_id = (
        None
        if result.error_device is None
        else problem.indexed_template.device_ids[result.error_device]
    )
    boundary_task_id = (
        None
        if result.error_boundary is None
        or result.error_boundary < 0
        or result.error_boundary >= len(problem.indexed_template.task_ids)
        else problem.indexed_template.task_ids[result.error_boundary]
    )
    required = result.required_bytes
    capacity = result.capacity_bytes
    if result.failure_kind == "workspace_capacity":
        return PressureFitInfeasibleError(
            f"task workspace {required} exceeds capacity {capacity} on {device_id!r}",
            kind="workspace_capacity",
            device_id=device_id,
            boundary_task_id=boundary_task_id,
            required_bytes=required,
            capacity_bytes=capacity,
        )
    if result.failure_kind == "required_capacity":
        return PressureFitInfeasibleError(
            f"required inputs and outputs need {required} bytes at "
            f"{boundary_task_id or 'initialization'} on {device_id!r}, "
            f"exceeding object capacity {capacity}",
            kind="required_capacity",
            device_id=device_id,
            boundary_task_id=boundary_task_id,
            required_bytes=required,
            capacity_bytes=capacity,
        )
    raise RuntimeError(
        f"compiled preflight returned unknown failure {result.failure_kind!r}"
    )


def _preflight_problems(
    problems: tuple[_SelectionProblem, ...],
) -> tuple[_SelectionProblem, ...]:
    """Keep selections that satisfy the semantic-capacity preflight."""

    valid: list[_SelectionProblem] = []
    failures: list[ValueError] = []
    for problem in problems:
        result = validate_program_problem(
            problem.indexed_template,
            admission=problem.indexed_admission,
        )
        if result.valid:
            valid.append(problem)
        else:
            failures.append(_preflight_error(problem, result))
    if valid:
        return tuple(valid)
    if failures:
        raise failures[0]
    raise PressureFitInfeasibleError(
        "no recomputation selection could be constructed",
        kind="recomputation_selection",
    )


def _worker_count(options: PressureFitOptions, count: int) -> int:
    if count <= 1 or options.workers == 1:
        return 1
    if options.workers > 1:
        return min(options.workers, count)
    return min(max(os.cpu_count() or 1, 1), count)


@cache
def _shared_worker_pool() -> ThreadPoolExecutor:
    """One process-wide pool for all compiled evaluation units.

    Concurrent plans — the speculative capacity-ladder rungs above
    all — submit their units here instead of nesting private pools,
    so live compiled threads never exceed the machine's cores and
    idle cores drain whichever rung still has work. Worker count is
    scheduling only; results are merged in deterministic unit order.
    """

    return ThreadPoolExecutor(max_workers=max(os.cpu_count() or 1, 1))


def _run_problems(
    problems: tuple[_SelectionProblem, ...],
    options: PressureFitOptions,
) -> tuple[CProblemResult | None, ...]:
    """Evaluate every recomputation selection in the planner.

    Each problem's portfolio is split by residency strategy into one
    compiled evaluation per (problem, strategy), so parallelism scales
    with problem_count x strategy_count instead of problem count
    alone. The per-problem merge restores the exact serial candidate
    order and tie-breaks, so the selected schedule is identical to a
    single-call evaluation; only cross-strategy cache-hit counters can
    differ.
    """

    strategies = options.residency_strategies
    units = tuple(
        (problem_index, replace(options, residency_strategies=(strategy,)))
        for problem_index, _problem in enumerate(problems)
        for strategy in strategies
    )

    def evaluate(
        unit: tuple[int, PressureFitOptions],
    ) -> CProblemResult | None:
        problem_index, unit_options = unit
        problem = problems[problem_index]
        return evaluate_program_problem(
            problem.indexed_template,
            unit_options,
            admission=problem.indexed_admission,
        )

    workers = _worker_count(options, len(units))
    if workers == 1:
        chunk_results = [evaluate(unit) for unit in units]
    elif options.workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            chunk_results = list(executor.map(evaluate, units))
    else:
        chunk_results = list(_shared_worker_pool().map(evaluate, units))
    per_problem = len(strategies)
    return tuple(
        _merge_strategy_results(
            chunk_results[index * per_problem : (index + 1) * per_problem]
        )
        for index in range(len(problems))
    )


def _merge_strategy_results(
    chunks: list[CProblemResult | None],
) -> CProblemResult | None:
    """Concatenate per-strategy evaluations back into one problem result."""

    if any(chunk is None for chunk in chunks):
        return None
    merged = [chunk for chunk in chunks if chunk is not None]
    if len(merged) == 1:
        return merged[0]
    candidates: list[CCandidateDiagnostic] = []
    selected_index: int | None = None
    selected_makespan: int | None = None
    selected_schedule = None
    repairs = PressureFitRepairDiagnostics()
    work = PressureFitWorkDiagnostics()
    for chunk in merged:
        offset = len(candidates)
        candidates.extend(chunk.candidates)
        repairs += chunk.repairs
        work += chunk.work
        if chunk.selected_candidate_index is None:
            continue
        assert chunk.selected_makespan_ns is not None
        if selected_makespan is None or chunk.selected_makespan_ns < selected_makespan:
            selected_index = offset + chunk.selected_candidate_index
            selected_makespan = chunk.selected_makespan_ns
            selected_schedule = chunk.selected_schedule
    return CProblemResult(
        selected_candidate_index=selected_index,
        selected_makespan_ns=selected_makespan,
        selected_schedule=selected_schedule,
        candidates=tuple(candidates),
        repairs=repairs,
        work=work,
    )


def _finish_pressurefit(
    program: Program,
    initial_residency: tuple[ResidencySpec, ...],
    final_residency: tuple[ResidencySpec, ...],
    config: SimulationConfig,
    options: PressureFitOptions,
    problems: tuple[_SelectionProblem, ...],
    results: tuple[CProblemResult, ...],
    admission: AdmissionFacts | None,
) -> PressureFitResult:
    """Decode the globally best compiled result and its diagnostics."""

    recomputation_problems: list[RecomputationProblemDiagnostics] = []
    selected: tuple[int, int, CProblemResult] | None = None
    for problem_index, (problem, result) in enumerate(
        zip(problems, results, strict=True)
    ):
        candidates = tuple(
            decode_candidate_diagnostic(
                candidate,
                selection_id=problem.selection_id,
                simulation=problem.indexed_template,
            )
            for candidate in result.candidates
        )
        selected_candidate = (
            None
            if result.selected_candidate_index is None
            else candidates[result.selected_candidate_index]
        )
        recomputation_problems.append(
            RecomputationProblemDiagnostics(
                selection_id=problem.selection_id,
                choices=tuple(
                    RecomputationChoiceDiagnostic(item.group_id, item.option_id)
                    for item in problem.selections
                ),
                selected_candidate_id=(
                    None
                    if selected_candidate is None
                    else selected_candidate.candidate_id
                ),
                selected_makespan_ns=(
                    None
                    if selected_candidate is None
                    else selected_candidate.makespan_ns
                ),
                candidate_evaluations=candidates,
                work=result.work,
            )
        )
        if result.selected_candidate_index is None:
            continue
        assert result.selected_makespan_ns is not None
        candidate_ordinal = (
            problem_index * len(result.candidates) + result.selected_candidate_index
        )
        key = (result.selected_makespan_ns, candidate_ordinal)
        if selected is None or key < (selected[0], selected[1]):
            selected = (key[0], key[1], result)

    if selected is None:
        frozen = tuple(
            candidate
            for problem in recomputation_problems
            for candidate in problem.candidate_evaluations
        )
        if any(item.status == "exhausted" for item in frozen):
            raise PressureFitSearchExhaustedError(
                "PressureFit exhausted its bounded candidate-repair budget "
                "before proving a feasible schedule",
                diagnostics=frozen,
            )
        first = frozen[0] if frozen else None
        physical_slack = tuple(
            candidate.error_required_bytes
            for result in results
            for candidate in result.candidates
            if candidate.status == 3 and candidate.error_required_bytes > 0
        )
        raise PressureFitInfeasibleError(
            "no simulator-valid PressureFit candidate satisfied the declared "
            "capacity and residency constraints",
            kind=(
                first.failure_kind
                if first is not None and first.failure_kind
                else "no_candidate"
            ),
            required_bytes=min(physical_slack) if physical_slack else None,
            capacity_bytes=(
                config.devices[0].capacity_bytes if len(config.devices) == 1 else None
            ),
            diagnostics=frozen,
        )

    _makespan, ordinal, result = selected
    per_problem = len(result.candidates)
    problem_index, candidate_index = divmod(ordinal, per_problem)
    problem = problems[problem_index]
    indexed_schedule = result.selected_schedule
    assert indexed_schedule is not None
    schedule = decode_schedule(indexed_schedule, problem.indexed_template)
    result_admission_calls = 0
    result_admission_time_ns = 0
    simulation_admission = None
    if problem.indexed_admission is not None:
        admission_started = time.perf_counter_ns()
        simulation_admission = evaluate_schedule_admission(
            problem.indexed_template,
            problem.indexed_admission,
            indexed_schedule,
        ).simulation_admission
        result_admission_time_ns = time.perf_counter_ns() - admission_started
        result_admission_calls = 1
    simulation_started = time.perf_counter_ns()
    simulation = simulate_template(
        problem.indexed_template,
        schedule,
        admission=simulation_admission,
    )
    result_simulation_time_ns = time.perf_counter_ns() - simulation_started
    selected_diagnostic = result.candidates[candidate_index]
    aggregate_work = PressureFitWorkDiagnostics()
    for problem_result in results:
        aggregate_work += problem_result.work
    aggregate_work += PressureFitWorkDiagnostics(
        result_simulation_calls=1,
        result_admission_calls=result_admission_calls,
        result_simulation_time_ns=result_simulation_time_ns,
        result_admission_time_ns=result_admission_time_ns,
    )
    diagnostics = PressureFitDiagnostics(
        selected_candidate_id=selected_diagnostic.candidate_id,
        selected_selection_id=problem.selection_id,
        selected_makespan_ns=simulation.makespan_ns,
        recomputation_problems=tuple(recomputation_problems),
        work=aggregate_work,
    )
    return PressureFitResult(
        program=program,
        options=options,
        initial_residency=initial_residency,
        final_residency=final_residency,
        simulation_config=config,
        schedule=schedule,
        selections=problem.selections,
        simulation=simulation,
        diagnostics=diagnostics,
        admission_facts=admission,
    )


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
    """Evaluate every selection through the planner."""

    _validate_pressurefit_inputs(
        program,
        initial_residency,
        final_residency,
        config,
        admission,
    )
    simulator_api()
    planner_api()

    selected_options = options or PressureFitOptions()
    portfolio = build_recomputation_portfolio(program)
    if progress is not None:
        progress(
            "PressureFit portfolio: "
            f"groups={len(program.recomputation_groups)}, "
            f"selections={len(portfolio)}"
        )
    started = time.perf_counter_ns()
    problems = _preflight_problems(
        _build_problems(
            program,
            initial_residency,
            final_residency,
            config,
            admission,
            portfolio=portfolio,
            progress=progress,
        )
    )
    results = _run_problems(problems, selected_options)
    valid_pairs = tuple(
        (problem, result)
        for problem, result in zip(problems, results, strict=True)
        if result is not None
    )
    if progress is not None:
        progress(
            "PressureFit compiled problems and candidates finished: "
            f"valid={len(valid_pairs)}/{len(problems)}, "
            "candidates="
            f"{sum(len(result.candidates) for _problem, result in valid_pairs)}, "
            f"workers={_worker_count(selected_options, len(problems))}, "
            f"elapsed={(time.perf_counter_ns() - started) / 1e9:.3f}s"
        )
    if not valid_pairs:
        raise RuntimeError(
            "PressureFit rejected every selection after semantic "
            "feasibility validation succeeded"
        )
    return _finish_pressurefit(
        program,
        initial_residency,
        final_residency,
        config,
        selected_options,
        tuple(problem for problem, _result in valid_pairs),
        tuple(result for _problem, result in valid_pairs),
        admission,
    )


def _round_up_admission_reserve(value: int) -> int:
    granularity = _ADMISSION_RESERVE_GRANULARITY_BYTES
    return ((value + granularity - 1) // granularity) * granularity


def _with_object_capacity(
    config: SimulationConfig,
    admission: AdmissionFacts,
    capacity_bytes: int,
    *,
    shared_execution_bytes: int,
) -> tuple[SimulationConfig, AdmissionFacts]:
    devices = tuple(
        replace(
            device,
            capacity_bytes=capacity_bytes + shared_execution_bytes,
        )
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
    """Select a schedule and refine dynamic-slab headroom."""

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
                    f"reserve_increment={increment}, object_capacity={capacity}"
                )
            current_config, current_admission = _with_object_capacity(
                current_config,
                current_admission,
                capacity,
                shared_execution_bytes=shared_residency_footprint(program).for_device(
                    current_admission.device_id
                ),
            )


__all__ = ["pressurefit", "validate_schedule_feasibility"]
