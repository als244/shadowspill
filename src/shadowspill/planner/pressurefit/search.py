"""Evaluating every recomputation selection and choosing one winner.

PressureFit is given a family of legal recomputation selections and has to
return the best schedule across all of them. This module owns that loop:
projecting each selection into what the library needs, dropping the ones that
cannot fit before paying to evaluate them, running the rest, and merging their
results into one answer.

It deliberately knows nothing about which selections are worth trying - that is
``recomputation`` - and nothing about what to do when admission refuses the
winner, which is ``refinement``.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from functools import cache

from shadowspill.ir import Program, RecomputationSelection, ResidencySpec
from shadowspill.simulator import SimulationConfig
from shadowspill.simulator.indexed import (
    IndexedSimulationTemplate,
    index_simulation_template,
    simulate_template,
)

from ..admission import AdmissionFacts
from ..diagnostics import (
    PressureFitRepairDiagnostics,
    PressureFitWorkDiagnostics,
    RecomputationChoiceDiagnostic,
    RecomputationProblemDiagnostics,
)
from ..library.admission import (
    IndexedAdmissionFacts,
    evaluate_schedule_admission,
    index_admission_facts,
)
from ..library.portfolio import (
    CCandidateDiagnostic,
    CPreflightResult,
    CProblemResult,
    decode_candidate_diagnostic,
    decode_schedule,
    evaluate_program_problem,
    validate_program_problem,
)
from ..model import (
    PressureFitDiagnostics,
    PressureFitInfeasibleError,
    PressureFitOptions,
    PressureFitResult,
    PressureFitSearchExhaustedError,
)


@dataclass(frozen=True, slots=True)
class SelectionProblem:
    """One recomputation selection projected into the planner ABI."""

    selections: tuple[RecomputationSelection, ...]
    selection_id: str
    indexed_template: IndexedSimulationTemplate
    indexed_admission: IndexedAdmissionFacts | None


def _selection_id(selections: tuple[RecomputationSelection, ...]) -> str:
    if not selections:
        return "none"
    return ",".join(f"{item.group_id}={item.option_id}" for item in selections)


def build_problems(
    program: Program,
    initial_residency: tuple[ResidencySpec, ...],
    final_residency: tuple[ResidencySpec, ...],
    config: SimulationConfig,
    admission: AdmissionFacts | None,
    *,
    portfolio: tuple[tuple[RecomputationSelection, ...], ...],
    progress: Callable[[str], None] | None,
) -> tuple[SelectionProblem, ...]:
    """Project each recomputation selection without Python residency matrices."""

    problems: list[SelectionProblem] = []
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
            SelectionProblem(
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
    problem: SelectionProblem,
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


def preflight_problems(
    problems: tuple[SelectionProblem, ...],
) -> tuple[SelectionProblem, ...]:
    """Keep selections that satisfy the semantic-capacity preflight."""

    valid: list[SelectionProblem] = []
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


def worker_count(options: PressureFitOptions, count: int) -> int:
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


def run_problems(
    problems: tuple[SelectionProblem, ...],
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

    workers = worker_count(options, len(units))
    if workers == 1:
        chunk_results = [evaluate(unit) for unit in units]
    elif options.workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            chunk_results = list(executor.map(evaluate, units))
    else:
        chunk_results = list(_shared_worker_pool().map(evaluate, units))
    per_problem = len(strategies)
    return tuple(
        merge_strategy_results(
            chunk_results[index * per_problem : (index + 1) * per_problem]
        )
        for index in range(len(problems))
    )


def merge_strategy_results(
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


def finish_pressurefit(
    program: Program,
    initial_residency: tuple[ResidencySpec, ...],
    final_residency: tuple[ResidencySpec, ...],
    config: SimulationConfig,
    options: PressureFitOptions,
    problems: tuple[SelectionProblem, ...],
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


__all__ = [
    "SelectionProblem",
    "build_problems",
    "finish_pressurefit",
    "merge_strategy_results",
    "preflight_problems",
    "run_problems",
    "worker_count",
]
