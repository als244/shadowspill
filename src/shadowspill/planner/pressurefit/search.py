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

import time
from collections.abc import Callable
from dataclasses import dataclass

from shadowspill.ir import Program, RecomputationSelection, ResidencySpec
from shadowspill.simulator import SimulationConfig
from shadowspill.simulator.indexed import (
    IndexedSimulationTemplate,
    index_simulation_template,
    simulate_template,
)

from ..admission import AdmissionFacts
from ..admission.indexed import (
    IndexedAdmissionFacts,
    evaluate_schedule_admission,
    index_admission_facts,
)
from ..best import BestPlaced
from ..diagnostics import (
    PressureFitDiagnostics,
    PressureFitSectionTiming,
    PressureFitWorkDiagnostics,
    RecomputationChoiceDiagnostic,
    RecomputationProblemDiagnostics,
)
from ..recomputation import Resolution
from ..request import PressureFitOptions
from ..result import (
    PressureFitInfeasibleError,
    PressureFitResult,
    PressureFitSearchExhaustedError,
)
from .candidates import (
    CPreflightResult,
    CProblemResult,
    decode_candidate_diagnostic,
    decode_schedule,
    evaluate_program_problems,
    validate_program_problem,
)


@dataclass(frozen=True, slots=True)
class SelectionProblem:
    """One recomputation selection projected into the planner ABI."""

    selections: tuple[RecomputationSelection, ...]
    selection_id: str
    indexed_template: IndexedSimulationTemplate
    indexed_admission: IndexedAdmissionFacts | None
    #: The same topology, for measuring layouts during the search. Kept
    #: apart from `indexed_admission` because supplying that switches on
    #: the dynamic-pool replay, which rejects plans certified fixed
    #: placement accepts.
    indexed_placement: IndexedAdmissionFacts | None = None


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
    placement: AdmissionFacts | None = None,
    resolutions: tuple[Resolution, ...],
    progress: Callable[[str], None] | None,
) -> tuple[SelectionProblem, ...]:
    """Project each recomputation selection without Python residency matrices."""

    problems: list[SelectionProblem] = []
    started = time.perf_counter_ns()
    for selection_index, selections in enumerate(resolutions, start=1):
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
                indexed_placement=(
                    index_admission_facts(placement, indexed_template)
                    if placement is not None
                    else None
                ),
            )
        )
        if progress is not None:
            progress(
                "PressureFit compiled problem "
                f"{selection_index}/{len(resolutions)}: "
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


def run_problems(
    problems: tuple[SelectionProblem, ...],
    options: PressureFitOptions,
    *,
    best: BestPlaced | None = None,
) -> tuple[CProblemResult | None, ...]:
    """Evaluate every resolved program in the planner, on its worker threads.

    One call, however many resolved programs there are. The library owns the
    threads and hands out candidates, so worker count and problem count are
    independent and the placement record is shared across all of them -- a
    plan placed under any resolved program bounds the search under the rest.
    """

    return evaluate_program_problems(
        tuple(
            (
                problem.indexed_template,
                problem.indexed_admission,
                problem.indexed_placement,
            )
            for problem in problems
        ),
        options,
        best_placed=0 if best is None else best.handle,
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
    best: BestPlaced | None = None,
) -> PressureFitResult:
    """Decode the plan the search placed, and its diagnostics.

    `best` is the authority on what won. Every candidate offers the plans it
    places to that record as it places them, so by the time this runs the
    record already holds the best plan anyone reached -- including one a
    previous call left there, which is the point of sharing it. Ranking the
    problems again here would be a second answer to a question already
    answered, and the two can disagree: a problem's own winner is the best
    plan *it* placed, which is not the best plan placed.
    """

    recomputation_problems: list[RecomputationProblemDiagnostics] = []
    selected: tuple[int, int, CProblemResult] | None = None
    held = None if best is None else best.read()
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
                started_ns=result.started_ns,
                finished_ns=result.finished_ns,
            )
        )
        if result.selected_candidate_index is None:
            continue
        assert result.selected_makespan_ns is not None
        candidate_ordinal = (
            problem_index * len(result.candidates) + result.selected_candidate_index
        )
        # The record decides; a problem is a candidate for decoding only
        # if it is holding the plan the record names. Ties fall to the
        # earlier problem, so the answer does not depend on arrival order.
        if held is not None and result.selected_makespan_ns != held.makespan_ns:
            continue
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
    # Materialising the winner is the caller's own `select` section: the
    # search has already chosen, and what remains is producing the plan it
    # chose. Admission and simulation here are the same work the search did,
    # so they add to the same counters.
    selected_started = time.perf_counter_ns()
    admission_calls = 0
    admission_ns = 0
    simulation_admission = None
    if problem.indexed_admission is not None:
        admission_started = time.perf_counter_ns()
        simulation_admission = evaluate_schedule_admission(
            problem.indexed_template,
            problem.indexed_admission,
            indexed_schedule,
        ).simulation_admission
        admission_ns = time.perf_counter_ns() - admission_started
        admission_calls = 1
    # At full capacity, which is what the plan will actually run at. A plan
    # built at a smaller capacity was *chosen* on how it behaves there, but
    # the machine it runs on is the one the caller described, so that is what
    # the reported timeline and the certificate measure.
    simulation = simulate_template(
        problem.indexed_template,
        schedule,
        admission=simulation_admission,
    )
    selected_ns = time.perf_counter_ns() - selected_started
    selected_diagnostic = result.candidates[candidate_index]
    aggregate_work = PressureFitWorkDiagnostics()
    for problem_result in results:
        aggregate_work += problem_result.work
    aggregate_work += PressureFitWorkDiagnostics(
        simulation_calls=1,
        admission_calls=admission_calls,
        sections=PressureFitSectionTiming(
            total_ns=selected_ns,
            select_ns=selected_ns,
            admit_ns=admission_ns,
        ),
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
    "preflight_problems",
    "run_problems",
]
