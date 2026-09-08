"""Evaluating every resolution and choosing one winner.

PressureFit is given a family of legal resolutions and has to
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

from shadowspill.ir import (
    MemoryActionKind,
    Program,
    ResidencySpec,
    TaskAlternativeChoice,
)
from shadowspill.planner.result import ResidentSlice
from shadowspill.simulator import SimulationConfig
from shadowspill.simulator.indexed import (
    IndexedSimulationTemplate,
    index_simulation_template,
    simulate_template,
)

from ..admission import AdmissionFacts
from ..admission.indexed import (
    EncodedIndexedSchedule,
    IndexedAdmissionFacts,
    encode_schedule,
    evaluate_schedule_admission,
    index_admission_facts,
)
from ..best import BestPlaced
from ..diagnostics import (
    INCUMBENT_CANDIDATE_ID,
    IncumbentDiagnostic,
    PressureFitDiagnostics,
    PressureFitSectionTiming,
    PressureFitWorkDiagnostics,
    ResolvedProgramDiagnostics,
    TaskAlternativeChoiceDiagnostic,
)
from ..recomputation import Resolution
from ..request import PressureFitOptions
from ..result import (
    PressureFitInfeasibleError,
    PressureFitResult,
    PressureFitSearchExhaustedError,
)
from .candidates import (
    _ACTION_KIND,
    CPreflightResult,
    CProblemResult,
    decode_candidate_diagnostic,
    decode_schedule,
    evaluate_program_problems,
    validate_program_problem,
)


@dataclass(frozen=True, slots=True)
class SelectionProblem:
    """One resolution projected into the planner ABI."""

    selections: tuple[TaskAlternativeChoice, ...]
    selection_id: str
    indexed_template: IndexedSimulationTemplate
    indexed_admission: IndexedAdmissionFacts | None
    #: The same topology, for measuring layouts during the search. Kept
    #: apart from `indexed_admission` because supplying that switches on
    #: the dynamic-pool replay, which rejects plans certified fixed
    #: placement accepts.
    indexed_placement: IndexedAdmissionFacts | None = None
    #: The plan to beat, encoded against this problem, when the caller has
    #: one for this resolution.
    incumbent: EncodedIndexedSchedule | None = None


def _selection_id(selections: tuple[TaskAlternativeChoice, ...]) -> str:
    if not selections:
        return "none"
    return ",".join(f"{item.group_id}={item.option_id}" for item in selections)


def _selected_traffic(program: Program, result: CProblemResult) -> tuple[int, int]:
    """What one problem's own best plan moves, fetched and evicted.

    The winner's traffic is on the plan summary. This is every other
    selection's, which is what says whether a selection that asks for less
    compute pays for it on the lanes instead. Read off the problem's own
    schedule, which the search already decoded, so it costs a pass over the
    actions rather than another simulation.
    """

    schedule = result.selected_schedule
    if schedule is None:
        return 0, 0
    sizes = [group.size_bytes for group in program.alias_groups]
    fetched = 0
    evicted = 0
    for alias, kind in zip(schedule.action_aliases, schedule.action_kinds, strict=True):
        size = sizes[alias]
        if _ACTION_KIND[kind] is MemoryActionKind.FETCH:
            fetched += size
        elif _ACTION_KIND[kind] is MemoryActionKind.EVICT:
            evicted += size
    return fetched, evicted


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
    incumbent: PressureFitResult | None = None,
) -> tuple[SelectionProblem, ...]:
    """Project each resolution without Python residency matrices.

    `incumbent` is a plan already in hand for this Program; it is encoded
    against the resolution it was found for, and a search over resolutions
    that do not include that one carries no plan to beat.
    """

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
        carried = (
            encode_schedule(incumbent.schedule, indexed_template)
            if incumbent is not None
            and _same_resolution(incumbent.selections, selections)
            else None
        )
        problems.append(
            SelectionProblem(
                selections=selections,
                selection_id=_selection_id(selections),
                indexed_template=indexed_template,
                incumbent=carried,
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


def _same_resolution(
    left: tuple[TaskAlternativeChoice, ...], right: tuple[TaskAlternativeChoice, ...]
) -> bool:
    """Whether two selections fix every alternative the same way."""

    return {item.group_id: item.option_id for item in left} == {
        item.group_id: item.option_id for item in right
    }


_INCUMBENT_STATUS = {0: "valid", 2: "infeasible", 3: "infeasible", 7: "unplaceable"}


def _incumbent_diagnostic(
    result: CProblemResult, incumbent: PressureFitResult | None
) -> IncumbentDiagnostic | None:
    """What became of the plan to beat under one problem, if it carried one."""

    outcome = result.incumbent
    if outcome is None:
        return None
    found_by, found_at = (None, None) if incumbent is None else _origin(incumbent)
    return IncumbentDiagnostic(
        status=_INCUMBENT_STATUS.get(outcome.status, "error"),
        makespan_ns=outcome.makespan_ns or None,
        required_bytes=outcome.required_bytes or None,
        selected=outcome.selected,
        schedule_digest=None if incumbent is None else incumbent.schedule.digest,
        found_by=found_by,
        found_at_capacity_bytes=found_at,
    )


def _origin(incumbent: PressureFitResult) -> tuple[str | None, int | None]:
    """The candidate that first found the plan in hand, and at what capacity.

    A plan handed on more than once was the plan to beat of the search that
    answered with it, so its origin is read through that search's record
    rather than stopping at the hand-off.
    """

    diagnostics = incumbent.diagnostics
    if diagnostics.selected_candidate_id == INCUMBENT_CANDIDATE_ID:
        for problem in diagnostics.resolved_programs:
            if (
                problem.selection_id == diagnostics.selected_selection_id
                and problem.incumbent is not None
            ):
                return (
                    problem.incumbent.found_by,
                    problem.incumbent.found_at_capacity_bytes,
                )
    devices = incumbent.simulation_config.devices
    return (
        diagnostics.selected_candidate_id,
        devices[0].capacity_bytes if len(devices) == 1 else None,
    )


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
    """Keep resolutions that satisfy the semantic-capacity preflight."""

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
        "no resolution could be constructed",
        kind="graph_pair_selection",
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
                problem.incumbent,
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
    *,
    placement: AdmissionFacts | None = None,
    incumbent: PressureFitResult | None = None,
) -> PressureFitResult:
    """Decode the plan the search placed, and its diagnostics.

    `best` is the authority on what won. Every candidate offers the plans it
    places to that record as it places them, so by the time this runs the
    record already holds the best plan anyone reached -- including one a
    previous call left there, which is the point of sharing it. Ranking the
    problems again here would be a second answer to a question already
    answered, and the two can disagree: a problem's own winner is the best
    plan *it* placed, which is not the best plan placed.

    `incumbent` is the plan to beat the problems were built with, so its
    outcome can say where it came from.
    """

    resolved_programs: list[ResolvedProgramDiagnostics] = []
    selected: tuple[tuple[int, int, int], int, int | None, CProblemResult] | None = (
        None
    )
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
        answered_by_incumbent = (
            result.incumbent is not None and result.incumbent.selected
        )
        selected_candidate = (
            None
            if result.selected_candidate_index is None
            else candidates[result.selected_candidate_index]
        )
        if answered_by_incumbent:
            selected_id: str | None = INCUMBENT_CANDIDATE_ID
        elif selected_candidate is not None:
            selected_id = selected_candidate.candidate_id
        else:
            selected_id = None
        fetched_bytes, evicted_bytes = _selected_traffic(program, result)
        resolved_programs.append(
            ResolvedProgramDiagnostics(
                selection_id=problem.selection_id,
                choices=tuple(
                    TaskAlternativeChoiceDiagnostic(item.group_id, item.option_id)
                    for item in problem.selections
                ),
                selected_candidate_id=selected_id,
                selected_makespan_ns=(
                    None if selected_id is None else result.selected_makespan_ns
                ),
                candidate_evaluations=candidates,
                work=result.work,
                started_ns=result.started_ns,
                finished_ns=result.finished_ns,
                evict_ineligible_aliases=result.evict_ineligible_aliases,
                evict_ineligible_bytes=result.evict_ineligible_bytes,
                fetched_bytes=fetched_bytes,
                evicted_bytes=evicted_bytes,
                incumbent=_incumbent_diagnostic(result, incumbent),
            )
        )
        if selected_id is None:
            continue
        assert result.selected_makespan_ns is not None
        # The record decides; a problem is a candidate for decoding only
        # if it is holding the plan the record names. Ties fall to the
        # earlier problem, so the answer does not depend on arrival order,
        # and within a problem to the plan to beat, which came before every
        # candidate.
        if held is not None and result.selected_makespan_ns != held.makespan_ns:
            continue
        candidate_index = (
            None if answered_by_incumbent else result.selected_candidate_index
        )
        key = (
            result.selected_makespan_ns,
            problem_index,
            -1 if candidate_index is None else candidate_index,
        )
        if selected is None or key < selected[0]:
            selected = (key, problem_index, candidate_index, result)

    if selected is None:
        frozen = tuple(
            candidate
            for problem in resolved_programs
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

    _key, problem_index, candidate_index, result = selected
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
    selected_candidate_id = (
        INCUMBENT_CANDIDATE_ID
        if candidate_index is None
        else result.candidates[candidate_index].candidate_id
    )
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
        selected_candidate_id=selected_candidate_id,
        selected_selection_id=problem.selection_id,
        selected_makespan_ns=simulation.makespan_ns,
        resolved_programs=tuple(resolved_programs),
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
        resident_slice=ResidentSlice(
            bytes=result.resident_slice_bytes,
            aliases=tuple(
                sorted(
                    problem.indexed_template.alias_ids[index]
                    for index in result.resident_aliases
                )
            ),
        ),
        admission_facts=admission,
        placement_facts=placement,
    )


__all__ = [
    "SelectionProblem",
    "build_problems",
    "finish_pressurefit",
    "preflight_problems",
    "run_problems",
]
