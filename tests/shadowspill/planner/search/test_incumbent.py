"""A plan already in hand is the plan to beat: the search never answers worse."""

from __future__ import annotations

import pytest

from shadowspill.ir import MemoryLocation, ResidencySpec
from shadowspill.planner import (
    GenericPlanningOptions,
    SearchOptions,
)
from shadowspill.planner.diagnostics import (
    INCUMBENT_CANDIDATE_ID,
    PlanningDiagnostics,
    ResolvedProgramDiagnostics,
)
from shadowspill.planner.search.algorithms.pressurefit import PressureFit
from shadowspill.planner.search.algorithms.pressurefit.options import (
    PressureFitOptions,
)

from .._examples import (
    config,
    exact_capacity_program,
    exact_capacity_residency,
    recomputation_program,
    training_chain_config,
    training_chain_initial,
    training_chain_program,
)

# Deterministic, so a candidate answers the same alone as among the others:
# what the full search finds is then never worse than any one candidate's.
_DETERMINISTIC = GenericPlanningOptions(
    minimum_object_bytes_evict_eligible=0, deterministic=True
)
EVERY_CANDIDATE = SearchOptions(generic=_DETERMINISTIC)
ONE_POOR_CANDIDATE = SearchOptions(
    generic=_DETERMINISTIC,
    algorithm=PressureFit(
        PressureFitOptions(
            residency_strategies=("tight-stall",),
            fetch_rules=("demand",),
            evaluate_coalesced=False,
        )
    ),
)


def _selected_problem(
    diagnostics: PlanningDiagnostics,
) -> ResolvedProgramDiagnostics:
    return next(
        problem
        for problem in diagnostics.resolved_programs
        if problem.selection_id == diagnostics.selected_selection_id
    )


def test_a_search_handed_its_own_answer_answers_with_it() -> None:
    initial, final = exact_capacity_residency()
    first = EVERY_CANDIDATE.resolved_algorithm(
        exact_capacity_program(),
        initial_residency=initial,
        final_residency=final,
        config=config(),
        generic=EVERY_CANDIDATE.generic,
    )
    again = EVERY_CANDIDATE.resolved_algorithm(
        exact_capacity_program(),
        initial_residency=initial,
        final_residency=final,
        config=config(),
        generic=EVERY_CANDIDATE.generic,
        incumbent=first,
    )

    assert again.schedule == first.schedule
    assert again.simulation.makespan_ns == first.simulation.makespan_ns
    assert again.diagnostics.selected_candidate_id == INCUMBENT_CANDIDATE_ID
    problem = _selected_problem(again.diagnostics)
    assert problem.selected_candidate_id == INCUMBENT_CANDIDATE_ID
    assert problem.incumbent is not None
    assert problem.incumbent.status == "valid"
    assert problem.incumbent.selected
    assert problem.incumbent.makespan_ns == first.simulation.makespan_ns
    assert problem.incumbent.required_bytes is None  # no pool to place into
    assert problem.incumbent.schedule_digest == first.schedule.digest
    assert problem.incumbent.found_by == first.diagnostics.selected_candidate_id
    capacity = config().devices[0].capacity_bytes
    assert problem.incumbent.found_at_capacity_bytes == capacity
    # every candidate still ran and is on the record
    assert problem.candidate_policy_count == first.diagnostics.candidate_policy_count
    # and the record round-trips through its serialized form
    restored = PlanningDiagnostics.from_value(again.diagnostics.to_dict(), "d")
    assert _selected_problem(restored).incumbent == problem.incumbent
    assert restored.selected_candidate_id == INCUMBENT_CANDIDATE_ID


def test_a_worse_search_answers_with_the_plan_it_was_handed() -> None:
    initial, final = exact_capacity_residency()
    program = exact_capacity_program()
    every = EVERY_CANDIDATE.resolved_algorithm(
        program,
        initial_residency=initial,
        final_residency=final,
        config=config(),
        generic=EVERY_CANDIDATE.generic,
    )
    poor = ONE_POOR_CANDIDATE.resolved_algorithm(
        program,
        initial_residency=initial,
        final_residency=final,
        config=config(),
        generic=ONE_POOR_CANDIDATE.generic,
    )
    handed = ONE_POOR_CANDIDATE.resolved_algorithm(
        program,
        initial_residency=initial,
        final_residency=final,
        config=config(),
        generic=ONE_POOR_CANDIDATE.generic,
        incumbent=every,
    )

    assert every.simulation.makespan_ns <= poor.simulation.makespan_ns
    # never worse than the plan in hand, and a tie keeps it
    assert handed.simulation.makespan_ns == every.simulation.makespan_ns
    assert handed.schedule == every.schedule
    assert handed.diagnostics.selected_candidate_id == INCUMBENT_CANDIDATE_ID
    assert handed.diagnostics.candidate_evaluation_count == 1


def test_more_memory_never_plans_worse_when_handed_the_smaller_budgets_plan() -> None:
    program = training_chain_program(5)
    initial = training_chain_initial(5)
    small = EVERY_CANDIDATE.resolved_algorithm(
        program,
        initial_residency=initial,
        config=training_chain_config(800),
        generic=EVERY_CANDIDATE.generic,
    )
    alone = EVERY_CANDIDATE.resolved_algorithm(
        program,
        initial_residency=initial,
        config=training_chain_config(1200),
        generic=EVERY_CANDIDATE.generic,
    )
    handed = EVERY_CANDIDATE.resolved_algorithm(
        program,
        initial_residency=initial,
        config=training_chain_config(1200),
        generic=EVERY_CANDIDATE.generic,
        incumbent=small,
    )

    assert handed.simulation.makespan_ns <= small.simulation.makespan_ns
    assert handed.simulation.makespan_ns <= alone.simulation.makespan_ns
    problem = _selected_problem(handed.diagnostics)
    assert problem.incumbent is not None
    assert problem.incumbent.status == "valid"
    assert problem.incumbent.found_at_capacity_bytes == (
        training_chain_config(800).devices[0].capacity_bytes
    )
    if handed.diagnostics.selected_candidate_id == INCUMBENT_CANDIDATE_ID:
        assert problem.incumbent.selected
        assert handed.schedule == small.schedule
        # handed on again, the plan still names the candidate that first
        # found it and the capacity it was found at, not the hand-off
        twice = EVERY_CANDIDATE.resolved_algorithm(
            program,
            initial_residency=initial,
            config=training_chain_config(1600),
            generic=EVERY_CANDIDATE.generic,
            incumbent=handed,
        )
        origin = _selected_problem(twice.diagnostics).incumbent
        assert origin is not None
        assert origin.found_by == small.diagnostics.selected_candidate_id
        assert origin.found_at_capacity_bytes == (
            training_chain_config(800).devices[0].capacity_bytes
        )
    else:
        assert not problem.incumbent.selected
        assert handed.simulation.makespan_ns < small.simulation.makespan_ns


def test_a_plan_for_another_program_is_refused() -> None:
    initial, final = exact_capacity_residency()
    first = EVERY_CANDIDATE.resolved_algorithm(
        exact_capacity_program(),
        initial_residency=initial,
        final_residency=final,
        config=config(),
        generic=EVERY_CANDIDATE.generic,
    )

    with pytest.raises(ValueError, match="different program"):
        EVERY_CANDIDATE.resolved_algorithm(
            recomputation_program(),
            initial_residency=(ResidencySpec("input_storage", MemoryLocation.DEVICE),),
            config=config(110),
            generic=EVERY_CANDIDATE.generic,
            incumbent=first,
        )


def test_the_plan_reaches_the_resolution_it_was_found_for() -> None:
    program = recomputation_program()
    residency = (ResidencySpec("input_storage", MemoryLocation.DEVICE),)
    first = EVERY_CANDIDATE.resolved_algorithm(
        program,
        initial_residency=residency,
        config=config(110),
        generic=EVERY_CANDIDATE.generic,
    )
    assert first.diagnostics.selected_selection_id == "activation_tradeoff=recompute"

    # searched over both resolutions, it reaches the one it was found for
    both = PressureFit(PressureFitOptions(resolution_options=("0", "1")))(
        program,
        initial_residency=residency,
        config=config(110),
        generic=EVERY_CANDIDATE.generic,
        incumbent=first,
    )
    carried = {
        problem.selection_id: problem.incumbent
        for problem in both.diagnostics.resolved_programs
    }
    assert carried["activation_tradeoff=recompute"] is not None
    assert carried["activation_tradeoff=save"] is None
    assert both.diagnostics.selected_candidate_id == INCUMBENT_CANDIDATE_ID

    # the all-recompute resolution is always searched, so a plan found for
    # it is carried whatever the options name
    saved_only = PressureFit(PressureFitOptions(resolution_options=("0",)))(
        program,
        initial_residency=residency,
        config=config(1_000),
        generic=EVERY_CANDIDATE.generic,
        incumbent=first,
    )
    carried = {
        problem.selection_id: problem.incumbent
        for problem in saved_only.diagnostics.resolved_programs
    }
    assert carried["activation_tradeoff=recompute"] is not None
    assert carried["activation_tradeoff=save"] is None


def test_a_plan_for_a_resolution_that_is_not_searched_is_not_carried() -> None:
    from shadowspill.simulator import SimulationConfig

    from ..search.toolkit.test_resolution import _ladder_program

    stages = 8
    program = _ladder_program(stages)
    initial = tuple(
        ResidencySpec(f"input_{index}", MemoryLocation.DEVICE)
        for index in range(stages)
    )
    machine = SimulationConfig.single_device(
        "cuda_0",
        device_capacity_bytes=260,
        spill_capacity_bytes=10_000,
        fetch_bandwidth_bytes_per_second=8_000_000,
        evict_bandwidth_bytes_per_second=8_000_000,
    )
    eighths = PressureFit(
        PressureFitOptions(resolution_options=tuple(f"{n}/8" for n in range(9)))
    )(
        program,
        initial_residency=initial,
        config=machine,
        generic=EVERY_CANDIDATE.generic,
    )
    quarters = EVERY_CANDIDATE.resolved_algorithm(
        program,
        initial_residency=initial,
        config=machine,
        generic=EVERY_CANDIDATE.generic,
        incumbent=eighths,
    )

    found_for = eighths.diagnostics.selected_selection_id
    searched = {
        problem.selection_id: problem
        for problem in quarters.diagnostics.resolved_programs
    }
    for selection_id, problem in searched.items():
        assert (problem.incumbent is not None) == (selection_id == found_for)
    if found_for in searched:
        assert quarters.diagnostics.selected_candidate_id == INCUMBENT_CANDIDATE_ID
        assert quarters.simulation.makespan_ns == eighths.simulation.makespan_ns
    else:
        # nothing carried: the search proceeds on its own, and the odd
        # eighth it was handed stays out of a search over quarters
        assert quarters.diagnostics.selected_candidate_id != INCUMBENT_CANDIDATE_ID
        assert quarters.diagnostics.resolved_program_count == 5
