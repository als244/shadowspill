"""Capacity refinement must survive unchanged residency and fetch schedules."""

import itertools

import pytest

from engine_tests.planning.memory_search._examples import (
    training_chain_config,
    training_chain_initial,
    training_chain_program,
)
from shadowspill.planning.memory_search import (
    AdmissionFacts,
    GenericPlanningOptions,
    TaskAdmissionSpec,
    TaskAllocationStep,
    TaskAllocationStepKind,
)
from shadowspill.planning.memory_search.admission.layout.build import (
    build_fixed_layout_admission,
)
from shadowspill.planning.memory_search.algorithms.pressurefit import (
    PressureFit,
    PressureFitOptions,
)


def _search(capacity, step):
    program = training_chain_program(3)
    sizes = {obj.object_id: obj.size_bytes for obj in program.objects}
    facts = AdmissionFacts(
        "cuda_0",
        capacity,
        capacity,
        1,
        tuple(
            TaskAdmissionSpec(
                task.task_id,
                fresh_output_aliases=task.outputs,
                allocation_steps=tuple(
                    TaskAllocationStep(
                        ordinal, TaskAllocationStepKind.ALLOCATE, sizes[name], name
                    )
                    for ordinal, name in enumerate(task.outputs)
                ),
            )
            for task in program.tasks
        ),
    )
    selected = PressureFit(
        PressureFitOptions(
            residency_strategies=("tight-stall",),
            fetch_rules=("packed-fit",),
            capacity_refinement_bytes=step,
            record_reduction_steps=True,
        )
    )(
        program,
        initial_residency=training_chain_initial(3),
        config=training_chain_config(capacity),
        generic=GenericPlanningOptions(minimum_object_bytes_evict_eligible=0),
        placement=facts,
        workers=1,
    )
    candidate = selected.diagnostics.resolved_programs[0].candidate_evaluations[0]
    return candidate, build_fixed_layout_admission(selected, facts)


@pytest.mark.parametrize("capacity", (272, 304, 336))
def test_small_capacity_steps_eventually_place_the_training_chain(capacity):
    candidate, admitted = _search(capacity, 1)
    assert candidate.capacity_refinements > 0
    assert admitted.layout.required_bytes <= capacity
    assert admitted.simulation.makespan_ns > 0


@pytest.mark.parametrize(("step", "give_backs"), ((1 << 28, [16, 32]), (8, [8, 8, 8])))
def test_a_round_that_changed_nothing_gives_back_twice_as_much(step, give_backs):
    # At 304 bytes the first round gives back its 16-byte overrun and the
    # extent stays where it was, so the next round gives back twice that,
    # never more than the bound.
    candidate, admitted = _search(304, step)
    capacities = [record.capacity_bytes for record in candidate.steps]
    assert [
        before - after
        for before, after in itertools.pairwise(capacities)
        if before != after
    ] == give_backs
    assert admitted.layout.required_bytes <= 304
