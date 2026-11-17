"""Capacity refinement must survive unchanged residency and fetch schedules."""

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


@pytest.mark.parametrize("capacity", (272, 304, 336))
def test_small_capacity_steps_eventually_place_the_training_chain(capacity):
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
            capacity_refinement_bytes=1,
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
    assert candidate.capacity_refinements > 0
    admitted = build_fixed_layout_admission(selected, facts)
    assert admitted.layout.required_bytes <= capacity
    assert admitted.simulation.makespan_ns > 0
