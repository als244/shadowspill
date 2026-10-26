"""A forbidden traversal is a planning rejection, not a broken profiling run."""

from dataclasses import replace

import pytest

from shadowspill import MemoryBudgets, plan_program
from shadowspill.errors import PlanInfeasibleError
from shadowspill.step import StepDataOrdering
from tests.shadowspill.offline._example import communication, profiled


@pytest.mark.parametrize(
    "reason",
    [
        "RNG effects require the original example order",
        "model mutations require the original example order",
    ],
)
def test_offline_planning_reports_the_saved_ordering_constraint(reason):
    source = profiled()
    source = replace(
        source,
        program=replace(
            source.program,
            bindings={
                "training_templates": [[], []],
                "ordering_constraints": [reason],
            },
        ),
    )
    with pytest.raises(PlanInfeasibleError) as caught:
        plan_program(
            source,
            memory_budgets=MemoryBudgets(200, 64),
            communication_info=communication(),
            ordering=StepDataOrdering(1, 2),
        )
    assert caught.value.kind == "ordering_constraints"
    assert caught.value.diagnostics == (reason,)
