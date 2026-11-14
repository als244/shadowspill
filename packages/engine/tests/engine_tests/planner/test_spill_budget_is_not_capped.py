"""A problem answers about a spill pool larger than the one it was profiled with.

Planning and running are different questions. Nothing a problem measured
depends on how large the spill pool was -- it holds what was evicted, and the
task profiles, the transfer calibration and the layout are unmoved by its
size -- so a problem profiled against a small pool answers honestly about a
large one. That is what a sweep over spill budgets, or over a machine nobody
owns yet, is asking.

What a run needs is checked where it can be: against the pool the runtime
actually has, when a plan is made through one.
"""

from __future__ import annotations

import pytest

from shadowspill.errors import PlanInfeasibleError
from shadowspill.ir import shared_residency_footprint

from .test_annotated_plan import _pressurefit_program


def test_a_spill_budget_above_the_profiled_pool_is_planned() -> None:
    problem = _pressurefit_program()
    beyond = problem.maximum_spill_budget_bytes * 4

    config, _facts = problem.machine_inputs(spill_budget_bytes=beyond)

    # The budget reaches exactly one place, and it is the one the simulator
    # charges evictions against.
    assert config.spill_capacity_bytes == beyond


def test_the_execution_budget_keeps_its_ceiling() -> None:
    """The device was measured; a budget past it would report what nothing did."""

    problem = _pressurefit_program()
    with pytest.raises(ValueError, match="execution budget exceeds"):
        problem.machine_inputs(
            execution_budget_bytes=problem.maximum_execution_budget_bytes + 1
        )


def test_an_execution_budget_below_the_reserve_is_infeasible() -> None:
    """A budget the program cannot fit is a point a sweep records, not a crash."""

    problem = _pressurefit_program()
    shared = shared_residency_footprint(problem.program).for_device(
        problem.admission_facts.device_id
    )
    smallest = problem.fixed_execution_bytes + shared + problem.object_reserve_bytes + 1

    problem.machine_inputs(execution_budget_bytes=smallest)
    with pytest.raises(PlanInfeasibleError, match="no positive pool/object") as caught:
        problem.machine_inputs(execution_budget_bytes=smallest - 1)
    assert caught.value.kind == "analytic_capacity"
    assert caught.value.required_bytes == smallest
    assert caught.value.capacity_bytes == smallest - 1


def test_the_recorded_maximum_is_left_as_provenance() -> None:
    problem = _pressurefit_program()
    assert problem.maximum_spill_budget_bytes > 0
