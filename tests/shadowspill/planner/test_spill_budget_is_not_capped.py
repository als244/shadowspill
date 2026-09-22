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


def test_the_recorded_maximum_is_left_as_provenance() -> None:
    problem = _pressurefit_program()
    assert problem.maximum_spill_budget_bytes > 0
