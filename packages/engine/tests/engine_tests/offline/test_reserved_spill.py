"""A Plan searched beside other operations' persistent state plans within the rest."""

import pytest

from shadowspill import MemoryBudgets, Plan, plan_program
from shadowspill.errors import PlanInfeasibleError

from ._example import communication, options, profiled


def planned(spill, reserved):
    return plan_program(
        profiled(),
        memory_budgets=MemoryBudgets(100, spill),
        communication_info=communication(),
        search_options=options(),
        reserved_spill_bytes=reserved,
    )


def test_the_search_plans_within_the_budget_less_the_reservation(tmp_path):
    # The example keeps 64 bytes of persistent spill.
    plan = planned(128, 64)
    assert plan.reserved_spill_bytes == 64
    assert plan.timeline.spill_peak_bytes <= 128 - 64
    plan.save(tmp_path / "plan.ssp")
    assert Plan.load(tmp_path / "plan.ssp").reserved_spill_bytes == 64
    with pytest.raises(PlanInfeasibleError):
        planned(128, 65)


@pytest.mark.parametrize("reserved", [-1, 129, 1.0])
def test_a_reservation_beyond_the_budget_is_refused(reserved):
    with pytest.raises(ValueError, match="reserved_spill_bytes"):
        planned(128, reserved)
