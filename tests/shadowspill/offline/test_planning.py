from dataclasses import replace

import pytest

from shadowspill import MemoryBudgets, apply_profiles, plan_program
from shadowspill.errors import PlanInfeasibleError
from shadowspill.ir import MemoryLocation, ResidencySpec
from tests.shadowspill.planner._examples import recomputation_program

from ._costed_fixture import split_costed_program
from ._example import communication, options, profiled, profiles, program


def test_known_fetch_update_writeback_and_total_gpu_budget():
    result = plan_program(
        profiled(),
        memory_budgets=MemoryBudgets(100, 64),
        communication_info=communication(),
        search_options=options(),
    )
    assert result.timeline.makespan_ns == 3000
    assert result.summary["execution_peak_bytes"] == 100
    assert result.summary["spill_peak_bytes"] == 64
    assert result.summary["fetch_bytes"] == 64
    assert result.summary["evict_bytes"] == 64
    assert result.timeline.task_intervals[1].start_ns == 1000
    assert result.simulation_only_reasons


def test_latency_is_charged_in_timeline():
    result = plan_program(
        profiled(),
        memory_budgets=MemoryBudgets(100, 64),
        communication_info=communication(2e-6),
        search_options=options(),
    )
    assert result.timeline.makespan_ns == 7000
    assert result.timeline.task_intervals[1].start_ns == 3000
    assert [t.end_ns - t.start_ns for t in result.timeline.transfer_intervals] == [
        3000,
        3000,
    ]


def test_duration_and_bandwidth_overrides_change_simulation_without_build():
    first = profiled()
    data = profiles()
    data.tasks["update_impl"] = replace(
        data.tasks["update_impl"], duration_seconds=5e-6
    )
    second = apply_profiles(program(), data)
    result = plan_program(
        second,
        memory_budgets=MemoryBudgets(100, 64),
        communication_info=replace(communication(), fetch_bytes_per_second=32e6),
        search_options=options(),
    )
    assert result.timeline.makespan_ns == 8000
    assert first.task_profiles["update_impl"].duration_seconds == 1e-6


@pytest.mark.parametrize(
    "budget", [MemoryBudgets(20, 64), MemoryBudgets(99, 64), MemoryBudgets(100, 63)]
)
def test_budget_failure_has_diagnostics(budget):
    with pytest.raises(PlanInfeasibleError) as error:
        plan_program(
            profiled(),
            memory_budgets=budget,
            communication_info=communication(),
            search_options=options(),
        )
    assert error.value.kind


def test_a_reservation_stays_out_of_the_budget_while_planning():
    """Room for a Program that will share the worker: the plan fits within the
    budget less the reservation and still declares the whole budget; a
    reservation that leaves too little is refused like a smaller budget."""
    arguments = dict(communication_info=communication(), search_options=options())
    result = plan_program(
        profiled(),
        memory_budgets=MemoryBudgets(110, 64),
        reserved_bytes=10,
        **arguments,
    )
    assert result.memory_budgets.execution_bytes == 110
    assert result.summary["execution_peak_bytes"] == 100
    with pytest.raises(PlanInfeasibleError) as error:
        plan_program(
            profiled(),
            memory_budgets=MemoryBudgets(110, 64),
            reserved_bytes=11,
            **arguments,
        )
    assert error.value.kind
    with pytest.raises(ValueError, match="nonnegative"):
        plan_program(
            profiled(),
            memory_budgets=MemoryBudgets(110, 64),
            reserved_bytes=-1,
            **arguments,
        )


def test_save_recompute_alternatives_remain_available():
    p, costs = split_costed_program(
        recomputation_program(),
        initial_residency=(ResidencySpec("input_storage", MemoryLocation.DEVICE),),
        fixed_device_bytes=0,
    )
    result = plan_program(
        apply_profiles(p, costs),
        memory_budgets=MemoryBudgets(120, 1000),
        communication_info=communication(),
        search_options=options(),
    )
    assert result.selections
    assert len(p.task_alternative_groups[0].options) == 2
    assert result.timeline.device_peaks[0].total_bytes <= 120
