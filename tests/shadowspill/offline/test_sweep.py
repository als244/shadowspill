"""CPU sweeps preserve every result and select winners from actual simulations."""

from dataclasses import replace

import pytest

from shadowspill import MemoryBudgets, SweepResult, apply_profiles, sweep_programs
from shadowspill.errors import PlanSearchExhaustedError

from ._example import communication, options, profiled, profiles, program


def test_sweep_keeps_each_plan_and_failure_across_archive_roundtrip(tmp_path):
    costs = profiles()
    costs.tasks["update_impl"] = replace(
        costs.tasks["update_impl"], duration_seconds=5e-6
    )
    budgets = (MemoryBudgets(99, 64), MemoryBudgets(100, 64), MemoryBudgets(200, 64))
    result = sweep_programs(
        {"slow": apply_profiles(program(), costs), "fast": profiled()},
        memory_budgets=budgets,
        communication_info=communication(),
        search_options=options(),
    )
    result.save(tmp_path / "sweep.ssp")
    loaded = SweepResult.load(tmp_path / "sweep.ssp")
    assert loaded.to_dict() == result.to_dict()
    assert len(loaded.cases) == 6
    assert len(loaded.best_per_budget) == 2
    for case in loaded.cases:
        if case.memory_budgets == budgets[0]:
            assert case.plan is None
            assert case.failure["type"] == "PlanInfeasibleError"
            assert case.failure["message"]
        else:
            plan = case.plan
            assert plan is not None and plan.simulation_only_reasons
            assert plan.timeline.makespan_ns == (
                7000 if case.candidate == "slow" else 3000
            )
            assert plan.memory_budgets == case.memory_budgets
            assert (
                plan.summary["execution_peak_bytes"]
                <= case.memory_budgets.execution_bytes
            )
            if case.candidate == "fast":
                assert loaded.best_per_budget[case.memory_budgets] is plan


def test_sweep_uses_supplied_communication_for_each_case():
    result = sweep_programs(
        {"a": profiled(), "b": profiled()},
        memory_budgets=[MemoryBudgets(100, 64), MemoryBudgets(200, 64)],
        communication_info=communication(2e-6),
        search_options=options(),
    )
    assert all(case.plan.timeline.makespan_ns == 7000 for case in result.cases)


def test_search_exhaustion_is_retained_without_hiding_programming_errors(monkeypatch):
    from shadowspill.planning import sweep

    original = sweep.plan_program
    calls = 0

    def search(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise PlanSearchExhaustedError("search limit", diagnostics=("limit",))
        return original(*args, **kwargs)

    monkeypatch.setattr(sweep, "plan_program", search)
    kwargs = dict(
        memory_budgets=[MemoryBudgets(100, 64)],
        communication_info=communication(),
        search_options=options(),
    )
    result = sweep_programs({"limited": profiled(), "next": profiled()}, **kwargs)
    assert result.cases[0].failure["type"] == "PlanSearchExhaustedError"
    assert result.cases[0].failure["diagnostics"] == ("limit",)
    assert result.cases[1].plan is not None

    def broken(*args, **kwargs):
        raise RuntimeError("invalid task definition")

    monkeypatch.setattr(sweep, "plan_program", broken)
    with pytest.raises(RuntimeError, match="invalid task definition"):
        sweep_programs({"broken": profiled()}, **kwargs)
