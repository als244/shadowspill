"""Budget search consumes prepared programs, and admission never reruns search."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from shadowspill.errors import AdmissionError
from shadowspill.pipeline.common import PlanningTimer
from shadowspill.planner import StepDataOrdering
from shadowspill.pytorch import PreparedStep, admit_step, search_prepared_steps
from shadowspill.pytorch.planning.training.plan import admit_selected_training_program
from shadowspill.runtime.topology import MemoryPool
from tests.benchmarking._fixtures import _fixture


def test_each_budget_returns_a_plan_with_estimated_behavior(tmp_path):
    step = _fixture()
    candidate = PreparedStep((step,), (), (), (), step.signature_digests, "", {})
    report = search_prepared_steps(
        {"geometry": candidate},
        budgets=[(224, 1024), (320, 1024)],
        artifact_store=tmp_path,
    )
    assert set(report.winner_plans) == {(224, 1024), (320, 1024)}
    assert len(report.geometries) == 1
    for budget, plan in report.winner_plans.items():
        assert plan.memory_budgets.execution_bytes == budget[0]
        summary = plan.summary.as_dict()
        assert summary["estimate"]
        assert summary["simulated_step_seconds"] > 0
        assert summary["device_peak_bytes"] > 0
        assert 0 < summary["physical_execution_peak_bytes"] <= budget[0]
        for name in (
            "spill_peak_bytes",
            "recomputing_group_percent",
            "idle_percent",
            "fetch_utilization_percent",
            "evict_utilization_percent",
            "transfer_bytes_fetched",
            "transfer_bytes_evicted",
            "fetch_blended_bandwidth_bytes_per_second",
            "evict_blended_bandwidth_bytes_per_second",
        ):
            assert name in summary
        restored = type(plan).from_json(plan.to_json())
        assert restored.summary.as_dict() == summary
        assert restored.digest == plan.digest


def test_exact_admission_checks_budgets_and_program_and_preserves_schedule(
    tmp_path, monkeypatch
):
    step = _fixture()
    candidate = PreparedStep((step,), (), (), (), step.signature_digests, "", {})
    report = search_prepared_steps(
        {"geometry": candidate}, budgets=[(224, 1024)], artifact_store=tmp_path
    )
    plan = report.winner_plans[(224, 1024)]
    import shadowspill.pytorch.planning.training.plan as module

    monkeypatch.setattr(
        module,
        "resolve_plan",
        lambda *a, **kw: pytest.fail("searched during admission"),
    )
    programs = SimpleNamespace(
        lowered=SimpleNamespace(
            program=step.problem.program,
            initial_residency=step.problem.initial_residency,
            final_residency=step.problem.final_residency,
        ),
        admission=step.problem.admission_facts,
        simulation_config=step.problem.simulation_config,
        measurements_by_profile={},
        dynamic_scratch_reserve_bytes=0,
    )
    selection = admit_selected_training_program(
        programs,
        plan,
        timer=PlanningTimer(verbose=False),
        execution_budget=224,
        spill_budget=1024,
    )
    assert selection.result.schedule == plan.result.schedule
    assert selection.result.selections == plan.result.selections
    assert selection.search_wall_time_ns == 0
    with pytest.raises(AdmissionError, match="budgets differ"):
        admit_selected_training_program(
            programs,
            plan,
            timer=PlanningTimer(verbose=False),
            execution_budget=225,
            spill_budget=1024,
        )
    programs.lowered.program = replace(step.problem.program, tasks=())
    with pytest.raises(AdmissionError, match="computation differs"):
        admit_selected_training_program(
            programs,
            plan,
            timer=PlanningTimer(verbose=False),
            execution_budget=224,
            spill_budget=1024,
        )


def test_prepared_admission_converts_physical_subbudget_and_equivalent_orderings(
    tmp_path, monkeypatch
):
    step = _fixture()
    equivalent = replace(
        step, data_ordering=StepDataOrdering(1, 1, reverse_breadth=False)
    )
    candidate = PreparedStep(
        (step, equivalent), (), (), (), step.signature_digests, "", {}
    )
    report = search_prepared_steps(
        {"geometry": candidate}, budgets=[(224, 1024)], artifact_store=tmp_path
    )
    plan = report.winner_plans[(224, 1024)]
    calls = []

    def bind(model, **kwargs):
        calls.append(kwargs)
        return "callable"

    monkeypatch.setattr("shadowspill.pytorch.api.plan_step", bind)
    runtime = SimpleNamespace(
        pools={"execution": MemoryPool("execution", 0, "device", 448, 512, 0)}
    )
    for ordering in (None, equivalent.data_ordering):
        assert (
            admit_step(
                object(),
                candidate,
                plan,
                runtime=runtime,
                execution="execution",
                data_ordering=ordering,
            )
            == "callable"
        )
        assert calls[-1]["execution_budget"] == 224 - 64
        assert calls[-1]["selected_plan"] is plan
    with pytest.raises(AdmissionError, match="budgets differ"):
        admit_step(
            object(),
            candidate,
            plan,
            runtime=runtime,
            execution="execution",
            execution_budget=225,
        )
