"""Offline exploration preserves measured evidence and executable admission."""

import json
import re
import subprocess
import sys
from dataclasses import replace
from itertools import product

import pytest

from engine_tests.training.test_local_planning import measured as measured
from shadowspill import MemoryBudgets
from shadowspill.artifacts._values import plain
from shadowspill.training.preparation.offline import search as local_search
from shadowspill_client import Memory


@pytest.mark.parametrize("measured", [1, 2], indirect=True)
def test_sweep_replans_each_point_and_preserves_inputs(measured):
    _, _, inputs, target = measured
    before = {
        key: (source.to_dict(), inputs.communication[key].to_dict())
        for key, source in inputs.sources.items()
    }
    measured_config = replace(target.layouts.setup.config, memory=inputs.config.memory)
    original = inputs.plan(config=measured_config)
    baseline = inputs.sweep()
    assert {row.candidate for row in baseline.rows} == set(original.candidates)
    for row in baseline.rows:
        assert row.failure is None
        assert row.predicted_seconds == max(
            original.plans[key]["timeline"]["makespan_ns"] / 1e9
            for key in inputs.candidates[row.candidate]
        )
    result = inputs.sweep(device_gib=[2, 3], spill_gib=[1, 2], transfer_scale=[0.5, 2])
    assert {(r.device_gib, r.spill_gib, r.transfer_scale) for r in result.rows} == set(
        product([2, 3], [1, 2], [0.5, 2])
    )
    for row in result.rows:
        assert row.failure is None
        assert row.device_peak_bytes > 0
        for plans in result._plans[row.point, row.candidate].values():
            for plan in plans.values():
                assert plan.memory_budgets == MemoryBudgets(
                    int(row.device_gib * 2**30), int(row.spill_gib * 2**30)
                )
                assert plan.communication_info.fetch_bytes_per_second == (
                    64e6 * row.transfer_scale
                )
    assert before == {
        key: (source.to_dict(), inputs.communication[key].to_dict())
        for key, source in inputs.sources.items()
    }
    for key, source in inputs.sources.items():
        recorded = result.measurements["captures"][key]
        assert recorded["profiled_program"] == source.digest
        assert plain(recorded["communication"]) == before[key][1]


def test_bandwidth_hypotheses_preserve_directional_ratios_and_memory(measured):
    _, _, inputs, _ = measured
    communication = {
        key: replace(
            value,
            fetch_concurrent_bytes_per_second=24e6,
            evict_concurrent_bytes_per_second=16e6,
            evict_bytes_per_second=48e6,
            fetch_latency_seconds=1e-6,
            evict_latency_seconds=2e-6,
            scale_up_bytes_per_second=100e9,
            scale_up_latency_seconds=3e-6,
        )
        for key, value in inputs.communication.items()
    }
    source = replace(inputs, communication=communication)
    result = source.sweep(transfer_scale=[0.5, 1, 2])
    for row in result.rows:
        assert row.failure is None
        for plans in result._plans[row.point, row.candidate].values():
            for key, plan in plans.items():
                measured = communication[inputs.requests[key].use.template].to_dict()
                hypothetical = plan.communication_info.to_dict()
                for field in (
                    "fetch_bytes_per_second",
                    "evict_bytes_per_second",
                    "fetch_concurrent_bytes_per_second",
                    "evict_concurrent_bytes_per_second",
                ):
                    assert hypothetical[field] == measured[field] * row.transfer_scale
                    measured.pop(field)
                    hypothetical.pop(field)
                if row.transfer_scale != 1:
                    assert hypothetical["metadata"].pop("timing_kind") == "estimated"
                assert hypothetical == measured


def test_all_infeasible_points_remain_reportable_but_not_executable(measured, tmp_path):
    _, _, inputs, target = measured
    result = inputs.sweep(device_gib=1 / 2**30)
    assert result.rows and all(row.failure for row in result.rows)
    assert all(row.predicted_seconds is None and not row.best for row in result.rows)
    report = result.write_report(tmp_path / "infeasible.html")
    assert "Predicted" in report.read_text()
    config = replace(target.layouts.setup.config, memory=Memory(1 / 2**30, 1))
    with pytest.raises(ValueError, match="selected candidates"):
        inputs.plan(config=config)


def test_programming_errors_are_not_reported_as_capacity_failures(
    measured, monkeypatch
):
    _, _, inputs, _ = measured

    def broken(*args, **kwargs):
        raise ValueError("broken source")

    monkeypatch.setattr(local_search, "plan_program", broken)
    with pytest.raises(ValueError, match="broken source"):
        inputs.sweep()


def test_different_capacities_are_not_ranked_against_each_other(measured):
    _, _, inputs, _ = measured
    key = next(iter(inputs.candidates))
    other = "0" * 64
    larger = plain(inputs.descriptions[key])
    larger["capacities"] = {k: v * 2 for k, v in larger["capacities"].items()}
    alternatives = replace(
        inputs,
        candidates={key: inputs.candidates[key], other: inputs.candidates[key]},
        descriptions={key: inputs.descriptions[key], other: larger},
    )
    result = alternatives.sweep()
    assert len({row.comparison for row in result.rows}) == 2
    assert all(row.best for row in result.rows)
    same_work = replace(alternatives, descriptions={key: larger, other: larger})
    assert sum(row.best for row in same_work.sweep().rows) == 1


def test_reports_and_old_exports_work_without_frameworks_or_catalog(measured, tmp_path):
    _, path, inputs, _ = measured
    result = inputs.sweep(device_gib=[2, 3])
    report = result.write_report(tmp_path / "sweep.html")
    links = re.findall(r'href="([^"]+)"', report.read_text())
    assert links and all((tmp_path / link).is_file() for link in links)
    assert len(json.loads((tmp_path / "sweep_reports/rows.json").read_text())) == len(
        result.rows
    )
    # Retained measurement exports need no model or recapture to explore again.
    old = replace(inputs, descriptions={}).sweep()
    assert all(row.predicted_seconds is not None for row in old.rows)
    assert len({r.comparison for r in old.rows}) == len(inputs.candidates)
    code = """
import sys
for name in ('torch', 'triton', 'mlops', 'shadowspill_workloads'):
    sys.modules[name] = None
from shadowspill import PlanningInputs
result = PlanningInputs.load(sys.argv[1]).sweep(transfer_scale=[0.5, 1])
assert result.rows and all(row.predicted_seconds is not None for row in result.rows)
result.write_report(sys.argv[2])
"""
    completed = subprocess.run(
        [sys.executable, "-c", code, str(path), str(tmp_path / "cpu.html")],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("budget", ["device_gib", "spill_gib"])
def test_zero_budgets_are_refused_offline(measured, budget):
    # Zero means automatic, which only the service can size from its hardware.
    _, _, inputs, target = measured
    with pytest.raises(ValueError, match="zero is automatic"):
        inputs.sweep(**{budget: [1, 0]})
    memory = Memory(**{"device_gib": 1, "spill_gib": 1, budget: 0})
    with pytest.raises(ValueError, match="zero is automatic"):
        inputs.plan(config=replace(target.layouts.setup.config, memory=memory))


@pytest.mark.parametrize("values", [[], [0], [-1], [float("inf")], [1, 1], [True]])
def test_invalid_grids_fail_before_search(measured, values):
    _, _, inputs, _ = measured
    with pytest.raises(ValueError):
        inputs.sweep(transfer_scale=values)
