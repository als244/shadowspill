"""Headroom sizes the pool; the enforcement flag controls external overruns."""

from argparse import Namespace
from dataclasses import replace
from types import SimpleNamespace

import pytest

from shadowspill.memory import device
from tools.qualification.device_defaults import numerical_defaults, performance_defaults
from tools.qualification.numerical.cli import _parser as numerical_parser
from tools.qualification.numerical.verdict import _budget_checks
from tools.qualification.performance.manifest import _manifest_with_overrides
from tools.qualification.performance_matrix import _parser as performance_parser


@pytest.mark.parametrize("bf16", [False, True])
@pytest.mark.parametrize("reject", [False, True])
def test_explicit_zero_survives_hardware_defaults(bf16, reject):
    numerical = numerical_parser().parse_args(
        [
            "run",
            "llama3",
            "out",
            "--external-headroom-mib",
            "0",
            *(["--reject-overbudget"] if reject else []),
        ]
    )
    performance = performance_parser().parse_args(
        ["--external-headroom-mib", "0", *(["--reject-overbudget"] if reject else [])]
    )
    numerical_defaults(numerical, hardware={"bf16": bf16})
    performance_defaults(performance, hardware={"bf16": bf16})
    assert numerical.external_headroom_mib == performance.external_headroom_mib == 0
    manifest = _manifest_with_overrides(
        "llama3",
        "mlops",
        spill_budget_gib=None,
        external_headroom_mib=performance.external_headroom_mib,
        reject_overbudget=performance.reject_overbudget,
    )
    pool = device(
        physical_capacity=manifest.device_physical_capacity_bytes,
        external_headroom=manifest.external_headroom_bytes,
        reject_overbudget=manifest.reject_overbudget,
    )
    assert pool.external_headroom == 0
    assert pool.reject_overbudget == numerical.reject_overbudget == reject


@pytest.mark.parametrize("headroom", [0, 512])
@pytest.mark.parametrize("reject", [False, True])
@pytest.mark.parametrize("failure", [None, "pool", "spill", "runtime"])
def test_numerical_reports_external_overshoot_but_enforces_pool_contracts(
    headroom, failure, reject
):
    cap = 5 << 30
    request = SimpleNamespace(
        device_budget=cap, external_headroom_mib=headroom, reject_overbudget=reject
    )
    run = SimpleNamespace(
        report=SimpleNamespace(predicted_device_peak_bytes=cap - 1),
        physical_statuses=[43] if failure == "runtime" else [0],
    )
    result = {
        "physical_budget_sealed": True,
        "peak_process_physical_bytes": cap + (1 << 30),
        "execution_pool_bytes": cap,
        "slab_peak_allocated_bytes": cap + 1 if failure == "pool" else cap,
        "spill_pool_bytes": 1 << 30,
        "spill_peak_allocated_bytes": (1 << 30) + (failure == "spill"),
    }
    passed = all(bool(check[1]) for check in _budget_checks(result, request, run))
    assert passed == (not reject and failure is None)


@pytest.mark.parametrize("headroom", [0, 512])
@pytest.mark.parametrize("reject", [False, True])
@pytest.mark.parametrize("failure", [None, "spill", "runtime"])
def test_performance_reports_external_overshoot_but_enforces_other_checks(
    headroom, failure, reject
):
    from tools.qualification.performance.baseline import regression_comparison
    from tools.qualification.performance.verdict import _gate_verdicts
    from workloads.full_model import manifest_for

    manifest = replace(
        manifest_for("llama3", "mlops"),
        external_headroom_bytes=headroom << 20,
        reject_overbudget=reject,
    )
    arguments = Namespace(groups=3, steps_per_group=4, skip_checkpoint=True)
    planned = SimpleNamespace(
        report=SimpleNamespace(
            predicted_makespan_ns=1_000_000_000, predicted_device_peak_bytes=1 << 30
        ),
        training=SimpleNamespace(_step=13),
    )
    statistics = SimpleNamespace(
        peak_process_physical_bytes=manifest.device_physical_capacity_bytes + (1 << 30),
        callback_failures=0,
        pointer_lookup_failures=0,
        runtime=SimpleNamespace(queued_actions=0, pending_retirements=0),
    )
    gates = _gate_verdicts(
        manifest,
        arguments,
        planned,
        SimpleNamespace(warm_objectives=[1.0], checkpoint_restored=False),
        SimpleNamespace(cycle_seconds=[1.0], measured_objectives=[[1.0]]),
        runtime=SimpleNamespace(
            pool_statistics=lambda _: SimpleNamespace(
                peak_allocated_bytes=manifest.spill_budget_bytes + (failure == "spill")
            )
        ),
        execution_statistics=statistics,
        runtime_delta=dict.fromkeys(
            (
                "device_allocations",
                "pinned_host_registrations",
                "event_driver_creates",
                "event_growth_rejections",
            ),
            0,
        ),
        physical_statuses=[43] if failure == "runtime" else [0],
        regression=regression_comparison(
            manifest, arguments, device_name="RTX 2080 Ti"
        ),
    )
    assert gates.passed == (not reject and failure is None)


def test_default_reservation_and_enforcement_are_independent():
    pool = device(physical_capacity=5 << 30)
    assert pool.external_headroom == 512 << 20
    assert pool.reject_overbudget is False
    for headroom in (0, 512 << 20):
        strict = device(
            physical_capacity=5 << 30,
            external_headroom=headroom,
            reject_overbudget=True,
        )
        reporting = replace(strict, reject_overbudget=False)
        assert reporting.physical_capacity == strict.physical_capacity
        assert reporting.external_headroom == strict.external_headroom


@pytest.mark.parametrize("reject", [False, True])
def test_plan_round_trip_preserves_the_enforcement_flag(reject):
    from shadowspill.ir import ExecutionPlan, index_execution_plan
    from tests.shadowspill.ir._examples import representative_plan

    plan = representative_plan()
    plan = replace(plan, admission=replace(plan.admission, reject_overbudget=reject))
    restored = ExecutionPlan.from_json(plan.to_json())
    assert restored.admission.reject_overbudget == reject
    assert index_execution_plan(restored).reject_overbudget == reject
