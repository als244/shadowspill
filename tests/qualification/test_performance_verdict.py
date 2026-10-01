"""A throughput baseline may judge only its own hardware and configuration."""

from argparse import Namespace
from dataclasses import replace

import pytest

from qualification.performance.baseline import regression_comparison
from qualification.performance.cases import manifest_for


def _baseline_cell():
    return replace(
        manifest_for("llama3", "mlops"),
        device_physical_capacity_bytes=16 << 30,
        model_dtype="bfloat16",
        grad_dtype="bfloat16",
        opt_state_dtype="bfloat16",
    )


def test_matching_hardware_and_configuration_apply_the_floor() -> None:
    cell = _baseline_cell()
    comparison = regression_comparison(
        cell, Namespace(), device_name="NVIDIA GeForce RTX 5090"
    )
    assert comparison.applicable
    assert comparison.floor == cell.regression_tokens_per_second


@pytest.mark.parametrize(
    "changes",
    [
        {"device_physical_capacity_bytes": 10 << 30},
        {"spill_budget_bytes": 90 << 30},
        {"model_dtype": "float16"},
        {"master_dtype": "float32"},
        {"grad_dtype": "float32"},
        {"opt_state_dtype": "float32"},
        {"accumulation_count": 4},
    ],
)
def test_a_configuration_change_does_not_inherit_an_unrelated_floor(changes) -> None:
    comparison = regression_comparison(
        replace(_baseline_cell(), **changes),
        Namespace(),
        device_name="NVIDIA GeForce RTX 5090",
    )
    assert not comparison.applicable
    assert comparison.mismatches


def test_another_gpu_reports_throughput_without_the_5090_floor() -> None:
    comparison = regression_comparison(
        _baseline_cell(), Namespace(), device_name="NVIDIA GeForce RTX 2080 Ti"
    )
    assert not comparison.applicable
    assert "device" in comparison.reason


def test_current_defaults_preserve_the_existing_baseline_on_the_original_gpu() -> None:
    comparison = regression_comparison(
        manifest_for("llama3", "mlops"),
        Namespace(),
        device_name="NVIDIA GeForce RTX 5090",
    )
    assert comparison.applicable


def test_remote_link_is_not_inferred_from_gpu_identity() -> None:
    comparison = regression_comparison(
        _baseline_cell(),
        Namespace(remote_spill="h:1:1"),
        device_name="NVIDIA GeForce RTX 5090",
    )
    assert not comparison.applicable
    assert "remote link" in comparison.reason


def test_a_cell_without_a_baseline_is_explicitly_not_applicable() -> None:
    comparison = regression_comparison(
        manifest_for("llama3", "pytorch"), Namespace(), device_name="anything"
    )
    assert not comparison.applicable
    assert comparison.floor is None


@pytest.mark.parametrize("failure", ["simulator", "allocation"])
def test_an_unrelated_gpu_still_fails_runtime_and_simulator_checks(failure) -> None:
    from types import SimpleNamespace

    from qualification.performance.verdict import _gate_verdicts

    manifest = _baseline_cell()
    arguments = Namespace(groups=3, steps_per_group=4, skip_checkpoint=True)
    regression = regression_comparison(manifest, arguments, device_name="RTX 2080 Ti")
    planned = SimpleNamespace(
        report=SimpleNamespace(
            predicted_makespan_ns=1_000_000_000, predicted_device_peak_bytes=1 << 30
        ),
        training=SimpleNamespace(_step=13),
    )
    warm = SimpleNamespace(warm_objectives=[1.0], checkpoint_restored=False)
    measured = SimpleNamespace(
        cycle_seconds=[1.25 if failure == "simulator" else 1.0],
        measured_objectives=[[1.0]],
    )
    statistics = SimpleNamespace(
        peak_process_physical_bytes=1 << 30,
        callback_failures=0,
        pointer_lookup_failures=0,
        runtime=SimpleNamespace(queued_actions=0, pending_retirements=0),
    )
    delta = dict.fromkeys(
        (
            "device_allocations",
            "pinned_host_registrations",
            "event_driver_creates",
            "event_growth_rejections",
        ),
        0,
    )
    if failure == "allocation":
        delta["device_allocations"] = 1
    gates = _gate_verdicts(
        manifest,
        arguments,
        planned,
        warm,
        measured,
        runtime=SimpleNamespace(
            pool_statistics=lambda _: SimpleNamespace(peak_allocated_bytes=0)
        ),
        execution_statistics=statistics,
        runtime_delta=delta,
        physical_statuses=[],
        regression=regression,
    )
    assert gates.regression_passed
    assert gates.regression_ratio is None
    assert not gates.passed
    assert not (
        gates.simulator_passed if failure == "simulator" else gates.strict_runtime
    )
