"""What the cell is judged to be, and the artifact that says so."""

from __future__ import annotations

import argparse
import math
import statistics
from dataclasses import dataclass
from typing import Any

from shadowspill.pytorch import (
    Runtime,
)
from shadowspill.schema import artifact_schema
from tools.qualification.runtime_evidence import (
    adapter_statistics,
    statistics_dict,
)
from workloads.full_model import FullModelManifest

from .phases import _Measurements, _PlannedCase, _WarmStep
from .readings import _artifact_identity, _runtime_delta, _wait_idle

_MINIMUM_REGRESSION_RATIO = 0.95


def regression_authority(
    manifest: FullModelManifest, arguments: argparse.Namespace
) -> float | None:
    """The throughput floor this cell is judged against, or ``None``.

    A cell spilling to a peer (``--remote-spill``) is judged against the
    manifest's remote floor, measured with the pool on a peer; a cell spilling
    to pinned host memory against the local one. The two differ by the link's
    ratio, so neither says anything about the other's run.
    """

    if getattr(arguments, "remote_spill", None) is not None:
        return manifest.remote_regression_tokens_per_second
    return manifest.regression_tokens_per_second

#: The simulator prices the selected span and the terminal tail; the opening
#: restore is unmodeled but, since first-use ordering of the initial
#: placement batch (shadowspill.ir.schedule.first_use_initial_order), bounded
#: by the first task's own inputs rather than the whole initial set. The
#: bound has room in it because the calibrated transfer bandwidths the plan
#: is priced against move run to run: a calibration below the rate the
#: hardware then delivers prices the step pessimistically without the
#: simulator being wrong. The remaining unmodeled terms are the
#: terminal-drain serialization, input staging, and profile fidelity.
_MAXIMUM_SIMULATOR_ERROR = 0.10


@dataclass(frozen=True, slots=True)
class _Gates:
    """Every gate's verdict, and the numbers each one was decided on."""

    median_step_seconds: float
    median_throughput: float
    simulator_relative_error: float
    regression_ratio: float | None
    predecessor_ratio: float | None
    expected_logical_steps: int
    protocol_complete: bool
    objectives_finite: bool
    logical_steps_passed: bool
    physical_passed: bool
    strict_runtime: bool
    simulator_passed: bool
    regression_passed: bool

    @property
    def passed(self) -> bool:
        return bool(
            self.protocol_complete
            and self.objectives_finite
            and self.logical_steps_passed
            and self.physical_passed
            and self.strict_runtime
            and self.simulator_passed
            and self.regression_passed
        )


def _gate_verdicts(
    manifest: FullModelManifest,
    arguments: argparse.Namespace,
    planned: _PlannedCase,
    warm: _WarmStep,
    measured: _Measurements,
    *,
    runtime: Runtime,
    execution_statistics: Any,
    runtime_delta: dict[str, int],
    physical_statuses: list[object],
) -> _Gates:
    """Decide every gate, each one naming the kind of failure it reports."""

    # The step is the compute stream's cycle, origin to origin; the host's
    # own view of a group is reported beside it and decides nothing.
    median_step_seconds = float(statistics.median(measured.cycle_seconds))
    median_throughput = manifest.tokens_per_step / median_step_seconds
    authority = regression_authority(manifest, arguments)
    predicted_seconds = planned.report.predicted_makespan_ns / 1e9
    simulator_relative_error = (
        (median_step_seconds - predicted_seconds) / predicted_seconds
        if predicted_seconds > 0.0
        else math.inf
    )
    expected_logical_steps = arguments.groups * arguments.steps_per_group + int(
        arguments.skip_checkpoint
    )
    return _Gates(
        median_step_seconds=median_step_seconds,
        median_throughput=median_throughput,
        simulator_relative_error=simulator_relative_error,
        regression_ratio=(
            None if authority is None else median_throughput / authority
        ),
        predecessor_ratio=(
            None
            if manifest.predecessor_tokens_per_second is None
            else median_throughput / manifest.predecessor_tokens_per_second
        ),
        expected_logical_steps=expected_logical_steps,
        protocol_complete=arguments.groups == 3 and arguments.steps_per_group == 4,
        objectives_finite=all(
            math.isfinite(value)
            for values in (warm.warm_objectives, *measured.measured_objectives)
            for value in values
        ),
        logical_steps_passed=bool(
            (arguments.skip_checkpoint or warm.checkpoint_restored)
            and planned.training._step == expected_logical_steps
        ),
        physical_passed=bool(
            not any(physical_statuses)
            and planned.report.predicted_device_peak_bytes
            <= manifest.device_physical_capacity_bytes
            and int(execution_statistics.peak_process_physical_bytes)
            <= manifest.device_physical_capacity_bytes
            and int(runtime.pool_statistics("spill").peak_allocated_bytes)
            <= manifest.spill_budget_bytes
        ),
        strict_runtime=bool(
            runtime_delta["device_allocations"] == 0
            and runtime_delta["pinned_host_registrations"] == 0
            and runtime_delta["event_driver_creates"] == 0
            and runtime_delta["event_growth_rejections"] == 0
            and int(execution_statistics.callback_failures) == 0
            and int(execution_statistics.pointer_lookup_failures) == 0
            and int(execution_statistics.runtime.queued_actions) == 0
            and int(execution_statistics.runtime.pending_retirements) == 0
        ),
        simulator_passed=abs(simulator_relative_error) <= _MAXIMUM_SIMULATOR_ERROR,
        regression_passed=bool(
            authority is None
            or median_throughput / authority >= _MINIMUM_REGRESSION_RATIO
        ),
    )


def _plan_only_result(
    manifest: FullModelManifest,
    planned: _PlannedCase,
    physical_statuses: list[object],
    *,
    planning_spill_budget: int,
    capabilities: dict[str, object],
) -> dict[str, object]:
    """The artifact for a cell that was asked to plan and stop."""

    report = planned.report
    return {
        "schema": artifact_schema("full_model_qualification"),
        "manifest": manifest.as_dict(),
        "plan_only": True,
        "passed": not any(physical_statuses),
        "planning_seconds": planned.planning_seconds,
        "phase_seconds": planned.phase_seconds,
        "predicted_makespan_seconds": report.predicted_makespan_ns / 1e9,
        "predicted_device_peak_bytes": report.predicted_device_peak_bytes,
        "predicted_spill_peak_bytes": report.predicted_spill_peak_bytes,
        "transfer_bytes_evicted": report.transfer_bytes_evicted,
        "transfer_bytes_fetched": report.transfer_bytes_fetched,
        "plan_report_artifact": _artifact_identity(planned.plan_path),
        "plan_records": planned.plan_records,
        "physical_budget_statuses": physical_statuses,
        "planning_spill_budget_bytes": planning_spill_budget,
        "runtime_transfer_capabilities": capabilities,
    }


def _measured_result(
    manifest: FullModelManifest,
    arguments: argparse.Namespace,
    planned: _PlannedCase,
    warm: _WarmStep,
    measured: _Measurements,
    *,
    runtime: Runtime,
    physical_statuses: list[object],
    planning_spill_budget: int,
    capabilities: dict[str, object],
    calibration_attempts: int,
) -> dict[str, object]:
    """The artifact for a measured cell: what was seen, and what it is judged to be."""

    # Releasing the retained StepResult tensors above enqueues retirements
    # through the free callback, after the last per-group drain.  Sample
    # the gate evidence at a quiesced boundary so pending_retirements
    # reflects a leak rather than a race with the worker.
    _wait_idle(planned.training)
    execution_statistics = adapter_statistics()
    runtime_delta = _runtime_delta(warm.execution_baseline, execution_statistics)
    gates = _gate_verdicts(
        manifest,
        arguments,
        planned,
        warm,
        measured,
        runtime=runtime,
        execution_statistics=execution_statistics,
        runtime_delta=runtime_delta,
        physical_statuses=physical_statuses,
    )
    report = planned.report
    return {
        "schema": artifact_schema("full_model_qualification"),
        "manifest": manifest.as_dict(),
        "plan_only": False,
        "passed": gates.passed,
        "protocol_complete": gates.protocol_complete,
        "groups": arguments.groups,
        "steps_per_group": arguments.steps_per_group,
        "planning_seconds": planned.planning_seconds,
        "phase_seconds": planned.phase_seconds,
        "checkpoint_seconds": warm.checkpoint_seconds,
        "checkpoint_skipped": bool(arguments.skip_checkpoint),
        "warm_seconds": warm.warm_seconds,
        "restore_seconds": warm.restore_seconds,
        "checkpoint_restored": warm.checkpoint_restored,
        "warm_objectives": warm.warm_objectives,
        "warm_diagnostics": warm.warm_diagnostics.as_dict(),
        "measured_objectives": measured.measured_objectives,
        "objectives_finite": gates.objectives_finite,
        "logical_steps": planned.training._step,
        "expected_logical_steps": gates.expected_logical_steps,
        "logical_steps_passed": gates.logical_steps_passed,
        "group_seconds": measured.group_seconds,
        "group_tokens_per_second": measured.group_tokens_per_second,
        "median_step_seconds": gates.median_step_seconds,
        "cycle_seconds": measured.cycle_seconds,
        "opening_delay_seconds": measured.opening_delay_seconds,
        "host_group_seconds": measured.host_group_seconds,
        "median_tokens_per_second": gates.median_throughput,
        "selected_task_span_seconds": measured.selected_spans,
        "median_selected_task_span_seconds": float(
            statistics.median(measured.selected_spans)
        ),
        "dispatch_seconds": measured.dispatch_seconds,
        "prior_invocation_drain_seconds": measured.prior_invocation_drain_seconds,
        "predicted_makespan_seconds": report.predicted_makespan_ns / 1e9,
        "simulator_relative_error": gates.simulator_relative_error,
        "simulator_gate_passed": gates.simulator_passed,
        "regression_throughput_ratio": gates.regression_ratio,
        "regression_gate_passed": gates.regression_passed,
        "predecessor_throughput_ratio": gates.predecessor_ratio,
        "predicted_device_peak_bytes": report.predicted_device_peak_bytes,
        "predicted_spill_peak_bytes": report.predicted_spill_peak_bytes,
        "transfer_bytes_evicted": report.transfer_bytes_evicted,
        "transfer_bytes_fetched": report.transfer_bytes_fetched,
        "physical_budget_statuses": physical_statuses,
        "physical_budget_passed": gates.physical_passed,
        "peak_process_physical_bytes": int(
            execution_statistics.peak_process_physical_bytes
        ),
        # What the provider actually took outside the slab, which is what the
        # configured provider headroom is a prediction of. Recorded so the
        # prediction can be set from measurement rather than from a round number.
        "observed_external_high_water_bytes": int(
            execution_statistics.observed_external_high_water_bytes
        ),
        "planning_spill_budget_bytes": planning_spill_budget,
        "calibration_attempts": calibration_attempts,
        "strict_runtime_passed": gates.strict_runtime,
        "runtime_delta": runtime_delta,
        "runtime_statistics": statistics_dict(execution_statistics),
        "runtime_transfer_capabilities": capabilities,
        "plan_report_artifact": _artifact_identity(planned.plan_path),
        "plan_records": planned.plan_records,
    }
