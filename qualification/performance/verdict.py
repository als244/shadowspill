"""What the cell is judged to be, and the artifact that says so."""

from __future__ import annotations

import argparse
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qualification.performance.cases import FullModelManifest
from qualification.runtime_evidence import (
    adapter_statistics,
    statistics_dict,
)
from shadowspill import Plan
from shadowspill.runtime import Runtime
from shadowspill.schema import artifact_schema

from .baseline import RegressionComparison
from .phases import Measurements, WarmStep
from .readings import _artifact_identity, _runtime_delta

_MINIMUM_REGRESSION_RATIO = 0.95


#: Simulation includes scheduled entry fetches, computation, and required
#: terminal transfers. The measured cycle also includes frontend/caller work;
#: profile fidelity and calibrated bandwidth variation account for other error.
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
    plan: Plan,
    warm: WarmStep,
    measured: Measurements,
    *,
    runtime: Runtime,
    logical_steps: int,
    execution_statistics: Any,
    runtime_delta: dict[str, int],
    physical_statuses: list[int],
    regression: RegressionComparison,
) -> _Gates:
    """Decide every gate, each one naming the kind of failure it reports."""

    # The step is the compute stream's cycle, origin to origin; the host's
    # own view of a group is reported beside it and decides nothing.
    median_step_seconds = float(statistics.median(measured.cycle_seconds))
    median_throughput = manifest.tokens_per_step / median_step_seconds
    authority = regression.floor if regression.applicable else None
    predicted_seconds = plan.summary["duration_seconds"]
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
        regression_ratio=(None if authority is None else median_throughput / authority),
        predecessor_ratio=(
            None
            if manifest.predecessor_tokens_per_second is None
            or not regression.applicable
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
            and logical_steps == expected_logical_steps
        ),
        physical_passed=bool(
            not any(physical_statuses)
            and plan.summary["execution_peak_bytes"]
            <= manifest.device_physical_capacity_bytes
            and (
                not manifest.reject_overbudget
                or int(execution_statistics.peak_process_physical_bytes)
                <= manifest.device_physical_capacity_bytes
            )
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


def plan_result(plan: Plan, directory: Path) -> dict[str, Any]:
    import json

    recorded = json.loads((directory / "phases.json").read_text())
    phases = {key: recorded[key] for key in ("build", "profile", "plan")}
    summary = plan.summary
    return {
        "planning_seconds": sum(phases.values()),
        "phase_seconds": phases,
        "predicted_makespan_seconds": summary["duration_seconds"],
        "predicted_device_peak_bytes": summary["execution_peak_bytes"],
        "predicted_spill_peak_bytes": summary["spill_peak_bytes"],
        "transfer_bytes_evicted": summary["evict_bytes"],
        "transfer_bytes_fetched": summary["fetch_bytes"],
        "plan_artifact": _artifact_identity(directory / "plan.ssp"),
        "phase_artifacts": str(directory),
        "communication_info": plan.communication_info.to_dict(),
    }


def measured_result(
    manifest: FullModelManifest,
    arguments: argparse.Namespace,
    plan: Plan,
    warm: WarmStep,
    measured: Measurements,
    *,
    runtime: Runtime,
    logical_steps: int,
    physical_statuses: list[int],
    regression: RegressionComparison,
    directory: Path,
) -> dict[str, Any]:
    from dataclasses import asdict

    execution_statistics = adapter_statistics()
    runtime_delta = _runtime_delta(warm.execution_baseline, execution_statistics)
    gates = _gate_verdicts(
        manifest,
        arguments,
        plan,
        warm,
        measured,
        runtime=runtime,
        logical_steps=logical_steps,
        execution_statistics=execution_statistics,
        runtime_delta=runtime_delta,
        physical_statuses=physical_statuses,
        regression=regression,
    )
    return {
        "schema": artifact_schema("full_model_qualification"),
        "manifest": manifest.as_dict(),
        "plan_only": False,
        "passed": gates.passed,
        **plan_result(plan, directory),
        **asdict(measured),
        "groups": arguments.groups,
        "steps_per_group": arguments.steps_per_group,
        "protocol_complete": gates.protocol_complete,
        "checkpoint_seconds": warm.checkpoint_seconds,
        "checkpoint_skipped": arguments.skip_checkpoint,
        "checkpoint_restored": warm.checkpoint_restored,
        "restore_seconds": warm.restore_seconds,
        "warm_seconds": warm.warm_seconds,
        "warm_objectives": warm.warm_objectives,
        "warm_diagnostics": warm.warm_diagnostics,
        "objectives_finite": gates.objectives_finite,
        "logical_steps": logical_steps,
        "expected_logical_steps": gates.expected_logical_steps,
        "logical_steps_passed": gates.logical_steps_passed,
        "median_step_seconds": gates.median_step_seconds,
        "median_tokens_per_second": gates.median_throughput,
        "simulator_relative_error": gates.simulator_relative_error,
        "simulator_gate_passed": gates.simulator_passed,
        "regression_comparison": regression.as_dict(),
        "regression_gate_applicable": regression.applicable,
        "regression_throughput_ratio": gates.regression_ratio,
        "regression_gate_passed": gates.regression_passed,
        "predecessor_throughput_ratio": gates.predecessor_ratio,
        "physical_budget_statuses": physical_statuses,
        "physical_budget_passed": gates.physical_passed,
        "physical_budget_enforced": True,
        "peak_process_physical_bytes": int(
            execution_statistics.peak_process_physical_bytes
        ),
        "observed_external_high_water_bytes": int(
            execution_statistics.observed_external_high_water_bytes
        ),
        "strict_runtime_passed": gates.strict_runtime,
        "runtime_delta": runtime_delta,
        "runtime_statistics": statistics_dict(execution_statistics),
    }
