"""One cell, phase by phase: calibrate, plan, warm, then measure."""

from __future__ import annotations

import argparse
import gc
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch

from shadowspill.memory import device, pinned_host, transfer_route
from shadowspill.pytorch import (
    Runtime,
    plan_step,
)
from tools.qualification.plan_record import write_plan_records
from tools.qualification.runtime_evidence import (
    adapter_statistics,
    check_physical_budget,
    measured_rate_clause,
)
from workloads.common.training import LEARNING_RATE, optimizer_state_init
from workloads.full_model import FullModelManifest

from .readings import (
    _calibration_suspect,
    _phase_seconds,
    _profile_metadata,
    _report_runtime_transfer_capabilities,
    _wait_idle,
)


@dataclass(frozen=True, slots=True)
class _PlannedCase:
    """What planning produced, and what it wrote about itself."""

    training: Any
    report: Any
    phase_seconds: dict[str, float]
    planning_seconds: float
    plan_path: Path
    plan_records: object


@dataclass(frozen=True, slots=True)
class _WarmStep:
    """The warm step and the checkpoint round trip taken around it."""

    checkpoint_step: int
    checkpoint_seconds: float
    checkpoint_restored: bool | None
    restore_seconds: float
    warm_seconds: float
    warm_objectives: list[float]
    warm_diagnostics: Any
    execution_baseline: Any


@dataclass(frozen=True, slots=True)
class _Measurements:
    """Every timed quantity the measured groups produced."""

    group_seconds: list[float]
    host_group_seconds: list[float]
    group_tokens_per_second: list[float]
    cycle_seconds: list[float]
    opening_delay_seconds: list[float]
    selected_spans: list[float]
    dispatch_seconds: list[float]
    prior_invocation_drain_seconds: list[float]
    measured_objectives: list[list[float]]


def _calibrated_runtime(
    manifest: FullModelManifest,
) -> tuple[Runtime, dict[str, object], int]:
    """Open the runtime and calibrate it, retrying a bimodal measurement.

    The runtime owns its physical capacities. Register and calibrate those
    capacities before anonymous workload state claims the host pages that
    will otherwise back the spill arena and alter sustained DMA bandwidth.

    Bidirectional-concurrent calibration is bimodal despite the runtime-first
    lifecycle (solo variance is expected and not the anomaly), and a degraded
    calibration steers planning toward a different, higher-traffic plan. The
    concurrent/solo ratio separates the two modes; a legitimately high solo at
    most triggers one benign extra probe. Persistently low results are
    recorded and planning proceeds against the final measurement.
    """

    runtime = Runtime(
        pools={
            "execution": device(
                physical_capacity=manifest.device_physical_capacity_bytes
            ),
            "spill": pinned_host(capacity=manifest.spill_budget_bytes),
        },
        routes={
            "fetch": transfer_route(source="spill", destination="execution"),
            "evict": transfer_route(source="execution", destination="spill"),
        },
    )
    calibration_attempts = 1
    while calibration_attempts < 4 and _calibration_suspect(runtime):
        print(
            "suspect bidirectional-concurrent calibration "
            f"(attempt {calibration_attempts}); recalibrating",
            flush=True,
        )
        runtime.calibrate_transfer_capabilities()
        calibration_attempts += 1
    return runtime, _report_runtime_transfer_capabilities(runtime), calibration_attempts


def _plan_case(
    case: Any,
    manifest: FullModelManifest,
    arguments: argparse.Namespace,
    *,
    runtime: Runtime,
    cache: Path,
    output: Path,
    planning_spill_budget: int,
) -> _PlannedCase:
    """Plan the step, then save the plan report and its records beside it."""

    planning_started = time.perf_counter()
    training = plan_step(
        case.model,
        objective=case.objective,
        optimizer=case.optimizer,
        optimizer_state_init=optimizer_state_init,
        hyperparams=("lr",),
        example_inputs=case.microbatches,
        runtime=runtime,
        execution="execution",
        spill="spill",
        spill_budget=planning_spill_budget,
        optimizer_ordering="stage_interleaved",
        verbose=True,
        artifact_store=cache,
        build_store=arguments.build_store,
        plan_store=arguments.plan_store,
        build_store_mode=arguments.build_store_mode,
        plan_store_mode=arguments.plan_store_mode,
        profiling_metadata=_profile_metadata(case.microbatches),
        export_bypass_key=arguments.export_bypass_key,
    )
    planning_seconds = time.perf_counter() - planning_started
    report = training.plan_report
    phases = _phase_seconds(report)
    plan_path = output.with_name(f"{output.stem}_plan_report.pt")
    torch.save(report, plan_path)
    fixtures = write_plan_records(
        results=report.search_results,
        directory=output.parent / f"{output.stem}_plan_records",
    )
    print(
        f"planned {manifest.identity}: total={planning_seconds:.3f}s "
        f"lowering={phases.get('capture_lowering', 0.0):.3f}s "
        f"compilation={phases.get('compiled_entrypoint_construction', 0.0):.3f}s "
        f"profiling={phases.get('unique_stage_warmup_profiling', 0.0):.3f}s "
        f"search={phases.get('search', 0.0):.3f}s",
        flush=True,
    )
    return _PlannedCase(
        training=training,
        report=report,
        phase_seconds=phases,
        planning_seconds=planning_seconds,
        plan_path=plan_path,
        plan_records=fixtures,
    )


def _warm_step(
    training: Any,
    case: Any,
    arguments: argparse.Namespace,
    physical_statuses: list[object],
) -> _WarmStep:
    """Take the warm step, and the checkpoint round trip around it."""

    checkpoint: dict[str, object] | None = None
    checkpoint_seconds = 0.0
    checkpoint_step = training._step
    if not arguments.skip_checkpoint:
        checkpoint_started = time.perf_counter()
        checkpoint = training.state_dict()
        checkpoint_seconds = time.perf_counter() - checkpoint_started
        checkpoint_step = cast(int, checkpoint["step"])
    warm_started = time.perf_counter()
    warm_result = training(
        case.microbatches,
        hyperparams={"lr": LEARNING_RATE},
        runtime_trace=True,
        profiler_annotations=arguments.profiler_annotations,
    )
    if warm_result.diagnostics is None:
        raise AssertionError("full-model warm trace omitted diagnostics")
    warm_diagnostics = warm_result.diagnostics.result()
    warm_seconds = time.perf_counter() - warm_started
    warm_objectives = [float(value) for value in warm_result.objectives]
    # The warm step's cycle would otherwise close at the first measured
    # step's origin and be counted with the group: close it here and
    # discard it, so every group closes exactly its own steps.
    training.mark_cycle_end()
    training.invocation_timings()
    physical_statuses.append(check_physical_budget())

    restore_seconds = 0.0
    checkpoint_restored: bool | None = None
    if checkpoint is not None:
        restore_started = time.perf_counter()
        training.load_state_dict(checkpoint)
        restore_seconds = time.perf_counter() - restore_started
        checkpoint_restored = training._step == checkpoint_step
    del checkpoint, warm_result
    gc.collect()
    _wait_idle(training)
    return _WarmStep(
        checkpoint_step=checkpoint_step,
        checkpoint_seconds=checkpoint_seconds,
        checkpoint_restored=checkpoint_restored,
        restore_seconds=restore_seconds,
        warm_seconds=warm_seconds,
        warm_objectives=warm_objectives,
        warm_diagnostics=warm_diagnostics,
        execution_baseline=adapter_statistics(),
    )


def _announce_prediction(
    manifest: FullModelManifest, report: Any, runtime: Any
) -> None:
    """State the plan's prediction, and the rates it was made against.

    It is said before the first group so the measured lines below can be read
    against it as they appear, and with its rates because that is what it is
    only as good as.
    """

    predicted_step_seconds = report.predicted_makespan_ns / 1e9
    planned = report.summary
    print(
        f"simulator predicts {manifest.identity}: "
        f"{predicted_step_seconds:.4f} s/step, "
        f"{manifest.tokens_per_step / predicted_step_seconds:.2f} tokens/s "
        f"(planned with: fetch "
        f"{planned.fetch_bandwidth_bytes_per_second / 1e9:.1f} GB/s, evict "
        f"{planned.evict_bandwidth_bytes_per_second / 1e9:.1f} GB/s"
        f"{measured_rate_clause(runtime)})"
        f"; unconstrained throughput "
        f"{planned.unconstrained_step_seconds:.4f} s/step, "
        f"{manifest.tokens_per_step / planned.unconstrained_step_seconds:.2f}"
        f" tokens/s",
        flush=True,
    )


def _measure_groups(
    training: Any,
    case: Any,
    manifest: FullModelManifest,
    arguments: argparse.Namespace,
    physical_statuses: list[object],
) -> _Measurements:
    """Run the timed groups, closing each group's cycles before it is read."""

    measured = _Measurements([], [], [], [], [], [], [], [], [])
    for group in range(arguments.groups):
        retained_results: list[Any] = []
        group_started = time.perf_counter()
        for step in range(arguments.steps_per_group):
            call_started = time.perf_counter()
            step_result = training(
                case.microbatches,
                hyperparams={"lr": LEARNING_RATE},
                profiler_annotations=arguments.profiler_annotations,
            )
            measured.dispatch_seconds.append(time.perf_counter() - call_started)
            measured.prior_invocation_drain_seconds.append(
                training._collect_prior_invocation_drain_seconds()
            )
            retained_results.append(step_result)
            print(
                f"{manifest.identity} group {group + 1}/{arguments.groups} "
                f"step {step + 1}/{arguments.steps_per_group} submitted",
                flush=True,
            )
        # The group's last cycle closes where a next step would begin,
        # recorded before the drain so the drain is not inside it.
        training.mark_cycle_end()
        _wait_idle(training)
        measured.host_group_seconds.append(time.perf_counter() - group_started)
        timings = training.invocation_timings()
        if len(timings) != arguments.steps_per_group:
            raise AssertionError(
                f"group {group + 1} closed {len(timings)} cycles for "
                f"{arguments.steps_per_group} steps"
            )
        measured.cycle_seconds.extend(item.cycle_seconds for item in timings)
        measured.opening_delay_seconds.extend(
            item.opening_delay_seconds for item in timings
        )
        measured.selected_spans.extend(item.selected_span_seconds for item in timings)
        elapsed = sum(item.cycle_seconds for item in timings)
        measured.group_seconds.append(elapsed)
        measured.group_tokens_per_second.append(
            manifest.tokens_per_step * arguments.steps_per_group / elapsed
        )
        for step_result in retained_results:
            measured.measured_objectives.append(
                [float(value) for value in step_result.objectives]
            )
        # StepResult tensors are caller-owned runtime outputs.  Only the
        # scalar qualification evidence is retained across groups.
        del step_result, retained_results
        physical_statuses.append(check_physical_budget())
        print(
            f"{manifest.identity} group {group + 1}: "
            f"{elapsed / arguments.steps_per_group:.6f}s/step, "
            f"{measured.group_tokens_per_second[-1]:.2f} tokens/s",
            flush=True,
        )
    return measured
