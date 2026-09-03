"""Run one full-model ShadowSpill-only throughput qualification cell."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import statistics
import time
import traceback
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import torch

from shadowspill.memory import device, pinned_host, transfer_route
from shadowspill.pytorch import (
    Runtime,
    plan_step,
)
from shadowspill.schema import artifact_schema
from tools.qualification.model_state import import_case_model, release_case_model
from tools.qualification.pressurefit_fixtures import write_pressurefit_fixtures
from tools.qualification.runtime_evidence import (
    adapter_statistics,
    check_physical_budget,
    statistics_dict,
)
from workloads.full_model import FullModelManifest, build_case, manifest_for
from workloads.providers import ModelImplementation

_MINIMUM_REGRESSION_RATIO = 0.95

#: The simulator prices the selected span and the terminal tail; the opening
#: restore is unmodeled but, since first-use ordering of the initial
#: placement batch (shadowspill.ir.schedule.first_use_initial_order), bounded
#: by the first task's own inputs rather than the whole initial set. On a
#: nominal calibration the error sits within a few percent and errs
#: pessimistic. The bound is 0.10 because the calibrated transfer bandwidths
#: the plan is priced against move run to run: a calibration that lands at
#: 23.7 GB/s instead of the usual 25.5 prices olmoe 6.3% slower than the
#: hardware then delivers (2026-09-02), which is the simulator being
#: pessimistic, not wrong. The remaining unmodeled terms are the
#: terminal-drain serialization, input staging, and profile fidelity.
_MAXIMUM_SIMULATOR_ERROR = 0.10


def _phase_seconds(report: Any) -> dict[str, float]:
    return {
        name: int(nanoseconds) / 1e9 for name, nanoseconds in report.phase_timings_ns
    }


def _profile_metadata(microbatches: tuple[tuple[object, ...], ...]) -> list[object]:
    result: list[object] = []
    for microbatch in microbatches:
        lengths = microbatch[2]
        if not isinstance(lengths, Sequence):
            raise TypeError("performance sequence lengths must be a sequence")
        result.append({"sequence_lengths": list(lengths)})
    return result


def _wait_idle(training: Any) -> None:
    """Drain terminal actions at a qualification measurement boundary."""

    training._executor._bridge.wait_idle()


def _runtime_delta(before: Any, after: Any) -> dict[str, int]:
    return {
        "device_allocations": int(
            after.backend.device_allocations - before.backend.device_allocations
        ),
        "pinned_host_registrations": int(
            after.backend.pinned_host_registrations
            - before.backend.pinned_host_registrations
        ),
        "event_driver_creates": int(
            after.runtime.event_lease_driver_creates
            - before.runtime.event_lease_driver_creates
        ),
        "event_growth_rejections": int(
            after.runtime.event_lease_growth_rejections
            - before.runtime.event_lease_growth_rejections
        ),
        "allocation_callbacks": int(
            after.allocation_callbacks - before.allocation_callbacks
        ),
        "free_callbacks": int(after.free_callbacks - before.free_callbacks),
        "fetch_transfers": int(
            after.runtime.fetch_transfers - before.runtime.fetch_transfers
        ),
        "evict_transfers": int(
            after.runtime.evict_transfers - before.runtime.evict_transfers
        ),
        "bytes_fetched": int(
            after.runtime.bytes_fetched - before.runtime.bytes_fetched
        ),
        "bytes_evicted": int(
            after.runtime.bytes_evicted - before.runtime.bytes_evicted
        ),
    }


def _artifact_identity(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _report_runtime_transfer_capabilities(runtime: Runtime) -> dict[str, object]:
    """Print and return the exact transfer measurements consumed by planning."""

    capabilities = runtime.transfer_capabilities
    print(
        "runtime transfer capabilities: "
        f"generation={capabilities.generation} digest={capabilities.digest}",
        flush=True,
    )
    for route_name, route in runtime.routes.items():
        profile = capabilities.route(route.source, route.destination)
        print(
            f"  {route_name} ({route.source}->{route.destination}): "
            f"effective={profile.bandwidth_bytes_per_second / 1e9:.3f} GB/s "
            f"concurrent={profile.concurrent_bandwidth_bytes_per_second / 1e9:.3f} "
            f"GB/s solo={profile.solo_bandwidth_bytes_per_second / 1e9:.3f} GB/s "
            f"latency={profile.latency_nanoseconds / 1e3:.3f} us "
            f"mode={profile.calibration_mode} "
            f"probe={profile.measured_copies}x"
            f"{profile.large_copy_bytes / (1 << 20):.0f} MiB",
            flush=True,
        )
    return capabilities.as_dict()


def _calibration_suspect(runtime: Runtime) -> bool:
    """Detect the degraded bidirectional-concurrent calibration mode."""

    capabilities = runtime.transfer_capabilities
    for route in runtime.routes.values():
        profile = capabilities.route(route.source, route.destination)
        solo = profile.solo_bandwidth_bytes_per_second
        concurrent = profile.concurrent_bandwidth_bytes_per_second
        if solo > 0 and concurrent < 0.65 * solo:
            return True
    return False


def _manifest_with_overrides(
    family: str,
    implementation: str,
    *,
    spill_budget_gib: int | None,
) -> FullModelManifest:
    """Resolve one immutable qualification manifest and CLI overrides."""

    manifest = manifest_for(family, cast(ModelImplementation, implementation))
    if spill_budget_gib is None:
        return manifest
    if spill_budget_gib <= 0:
        raise ValueError("spill-budget-gib must be positive")
    return replace(manifest, spill_budget_bytes=spill_budget_gib << 30)


def _planning_spill_budget(
    manifest: FullModelManifest,
    *,
    planning_spill_budget_gib: int | None,
) -> int:
    """Resolve a plan budget bounded by the runtime spill-pool capacity."""

    if planning_spill_budget_gib is None:
        return manifest.spill_budget_bytes
    if planning_spill_budget_gib <= 0:
        raise ValueError("planning-spill-budget-gib must be positive")
    budget = planning_spill_budget_gib << 30
    if budget > manifest.spill_budget_bytes:
        raise ValueError(
            "planning spill budget exceeds the configured runtime spill pool: "
            f"budget={budget}, capacity={manifest.spill_budget_bytes}"
        )
    return budget


def _run(arguments: argparse.Namespace) -> dict[str, object]:
    manifest = _manifest_with_overrides(
        arguments.family,
        arguments.implementation,
        spill_budget_gib=arguments.spill_budget_gib,
    )
    planning_spill_budget = _planning_spill_budget(
        manifest,
        planning_spill_budget_gib=arguments.planning_spill_budget_gib,
    )
    output = arguments.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    cache = (
        arguments.artifact_store_dir.expanduser().resolve()
        if arguments.artifact_store_dir is not None
        else output.parent / "artifact_store" / manifest.identity
    )
    # The runtime owns its physical capacities.  Register and calibrate those
    # capacities before anonymous workload state claims the host pages that
    # will otherwise back the spill arena and alter sustained DMA bandwidth.
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
    # Bidirectional-concurrent calibration is bimodal on this host
    # despite the runtime-first lifecycle (about 21 versus about 25
    # GB/s across runs; solo variance is expected and not the anomaly),
    # and a degraded calibration steers planning toward a different,
    # higher-traffic plan.  The concurrent/solo ratio separates the two
    # observed modes; a legitimately high solo at most triggers one
    # benign extra probe.  Persistently low results are recorded and
    # planning proceeds against the final measurement.
    calibration_attempts = 1
    while calibration_attempts < 4 and _calibration_suspect(runtime):
        print(
            "suspect bidirectional-concurrent calibration "
            f"(attempt {calibration_attempts}); recalibrating",
            flush=True,
        )
        runtime.calibrate_transfer_capabilities()
        calibration_attempts += 1
    runtime_transfer_capabilities = _report_runtime_transfer_capabilities(runtime)
    case = build_case(manifest, seed=arguments.seed)
    with case.implementations():
        case = import_case_model(case, runtime=runtime)
        model = case.model
        planning_started = time.perf_counter()
        training = plan_step(
            model,
            objective=case.objective,
            opt=case.optimizer,
            example_inputs=case.microbatches,
            runtime=runtime,
            execution="execution",
            spill="spill",
            spill_budget=planning_spill_budget,
            optimizer_ordering="stage_interleaved",
            verbose=True,
            artifact_store_dir=cache,
            profiling_metadata=_profile_metadata(case.microbatches),
            save_plan=True,
            force_fresh=arguments.force_fresh,
            overwrite_plan=arguments.force_fresh,
            implementation_revision=arguments.implementation_revision,
        )
        planning_seconds = time.perf_counter() - planning_started
        report = training.plan_report
        phases = _phase_seconds(report)
        plan_path = output.with_name(f"{output.stem}_plan_report.pt")
        torch.save(report, plan_path)
        fixtures = write_pressurefit_fixtures(
            results=report.pressurefit_results,
            directory=output.parent / f"{output.stem}_pressurefit",
        )
        print(
            f"planned {manifest.identity}: total={planning_seconds:.3f}s "
            f"lowering={phases.get('capture_lowering', 0.0):.3f}s "
            f"compilation={phases.get('compiled_entrypoint_construction', 0.0):.3f}s "
            f"profiling={phases.get('unique_stage_warmup_profiling', 0.0):.3f}s "
            f"pressurefit={phases.get('pressurefit_simulation', 0.0):.3f}s",
            flush=True,
        )

        physical_statuses = [check_physical_budget()]
        if arguments.plan_only:
            result: dict[str, object] = {
                "schema": artifact_schema("full_model_qualification"),
                "manifest": manifest.as_dict(),
                "plan_only": True,
                "passed": not any(physical_statuses),
                "planning_seconds": planning_seconds,
                "phase_seconds": phases,
                "predicted_makespan_seconds": report.predicted_makespan_ns / 1e9,
                "predicted_device_peak_bytes": report.predicted_device_peak_bytes,
                "predicted_spill_peak_bytes": report.predicted_spill_peak_bytes,
                "transfer_bytes_evicted": report.transfer_bytes_evicted,
                "transfer_bytes_fetched": report.transfer_bytes_fetched,
                "plan_report_artifact": _artifact_identity(plan_path),
                "pressurefit_fixtures": fixtures,
                "physical_budget_statuses": physical_statuses,
                "planning_spill_budget_bytes": planning_spill_budget,
                "runtime_transfer_capabilities": runtime_transfer_capabilities,
            }
            training.close()
            release_case_model(case, runtime=runtime)
            runtime.close()
            return result

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
            runtime_trace=True,
            profiler_annotations=arguments.profiler_annotations,
        )
        if warm_result.diagnostics is None:
            raise AssertionError("full-model warm trace omitted diagnostics")
        warm_diagnostics = warm_result.diagnostics.result()
        warm_seconds = time.perf_counter() - warm_started
        warm_objectives = [float(value) for value in warm_result.objectives]
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
        execution_baseline = adapter_statistics()

        group_seconds: list[float] = []
        group_tokens_per_second: list[float] = []
        selected_spans: list[float] = []
        dispatch_seconds: list[float] = []
        measured_objectives: list[list[float]] = []
        for group in range(arguments.groups):
            retained_results: list[Any] = []
            group_started = time.perf_counter()
            for step in range(arguments.steps_per_group):
                training._arm_selected_span_timing()
                call_started = time.perf_counter()
                step_result = training(
                    case.microbatches,
                    profiler_annotations=arguments.profiler_annotations,
                )
                dispatch_seconds.append(time.perf_counter() - call_started)
                selected_spans.append(training._collect_selected_span_seconds())
                retained_results.append(step_result)
                print(
                    f"{manifest.identity} group {group + 1}/{arguments.groups} "
                    f"step {step + 1}/{arguments.steps_per_group} submitted",
                    flush=True,
                )
            _wait_idle(training)
            elapsed = time.perf_counter() - group_started
            group_seconds.append(elapsed)
            group_tokens_per_second.append(
                manifest.tokens_per_step * arguments.steps_per_group / elapsed
            )
            for step_result in retained_results:
                measured_objectives.append(
                    [float(value) for value in step_result.objectives]
                )
            # StepResult tensors are caller-owned runtime outputs.  Only the
            # scalar qualification evidence is retained across groups.
            del step_result, retained_results
            physical_statuses.append(check_physical_budget())
            print(
                f"{manifest.identity} group {group + 1}: "
                f"{elapsed / arguments.steps_per_group:.6f}s/step, "
                f"{group_tokens_per_second[-1]:.2f} tokens/s",
                flush=True,
            )

        # Releasing the retained StepResult tensors above enqueues retirements
        # through the free callback, after the last per-group drain.  Sample
        # the gate evidence at a quiesced boundary so pending_retirements
        # reflects a leak rather than a race with the worker.
        _wait_idle(training)
        execution_statistics = adapter_statistics()
        runtime_delta = _runtime_delta(execution_baseline, execution_statistics)
        median_group_seconds = float(statistics.median(group_seconds))
        median_step_seconds = median_group_seconds / arguments.steps_per_group
        median_throughput = manifest.tokens_per_step / median_step_seconds
        predicted_seconds = report.predicted_makespan_ns / 1e9
        simulator_relative_error = (
            (median_step_seconds - predicted_seconds) / predicted_seconds
            if predicted_seconds > 0.0
            else math.inf
        )
        regression_ratio = (
            None
            if manifest.regression_tokens_per_second is None
            else median_throughput / manifest.regression_tokens_per_second
        )
        predecessor_ratio = (
            None
            if manifest.predecessor_tokens_per_second is None
            else median_throughput / manifest.predecessor_tokens_per_second
        )
        objectives_finite = all(
            math.isfinite(value)
            for values in (warm_objectives, *measured_objectives)
            for value in values
        )
        protocol_steps = arguments.groups * arguments.steps_per_group
        protocol_complete = arguments.groups == 3 and arguments.steps_per_group == 4
        strict_runtime = bool(
            runtime_delta["device_allocations"] == 0
            and runtime_delta["pinned_host_registrations"] == 0
            and runtime_delta["event_driver_creates"] == 0
            and runtime_delta["event_growth_rejections"] == 0
            and int(execution_statistics.callback_failures) == 0
            and int(execution_statistics.pointer_lookup_failures) == 0
            and int(execution_statistics.runtime.queued_actions) == 0
            and int(execution_statistics.runtime.pending_retirements) == 0
        )
        physical_passed = bool(
            not any(physical_statuses)
            and report.predicted_device_peak_bytes
            <= manifest.device_physical_capacity_bytes
            and int(execution_statistics.peak_process_physical_bytes)
            <= manifest.device_physical_capacity_bytes
            and int(execution_statistics.runtime.spill_peak_allocated_bytes)
            <= manifest.spill_budget_bytes
        )
        simulator_passed = abs(simulator_relative_error) <= _MAXIMUM_SIMULATOR_ERROR
        regression_passed = bool(
            regression_ratio is None or regression_ratio >= _MINIMUM_REGRESSION_RATIO
        )
        expected_logical_steps = protocol_steps + int(arguments.skip_checkpoint)
        logical_steps_passed = bool(
            (arguments.skip_checkpoint or checkpoint_restored)
            and training._step == expected_logical_steps
        )
        result = {
            "schema": artifact_schema("full_model_qualification"),
            "manifest": manifest.as_dict(),
            "plan_only": False,
            "passed": bool(
                protocol_complete
                and objectives_finite
                and logical_steps_passed
                and physical_passed
                and strict_runtime
                and simulator_passed
                and regression_passed
            ),
            "protocol_complete": protocol_complete,
            "groups": arguments.groups,
            "steps_per_group": arguments.steps_per_group,
            "planning_seconds": planning_seconds,
            "phase_seconds": phases,
            "checkpoint_seconds": checkpoint_seconds,
            "checkpoint_skipped": bool(arguments.skip_checkpoint),
            "warm_seconds": warm_seconds,
            "restore_seconds": restore_seconds,
            "checkpoint_restored": checkpoint_restored,
            "warm_objectives": warm_objectives,
            "warm_diagnostics": warm_diagnostics.as_dict(),
            "measured_objectives": measured_objectives,
            "objectives_finite": objectives_finite,
            "logical_steps": training._step,
            "expected_logical_steps": expected_logical_steps,
            "logical_steps_passed": logical_steps_passed,
            "group_seconds": group_seconds,
            "group_tokens_per_second": group_tokens_per_second,
            "median_step_seconds": median_step_seconds,
            "median_tokens_per_second": median_throughput,
            "selected_task_span_seconds": selected_spans,
            "median_selected_task_span_seconds": float(
                statistics.median(selected_spans)
            ),
            "dispatch_seconds": dispatch_seconds,
            "predicted_makespan_seconds": predicted_seconds,
            "simulator_relative_error": simulator_relative_error,
            "simulator_gate_passed": simulator_passed,
            "regression_throughput_ratio": regression_ratio,
            "regression_gate_passed": regression_passed,
            "predecessor_throughput_ratio": predecessor_ratio,
            "predicted_device_peak_bytes": report.predicted_device_peak_bytes,
            "predicted_spill_peak_bytes": report.predicted_spill_peak_bytes,
            "transfer_bytes_evicted": report.transfer_bytes_evicted,
            "transfer_bytes_fetched": report.transfer_bytes_fetched,
            "physical_budget_statuses": physical_statuses,
            "physical_budget_passed": physical_passed,
            "planning_spill_budget_bytes": planning_spill_budget,
            "calibration_attempts": calibration_attempts,
            "strict_runtime_passed": strict_runtime,
            "runtime_delta": runtime_delta,
            "runtime_statistics": statistics_dict(execution_statistics),
            "runtime_transfer_capabilities": runtime_transfer_capabilities,
            "plan_report_artifact": _artifact_identity(plan_path),
            "pressurefit_fixtures": fixtures,
        }
        training.close()
        # Qualification never reuses the model; an export copy would stack an
        # anonymous full-model allocation on the registered spill arena.
        release_case_model(case, runtime=runtime)
        runtime.close()
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("family", choices=("llama3", "qwen35", "olmoe"))
    parser.add_argument("implementation", choices=("pytorch", "mlops"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=20_260_811)
    parser.add_argument("--groups", type=int, default=3)
    parser.add_argument("--steps-per-group", type=int, default=4)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--measure-only",
        action="store_true",
        help=(
            "report throughput without judging it; the gates compare against "
            "floors measured on one machine, so they carry no meaning on "
            "another. The artifact still records every gate field"
        ),
    )
    parser.add_argument(
        "--skip-checkpoint",
        action="store_true",
        help=(
            "run the throughput protocol without the anonymous full-state "
            "checkpoint copy; this is a runtime probe, not checkpoint qualification"
        ),
    )
    parser.add_argument(
        "--profiler-annotations",
        action="store_true",
        help=(
            "emit profiler ranges around task boundaries and compiled calls, so an "
            "external profiler can attribute time to the task that spent it"
        ),
    )
    parser.add_argument("--force-fresh", action="store_true")
    parser.add_argument("--artifact-store-dir", type=Path)
    parser.add_argument("--implementation-revision")
    parser.add_argument(
        "--spill-budget-gib",
        type=int,
        help="override the manifest's runtime spill-pool capacity",
    )
    parser.add_argument(
        "--planning-spill-budget-gib",
        type=int,
        help="use a smaller planning budget within the runtime spill pool",
    )
    arguments = parser.parse_args()
    if arguments.groups <= 0 or arguments.steps_per_group <= 0:
        parser.error("groups and steps-per-group must be positive")
    if arguments.plan_only and arguments.skip_checkpoint:
        parser.error("--skip-checkpoint has no effect with --plan-only")
    if arguments.plan_only and arguments.measure_only:
        parser.error("--measure-only has nothing to measure with --plan-only")
    if arguments.spill_budget_gib is not None and arguments.spill_budget_gib <= 0:
        parser.error("--spill-budget-gib must be positive")
    if (
        arguments.planning_spill_budget_gib is not None
        and arguments.planning_spill_budget_gib <= 0
    ):
        parser.error("--planning-spill-budget-gib must be positive")
    try:
        result = _run(arguments)
    except BaseException as error:
        notes = tuple(str(note) for note in getattr(error, "__notes__", ()))
        failure = {
            "schema": artifact_schema("full_model_qualification_failure"),
            "family": arguments.family,
            "implementation": arguments.implementation,
            "error_type": type(error).__name__,
            "error": str(error),
            "error_notes": notes,
            "traceback": "".join(traceback.format_exception(error)),
        }
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.with_suffix(".failure.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n"
        )
        raise
    arguments.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    identity = f"{arguments.implementation}_{arguments.family}"
    if arguments.measure_only:
        print(f"RESULT MEASURED: {identity}", flush=True)
    else:
        print(
            f"RESULT {'PASS' if result['passed'] else 'FAIL'}: {identity}", flush=True
        )
    if not arguments.plan_only:
        gates = (
            ("protocol_complete", "PROTOCOL"),
            ("objectives_finite", "OBJECTIVES"),
            ("logical_steps_passed", "LOGICAL STEPS"),
            ("physical_budget_passed", "PHYSICAL BUDGETS"),
            ("strict_runtime_passed", "STRICT RUNTIME"),
            ("simulator_gate_passed", "SIMULATOR"),
            ("regression_gate_passed", "REGRESSION"),
        )
        if not arguments.measure_only:
            for key, label in gates:
                print(f"  GATE {label}: {'pass' if result[key] else 'FAIL'}")
        print(
            f"  MEDIAN STEP: {result['median_step_seconds']:.4f} seconds "
            f"({result['median_tokens_per_second']:.1f} tokens/s)"
        )
        print(
            f"  PREDICTED STEP: {result['predicted_makespan_seconds']:.4f} "
            f"seconds (simulator error {result['simulator_relative_error']:+.2%})"
        )
        # Both ratios divide by throughput measured on the machine that set
        # the floors, so on any other machine they describe the hardware
        # rather than this run.  Measure-only reports the measurement itself.
        if not arguments.measure_only:
            ratio = result["regression_throughput_ratio"]
            if isinstance(ratio, float):
                print(f"  REGRESSION RATIO: {ratio:.2%}")
            # Reported, never gated: how close this run is to the predecessor
            # system ShadowSpill replaces. See workloads.full_model for the
            # provenance.
            ratio = result["predecessor_throughput_ratio"]
            if isinstance(ratio, float):
                print(f"  PREDECESSOR RATIO: {ratio:.2%}")
    print(f"  PLANNING: {result['planning_seconds']:.3f} seconds")
    print(f"  ARTIFACT: {arguments.output}", flush=True)
    if not result["passed"] and not arguments.plan_only and not arguments.measure_only:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
