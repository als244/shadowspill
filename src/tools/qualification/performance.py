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

from shadowspill.memory import device, pinned_host
from shadowspill.pytorch import (
    Runtime,
    plan_step,
)
from tools.qualification.model_state import export_case_model, import_case_model
from tools.qualification.pressurefit_fixtures import write_pressurefit_fixtures
from tools.qualification.runtime_evidence import (
    adapter_statistics,
    check_physical_budget,
    statistics_dict,
)
from workloads.full_model import FullModelManifest, build_case, manifest_for
from workloads.providers import ModelImplementation

_MINIMUM_HISTORICAL_RATIO = 0.95
_MAXIMUM_SIMULATOR_ERROR = 0.05


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
            after.cuda.device_allocations - before.cuda.device_allocations
        ),
        "pinned_host_allocations": int(
            after.cuda.pinned_host_allocations - before.cuda.pinned_host_allocations
        ),
        "event_driver_creates": int(
            after.cuda.event_pool_driver_creates - before.cuda.event_pool_driver_creates
        ),
        "event_growth_rejections": int(
            after.cuda.event_pool_growth_rejections
            - before.cuda.event_pool_growth_rejections
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


def _run(arguments: argparse.Namespace) -> dict[str, object]:
    manifest = _manifest_with_overrides(
        arguments.family,
        arguments.implementation,
        spill_budget_gib=arguments.spill_budget_gib,
    )
    case = build_case(manifest, seed=arguments.seed)
    output = arguments.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    cache = (
        arguments.planning_cachedir.expanduser().resolve()
        if arguments.planning_cachedir is not None
        else output.parent / "planning_cache" / manifest.identity
    )
    with case.implementations():
        runtime = Runtime(
            pools={
                "execution": device(
                    physical_capacity=manifest.device_physical_capacity_bytes
                ),
                "spill": pinned_host(capacity=manifest.spill_budget_bytes),
            }
        )
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
            spill_budget=manifest.spill_budget_bytes,
            optimizer_ordering="stage_interleaved",
            verbose=True,
            planning_cachedir=cache,
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
                "schema": "shadowspill.full_model_qualification/v1",
                "manifest": manifest.as_dict(),
                "plan_only": True,
                "passed": not any(physical_statuses),
                "planning_seconds": planning_seconds,
                "phase_seconds": phases,
                "predicted_makespan_seconds": report.predicted_makespan_ns / 1e9,
                "predicted_device_peak_bytes": report.predicted_device_peak_bytes,
                "predicted_host_peak_bytes": report.predicted_host_peak_bytes,
                "transfer_bytes_evicted": report.transfer_bytes_evicted,
                "transfer_bytes_fetched": report.transfer_bytes_fetched,
                "plan_report_artifact": _artifact_identity(plan_path),
                "pressurefit_fixtures": fixtures,
                "physical_budget_statuses": physical_statuses,
            }
            training.close()
            export_case_model(case, runtime=runtime)
            runtime.close()
            return result

        checkpoint_started = time.perf_counter()
        checkpoint = training.state_dict()
        checkpoint_seconds = time.perf_counter() - checkpoint_started
        checkpoint_step = checkpoint["step"]
        warm_started = time.perf_counter()
        warm_result = training(case.microbatches, runtime_trace=True)
        if warm_result.diagnostics is None:
            raise AssertionError("full-model warm trace omitted diagnostics")
        warm_diagnostics = warm_result.diagnostics.result()
        warm_seconds = time.perf_counter() - warm_started
        warm_objectives = [float(value) for value in warm_result.objectives]
        physical_statuses.append(check_physical_budget())

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
                step_result = training(case.microbatches)
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
            physical_statuses.append(check_physical_budget())
            print(
                f"{manifest.identity} group {group + 1}: "
                f"{elapsed / arguments.steps_per_group:.6f}s/step, "
                f"{group_tokens_per_second[-1]:.2f} tokens/s",
                flush=True,
            )

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
        historical_ratio = (
            None
            if manifest.historical_tokens_per_second is None
            else median_throughput / manifest.historical_tokens_per_second
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
            and runtime_delta["pinned_host_allocations"] == 0
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
        historical_passed = bool(
            historical_ratio is None or historical_ratio >= _MINIMUM_HISTORICAL_RATIO
        )
        logical_steps_passed = bool(
            checkpoint_restored and training._step == protocol_steps
        )
        result = {
            "schema": "shadowspill.full_model_qualification/v1",
            "manifest": manifest.as_dict(),
            "plan_only": False,
            "passed": bool(
                protocol_complete
                and objectives_finite
                and logical_steps_passed
                and physical_passed
                and strict_runtime
                and simulator_passed
                and historical_passed
            ),
            "protocol_complete": protocol_complete,
            "groups": arguments.groups,
            "steps_per_group": arguments.steps_per_group,
            "planning_seconds": planning_seconds,
            "phase_seconds": phases,
            "checkpoint_seconds": checkpoint_seconds,
            "warm_seconds": warm_seconds,
            "restore_seconds": restore_seconds,
            "checkpoint_restored": checkpoint_restored,
            "warm_objectives": warm_objectives,
            "warm_diagnostics": warm_diagnostics.as_dict(),
            "measured_objectives": measured_objectives,
            "objectives_finite": objectives_finite,
            "logical_steps": training._step,
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
            "historical_throughput_ratio": historical_ratio,
            "historical_gate_passed": historical_passed,
            "predicted_device_peak_bytes": report.predicted_device_peak_bytes,
            "predicted_host_peak_bytes": report.predicted_host_peak_bytes,
            "transfer_bytes_evicted": report.transfer_bytes_evicted,
            "transfer_bytes_fetched": report.transfer_bytes_fetched,
            "physical_budget_statuses": physical_statuses,
            "physical_budget_passed": physical_passed,
            "strict_runtime_passed": strict_runtime,
            "runtime_delta": runtime_delta,
            "runtime_statistics": statistics_dict(execution_statistics),
            "plan_report_artifact": _artifact_identity(plan_path),
            "pressurefit_fixtures": fixtures,
        }
        training.close()
        export_case_model(case, runtime=runtime)
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
    parser.add_argument("--force-fresh", action="store_true")
    parser.add_argument("--planning-cachedir", type=Path)
    parser.add_argument("--implementation-revision")
    parser.add_argument(
        "--spill-budget-gib",
        type=int,
        help="override the manifest's pinned spill-pool capacity",
    )
    arguments = parser.parse_args()
    if arguments.groups <= 0 or arguments.steps_per_group <= 0:
        parser.error("groups and steps-per-group must be positive")
    if arguments.spill_budget_gib is not None and arguments.spill_budget_gib <= 0:
        parser.error("--spill-budget-gib must be positive")
    try:
        result = _run(arguments)
    except BaseException as error:
        notes = tuple(str(note) for note in getattr(error, "__notes__", ()))
        failure = {
            "schema": "shadowspill.full_model_qualification_failure/v1",
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
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"] and not arguments.plan_only:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
