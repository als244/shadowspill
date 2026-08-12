"""Fresh-process compiled-reference/planned numerical qualification."""

from __future__ import annotations

import argparse
import copy
import ctypes
import hashlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from shadowspill.pytorch import plan
from shadowspill.pytorch._abi import AdapterStatistics
from shadowspill.pytorch._allocator import installed_allocator

from .cases import DEFAULT_DEVICE_BUDGETS, ModelImplementation, build_case
from .fixtures import write_pressurefit_fixtures
from .metrics import compare_states, cpu_state, state_digest

_HOST_BUDGET = 64 << 30
_LOSS_RELATIVE_TOLERANCE = 0.01
_LOSS_ABSOLUTE_TOLERANCE = 2e-5
_MINIMUM_COSINE = 0.999
_MAXIMUM_RELATIVE_L2 = 0.025
_MINIMUM_SIGN_AGREEMENT = 0.99
_REFERENCE_EXECUTION = "torch.compile.inductor.fullgraph"


def _meets_tensor_tolerance(metric: Any) -> bool:
    return bool(
        metric.cosine >= _MINIMUM_COSINE
        and metric.relative_l2 <= _MAXIMUM_RELATIVE_L2
        and metric.sign_agreement >= _MINIMUM_SIGN_AGREEMENT
    )


def _optimizer_steps(checkpoint: object) -> dict[str, int]:
    if not isinstance(checkpoint, dict):
        return {}
    optimizer = checkpoint.get("optimizer")
    if not isinstance(optimizer, dict):
        return {}
    state = optimizer.get("state")
    if not isinstance(state, dict):
        return {}
    result: dict[str, int] = {}
    for parameter_id, values in state.items():
        if not isinstance(values, dict):
            continue
        step = values.get("step")
        if isinstance(step, torch.Tensor) and step.numel() == 1:
            result[str(parameter_id)] = int(step)
    return result


def _cuda_microbatches(values: list[list[Any]]) -> list[list[Any]]:
    return [
        [item.cuda() if isinstance(item, torch.Tensor) else item for item in microbatch]
        for microbatch in values
    ]


def _json_argument(value: str, *, description: str) -> Any:
    source = value
    if value.startswith("@"):
        path = Path(value[1:]).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"{description} file does not exist: {path}")
        source = path.read_text()
    try:
        return json.loads(source)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid {description} JSON: {exc}") from exc


def _case_options(values: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for value in values:
        name, separator, encoded = value.partition("=")
        if separator == "" or not name:
            raise ValueError("case options must use NAME=JSON")
        result[name] = _json_argument(encoded, description=f"case option {name!r}")
    return result


def _case_identity(
    *,
    model_name: str,
    model_implementation: ModelImplementation,
    seed: int,
    model_config: dict[str, Any],
    data_geometry: list[dict[str, Any]] | None,
    case_factory: str | None,
    case_options: dict[str, Any],
) -> str:
    payload = {
        "reference_execution": _REFERENCE_EXECUTION,
        "model_name": model_name,
        "model_implementation": model_implementation,
        "seed": seed,
        "model_config": model_config,
        "data_geometry": data_geometry,
        "case_factory": case_factory,
        "case_options": case_options,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _adapter_statistics() -> AdapterStatistics:
    installed = installed_allocator()
    if installed is None:
        raise RuntimeError("ShadowSpill allocator is not installed")
    result = AdapterStatistics()
    status = int(
        installed.library.shadowspill_pytorch_allocator_statistics(ctypes.byref(result))
    )
    if status != 0:
        raise RuntimeError(f"allocator statistics failed with status {status}")
    return result


def _check_physical_budget() -> int:
    installed = installed_allocator()
    if installed is None:
        raise RuntimeError("ShadowSpill allocator is not installed")
    return int(installed.library.shadowspill_pytorch_check_physical_budget())


def _planning_breakdown(
    phase_seconds: dict[str, float], *, planning_seconds: float
) -> dict[str, float]:
    """Return non-overlapping public planning phases for matrix comparisons."""

    lowering_aot = phase_seconds.get("capture_lowering", 0.0)
    profiling = phase_seconds.get("structural_profiling", 0.0)
    compilation = phase_seconds.get("compilation", 0.0)
    program_lowering = phase_seconds.get("program_lowering", 0.0)
    pressurefit = phase_seconds.get("pressurefit_simulation", 0.0)
    admission = phase_seconds.get("host_admission", 0.0) + phase_seconds.get(
        "slab_admission", 0.0
    )
    classified = (
        lowering_aot
        + profiling
        + compilation
        + program_lowering
        + pressurefit
        + admission
    )
    return {
        "lowering_aot": lowering_aot,
        "profiling": profiling,
        "compiled_entrypoint_finalization": compilation,
        "canonical_program_lowering": program_lowering,
        "pressurefit": pressurefit,
        "physical_admission": admission,
        "other": max(0.0, planning_seconds - classified),
        "total": planning_seconds,
    }


def _reference_worker(
    family: str,
    model_implementation: ModelImplementation,
    output: Path,
    *,
    seed: int,
    model_config: dict[str, Any],
    data_geometry: list[dict[str, Any]] | None,
    case_factory: str | None,
    case_options: dict[str, Any],
) -> None:
    case = build_case(
        family,
        model_implementation=model_implementation,
        seed=seed,
        model_config=model_config,
        data_geometry=data_geometry,
        case_factory=case_factory,
        case_options=case_options,
    )
    model = case.model.cuda()
    microbatches = _cuda_microbatches(case.microbatches)
    optimizer = case.optimizer(model.parameters())

    def reference_objective(*microbatch: Any) -> torch.Tensor:
        return case.objective(model, *microbatch)

    compiled_objective: Callable[..., torch.Tensor] = torch.compile(
        reference_objective,
        fullgraph=True,
        dynamic=False,
    )
    losses: list[list[float]] = []
    timings: list[float] = []
    compute_timings: list[float] = []
    with case.implementations():
        for step in range(5):
            optimizer.zero_grad(set_to_none=True)
            compute_start = torch.cuda.Event(enable_timing=True)
            compute_end = torch.cuda.Event(enable_timing=True)
            started = time.perf_counter()
            compute_start.record(torch.cuda.current_stream())
            step_losses: list[float] = []
            for microbatch in microbatches:
                loss = compiled_objective(*microbatch)
                loss.backward()
                step_losses.append(float(loss.detach()))
            optimizer.step()
            compute_end.record(torch.cuda.current_stream())
            torch.cuda.current_stream().synchronize()
            elapsed = time.perf_counter() - started
            timings.append(elapsed)
            compute_timings.append(float(compute_start.elapsed_time(compute_end)) / 1e3)
            losses.append(step_losses)
            print(
                f"reference {model_implementation}/{family} "
                f"step {step + 1}/5: {elapsed:.3f}s",
                flush=True,
            )
    artifact = {
        "schema": "shadowspill.compiled_reference/v1",
        "reference_execution": _REFERENCE_EXECUTION,
        "family": family,
        "model_implementation": model_implementation,
        "case_identity": _case_identity(
            model_name=family,
            model_implementation=model_implementation,
            seed=seed,
            model_config=model_config,
            data_geometry=data_geometry,
            case_factory=case_factory,
            case_options=case_options,
        ),
        "losses": losses,
        "step_seconds": timings,
        "compute_step_seconds": compute_timings,
        "model": cpu_state(model.state_dict()),
        "optimizer": cpu_state(optimizer.state_dict()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output)


def _planned_worker(
    family: str,
    model_implementation: ModelImplementation,
    reference_path: Path,
    result_path: Path,
    device_budget: int,
    *,
    seed: int,
    model_config: dict[str, Any],
    data_geometry: list[dict[str, Any]] | None,
    case_factory: str | None,
    case_options: dict[str, Any],
) -> None:
    identity = _case_identity(
        model_name=family,
        model_implementation=model_implementation,
        seed=seed,
        model_config=model_config,
        data_geometry=data_geometry,
        case_factory=case_factory,
        case_options=case_options,
    )
    case = build_case(
        family,
        model_implementation=model_implementation,
        seed=seed,
        model_config=model_config,
        data_geometry=data_geometry,
        case_factory=case_factory,
        case_options=case_options,
    )
    with case.implementations():
        planning_started = time.perf_counter()
        training = plan(
            case.model,
            objective=case.objective,
            opt=case.optimizer,
            example_inputs=case.microbatches,
            device_budget=device_budget,
            host_budget=_HOST_BUDGET,
        )
        planning_seconds = time.perf_counter() - planning_started
        planning_phases = {
            name: nanoseconds / 1e9
            for name, nanoseconds in training.plan_report.phase_timings_ns
        }
        print(
            f"planned {model_implementation}/{family}: "
            f"total={planning_seconds:.3f}s, "
            f"lowering_aot={planning_phases.get('capture_lowering', 0.0):.3f}s, "
            f"profiling={planning_phases.get('structural_profiling', 0.0):.3f}s, "
            "pressurefit="
            f"{planning_phases.get('pressurefit_simulation', 0.0):.3f}s",
            flush=True,
        )
        physical_statuses = [_check_physical_budget()]
        execution_baseline = _adapter_statistics()
        losses: list[list[float]] = []
        timings: list[float] = []
        compute_timings: list[float] = []
        checkpoint: object | None = None
        expected_replay: list[list[float]] = []
        for step in range(5):
            training._arm_compute_timing()
            started = time.perf_counter()
            step_result = training(case.microbatches)
            torch.cuda.current_stream().synchronize()
            physical_statuses.append(_check_physical_budget())
            timings.append(time.perf_counter() - started)
            compute_timings.append(training._collect_compute_seconds())
            values = [float(item) for item in step_result.objectives]
            losses.append(values)
            print(
                f"shadowspill {model_implementation}/{family} "
                f"step {step + 1}/5: {timings[-1]:.3f}s",
                flush=True,
            )
            if step == 2:
                checkpoint = copy.deepcopy(training.state_dict())
            elif step > 2:
                expected_replay.append(values)
        uninterrupted_state = training.state_dict()
        uninterrupted_digest = state_digest(uninterrupted_state)
        if checkpoint is None:
            raise AssertionError("step-three checkpoint was not captured")
        training.load_state_dict(checkpoint)
        replay_losses: list[list[float]] = []
        for replay_step in range(2):
            replay_started = time.perf_counter()
            step_result = training(case.microbatches)
            torch.cuda.current_stream().synchronize()
            physical_statuses.append(_check_physical_budget())
            replay_losses.append([float(item) for item in step_result.objectives])
            print(
                f"shadowspill {model_implementation}/{family} replay "
                f"{replay_step + 1}/2: "
                f"{time.perf_counter() - replay_started:.3f}s",
                flush=True,
            )
        final_state = training.state_dict()
        replay_digest = state_digest(final_state)
        report = training.plan_report
        pressurefit_fixtures = write_pressurefit_fixtures(
            results=report.pressurefit_results,
            directory=result_path.parent / f"{result_path.stem}_pressurefit",
        )
        runtime_statistics = _adapter_statistics()
        training.close()

    reference = torch.load(reference_path, map_location="cpu", weights_only=True)
    if (
        reference.get("schema") != "shadowspill.compiled_reference/v1"
        or reference.get("reference_execution") != _REFERENCE_EXECUTION
        or reference.get("family") != family
        or reference.get("model_implementation") != model_implementation
        or reference.get("case_identity") != identity
    ):
        raise RuntimeError(
            "compiled reference identity differs from requested qualification; "
            "regenerate it without --reuse-reference"
        )
    tensor_results, exact_failures = compare_states(
        {"model": reference["model"], "optimizer": reference["optimizer"]},
        {"model": final_state["model"], "optimizer": final_state["optimizer"]},
    )
    loss_failures: list[str] = []
    worst_loss_relative = 0.0
    for step, (expected_step, actual_step) in enumerate(
        zip(reference["losses"], losses, strict=True), start=1
    ):
        for microbatch, (expected, actual) in enumerate(
            zip(expected_step, actual_step, strict=True), start=1
        ):
            relative = abs(actual - expected) / max(abs(expected), 1e-30)
            worst_loss_relative = max(worst_loss_relative, relative)
            if abs(actual - expected) > (
                _LOSS_ABSOLUTE_TOLERANCE + _LOSS_RELATIVE_TOLERANCE * abs(expected)
            ):
                loss_failures.append(
                    f"step {step} microbatch {microbatch}: "
                    f"expected={expected}, actual={actual}"
                )
    metric_failures = [
        name
        for name, metric in tensor_results.items()
        if not _meets_tensor_tolerance(metric)
    ]
    selections = tuple(
        (item.group_id, item.option_id) for item in report.execution_plan.selections
    )
    phase_seconds = {
        name: nanoseconds / 1e9 for name, nanoseconds in report.phase_timings_ns
    }
    qualification_result = {
        "schema": "shadowspill.numerical_qualification/v4",
        "reference_execution": _REFERENCE_EXECUTION,
        "family": family,
        "model_implementation": model_implementation,
        "case_identity": identity,
        "case_request": {
            "model_name": family,
            "model_implementation": model_implementation,
            "seed": seed,
            "model_config": model_config,
            "data_geometry": data_geometry,
            "case_factory": case_factory,
            "case_options": case_options,
        },
        "device_budget_bytes": device_budget,
        "tolerances": {
            "loss_rtol": _LOSS_RELATIVE_TOLERANCE,
            "loss_atol": _LOSS_ABSOLUTE_TOLERANCE,
            "minimum_cosine": _MINIMUM_COSINE,
            "maximum_relative_l2": _MAXIMUM_RELATIVE_L2,
            "minimum_sign_agreement": _MINIMUM_SIGN_AGREEMENT,
        },
        "planning_seconds": planning_seconds,
        "phase_seconds": phase_seconds,
        "planning_breakdown_seconds": _planning_breakdown(
            phase_seconds, planning_seconds=planning_seconds
        ),
        "pressurefit_seconds": phase_seconds.get("pressurefit_simulation", 0.0),
        "planned_step_seconds": timings,
        "planned_compute_seconds": compute_timings,
        "reference_step_seconds": reference["step_seconds"],
        "reference_compute_seconds": reference.get("compute_step_seconds", []),
        "planned_losses": losses,
        "reference_losses": reference["losses"],
        "worst_loss_relative": worst_loss_relative,
        "loss_failures": loss_failures,
        "minimum_cosine": min(
            (item.cosine for item in tensor_results.values()), default=1.0
        ),
        "maximum_relative_l2": max(
            (item.relative_l2 for item in tensor_results.values()), default=0.0
        ),
        "minimum_sign_agreement": min(
            (item.sign_agreement for item in tensor_results.values()), default=1.0
        ),
        "metric_failure_keys": metric_failures,
        "exact_failures": exact_failures,
        "checkpoint_replay_bitwise": (
            uninterrupted_digest == replay_digest and expected_replay == replay_losses
        ),
        "checkpoint_steps": _optimizer_steps(checkpoint),
        "uninterrupted_steps": _optimizer_steps(uninterrupted_state),
        "replay_steps": _optimizer_steps(final_state),
        "transfer_bytes_to_host": report.transfer_bytes_to_host,
        "transfer_bytes_to_device": report.transfer_bytes_to_device,
        "selected_recomputation": any(
            option != "save" for _group, option in selections
        ),
        "selection_count": len(selections),
        "task_count": len(
            report.execution_plan.program.selected_tasks(
                report.execution_plan.selections
            )
        ),
        "action_count": len(report.transfer_actions),
        "predicted_makespan_seconds": report.predicted_makespan_ns / 1e9,
        "predicted_device_peak_bytes": report.predicted_device_peak_bytes,
        "predicted_host_peak_bytes": report.predicted_host_peak_bytes,
        "predicted_fragmentation_bytes": (
            report.execution_plan.admission.predicted_fragmentation_bytes
        ),
        "fixed_slab_bytes": report.fixed_slab_bytes,
        "physical_budget_statuses": physical_statuses,
        "physical_budget_sealed": bool(runtime_statistics.physical_budget_sealed),
        "peak_process_physical_bytes": int(
            runtime_statistics.peak_process_physical_bytes
        ),
        "observed_external_high_water_bytes": int(
            runtime_statistics.observed_external_high_water_bytes
        ),
        "slab_bytes": int(runtime_statistics.runtime.slab_bytes),
        "slab_peak_allocated_bytes": int(
            runtime_statistics.runtime.peak_allocated_bytes
        ),
        "host_arena_bytes": int(runtime_statistics.runtime.host_arena_bytes),
        "host_peak_allocated_bytes": int(
            runtime_statistics.runtime.host_peak_allocated_bytes
        ),
        "callback_failures": int(runtime_statistics.callback_failures),
        "pointer_lookup_failures": int(runtime_statistics.pointer_lookup_failures),
        "allocation_event_overflow": bool(
            runtime_statistics.runtime.allocation_event_overflow
        ),
        "cuda_device_allocations": int(runtime_statistics.cuda.device_allocations),
        "steady_state_cuda_device_allocations": int(
            runtime_statistics.cuda.device_allocations
            - execution_baseline.cuda.device_allocations
        ),
        "steady_state_pinned_host_allocations": int(
            runtime_statistics.cuda.pinned_host_allocations
            - execution_baseline.cuda.pinned_host_allocations
        ),
        "event_pool_capacity": int(runtime_statistics.cuda.event_pool_capacity),
        "event_pool_peak_in_use": int(runtime_statistics.cuda.event_pool_peak_in_use),
        "event_pool_driver_creates": int(
            runtime_statistics.cuda.event_pool_driver_creates
        ),
        "steady_state_event_pool_driver_creates": int(
            runtime_statistics.cuda.event_pool_driver_creates
            - execution_baseline.cuda.event_pool_driver_creates
        ),
        "event_pool_growth_rejections": int(
            runtime_statistics.cuda.event_pool_growth_rejections
        ),
        "event_pool_sealed": bool(runtime_statistics.cuda.event_pool_sealed),
        "profile_cache_hits": report.profile_cache_hits,
        "profile_cache_misses": report.profile_cache_misses,
        "profile_unique_keys": report.profile_unique_keys,
        "captured_stage_count": report.captured_stage_count,
        "aot_unique_stage_abis": report.aot_unique_stage_abis,
        "aot_graph_pair_cache_hits": report.aot_graph_pair_cache_hits,
        "aot_graph_pair_cache_misses": report.aot_graph_pair_cache_misses,
        "recomputation_cache_hits": report.recomputation_cache_hits,
        "recomputation_cache_misses": report.recomputation_cache_misses,
        "cold_cache_requested": os.environ.get("SHADOWSPILL_QUALIFICATION_COLD") == "1",
        "pressurefit_fixtures": pressurefit_fixtures,
        "reference_state_digest": state_digest(
            {"model": reference["model"], "optimizer": reference["optimizer"]}
        ),
        "planned_state_digest": state_digest(
            {"model": final_state["model"], "optimizer": final_state["optimizer"]}
        ),
    }
    qualification_result["reference_bitwise_equal"] = bool(
        qualification_result["reference_state_digest"]
        == qualification_result["planned_state_digest"]
    )
    qualification_result["cold_cache_confirmed"] = bool(
        not qualification_result["cold_cache_requested"]
        or (
            qualification_result["profile_cache_hits"] == 0
            and qualification_result["recomputation_cache_hits"] == 0
        )
    )
    qualification_result["passed"] = bool(
        not loss_failures
        and not metric_failures
        and not exact_failures
        and qualification_result["checkpoint_replay_bitwise"]
        and report.transfer_bytes_to_host > 0
        and report.transfer_bytes_to_device > 0
        and qualification_result["selected_recomputation"]
        and report.predicted_device_peak_bytes <= device_budget
        and not any(physical_statuses)
        and qualification_result["physical_budget_sealed"]
        and qualification_result["peak_process_physical_bytes"] <= device_budget
        and qualification_result["slab_peak_allocated_bytes"]
        <= qualification_result["slab_bytes"]
        and qualification_result["host_peak_allocated_bytes"]
        <= qualification_result["host_arena_bytes"]
        <= _HOST_BUDGET
        and qualification_result["callback_failures"] == 0
        and qualification_result["pointer_lookup_failures"] == 0
        and not qualification_result["allocation_event_overflow"]
        and qualification_result["cuda_device_allocations"] == 1
        and qualification_result["steady_state_cuda_device_allocations"] == 0
        and qualification_result["steady_state_pinned_host_allocations"] == 0
        and qualification_result["steady_state_event_pool_driver_creates"] == 0
        and qualification_result["event_pool_growth_rejections"] == 0
        and qualification_result["event_pool_sealed"]
        and qualification_result["cold_cache_confirmed"]
    )
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(qualification_result, indent=2, sort_keys=True) + "\n"
    )
    if not qualification_result["passed"]:
        raise AssertionError(
            f"{model_implementation} {family} numerical qualification failed: "
            f"{qualification_result}"
        )


def _orchestrate(
    family: str,
    model_implementation: ModelImplementation,
    result_directory: Path,
    device_budget: int,
    *,
    seed: int,
    model_config_argument: str,
    data_geometry_argument: str | None,
    case_factory: str | None,
    case_option_arguments: list[str],
) -> None:
    result_directory.mkdir(parents=True, exist_ok=True)
    prefix = f"{model_implementation}_{family}"
    reference = result_directory / f"{prefix}_reference.pt"
    result = result_directory / f"{prefix}.json"
    base = [sys.executable, "-m", "qualification.numerical.run"]
    options = ["--seed", str(seed), "--model-config", model_config_argument]
    if data_geometry_argument is not None:
        options.extend(("--data-geometry", data_geometry_argument))
    if case_factory is not None:
        options.extend(("--case-factory", case_factory))
    for value in case_option_arguments:
        options.extend(("--case-option", value))
    environment = dict(os.environ)
    subprocess.run(
        [
            *base,
            "_reference",
            family,
            str(reference),
            "--model-implementation",
            model_implementation,
            *options,
        ],
        check=True,
        env=environment,
    )
    subprocess.run(
        [
            *base,
            "_planned",
            family,
            str(reference),
            str(result),
            str(device_budget),
            "--model-implementation",
            model_implementation,
            *options,
        ],
        check=True,
        env=environment,
    )
    print(result.read_text(), end="")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("run", "_reference", "_planned"))
    parser.add_argument("family", help="built-in family or custom model name")
    parser.add_argument("paths", nargs="*")
    parser.add_argument("--device-budget", type=int)
    parser.add_argument(
        "--model-implementation",
        choices=("pytorch", "mlops"),
        default="pytorch",
        help="pure PyTorch is the formal numerical authority",
    )
    parser.add_argument("--seed", type=int, default=20_260_811)
    parser.add_argument(
        "--model-config",
        default="{}",
        metavar="JSON|@FILE",
        help="built-in dataclass field overrides or custom-factory configuration",
    )
    parser.add_argument(
        "--data-geometry",
        metavar="JSON|@FILE",
        help="microbatch geometry list; omitted uses the built-in two-shape gate",
    )
    parser.add_argument(
        "--case-factory",
        metavar="MODULE:FUNCTION",
        help="factory for a model not in the built-in verification registry",
    )
    parser.add_argument(
        "--case-option",
        action="append",
        default=[],
        metavar="NAME=JSON",
        help="repeatable custom-factory option",
    )
    arguments = parser.parse_args()
    family = str(arguments.family)
    model_implementation = arguments.model_implementation
    try:
        model_config_value = _json_argument(
            arguments.model_config, description="model config"
        )
        if not isinstance(model_config_value, dict):
            raise ValueError("model config must decode to an object")
        data_geometry_value = None
        if arguments.data_geometry is not None:
            decoded_geometry = _json_argument(
                arguments.data_geometry, description="data geometry"
            )
            if not isinstance(decoded_geometry, list) or not all(
                isinstance(item, dict) for item in decoded_geometry
            ):
                raise ValueError("data geometry must decode to a list of objects")
            data_geometry_value = decoded_geometry
        case_options_value = _case_options(arguments.case_option)
    except ValueError as exc:
        parser.error(str(exc))
    if family not in DEFAULT_DEVICE_BUDGETS and arguments.case_factory is None:
        parser.error("unknown model name requires --case-factory MODULE:FUNCTION")
    if arguments.mode == "run":
        if len(arguments.paths) != 1:
            parser.error("run requires one result directory")
        if arguments.device_budget is None and family not in DEFAULT_DEVICE_BUDGETS:
            parser.error("custom model run requires --device-budget")
        _orchestrate(
            family,
            model_implementation,
            Path(arguments.paths[0]),
            arguments.device_budget or DEFAULT_DEVICE_BUDGETS.get(family, 0),
            seed=arguments.seed,
            model_config_argument=arguments.model_config,
            data_geometry_argument=arguments.data_geometry,
            case_factory=arguments.case_factory,
            case_option_arguments=arguments.case_option,
        )
    elif arguments.mode == "_reference":
        if len(arguments.paths) != 1:
            parser.error("_reference requires one output path")
        _reference_worker(
            family,
            model_implementation,
            Path(arguments.paths[0]),
            seed=arguments.seed,
            model_config=model_config_value,
            data_geometry=data_geometry_value,
            case_factory=arguments.case_factory,
            case_options=case_options_value,
        )
    else:
        if len(arguments.paths) != 3:
            parser.error("_planned requires reference, result, and device budget")
        _planned_worker(
            family,
            model_implementation,
            Path(arguments.paths[0]),
            Path(arguments.paths[1]),
            int(arguments.paths[2]),
            seed=arguments.seed,
            model_config=model_config_value,
            data_geometry=data_geometry_value,
            case_factory=arguments.case_factory,
            case_options=case_options_value,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
