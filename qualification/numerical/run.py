"""Fresh-process eager/planned numerical qualification orchestrator."""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

from shadowspill.pytorch import plan

from .cases import DEFAULT_DEVICE_BUDGETS, ModelImplementation, build_case
from .metrics import compare_states, cpu_state, state_digest

_HOST_BUDGET = 64 << 30
_LOSS_RELATIVE_TOLERANCE = 0.01
_LOSS_ABSOLUTE_TOLERANCE = 2e-5
_MINIMUM_COSINE = 0.999
_MAXIMUM_RELATIVE_L2 = 0.025
_MINIMUM_SIGN_AGREEMENT = 0.99


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


def _eager_worker(
    family: str, model_implementation: ModelImplementation, output: Path
) -> None:
    case = build_case(family, model_implementation=model_implementation)
    model = case.model.cuda()
    microbatches = _cuda_microbatches(case.microbatches)
    optimizer = case.optimizer(model.parameters())
    losses: list[list[float]] = []
    timings: list[float] = []
    with case.implementations():
        for _step in range(5):
            optimizer.zero_grad(set_to_none=True)
            started = time.perf_counter()
            step_losses: list[float] = []
            for microbatch in microbatches:
                loss = case.objective(model, *microbatch)
                loss.backward()
                step_losses.append(float(loss.detach()))
            optimizer.step()
            torch.cuda.current_stream().synchronize()
            timings.append(time.perf_counter() - started)
            losses.append(step_losses)
    artifact = {
        "family": family,
        "model_implementation": model_implementation,
        "losses": losses,
        "step_seconds": timings,
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
) -> None:
    case = build_case(family, model_implementation=model_implementation)
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
        losses: list[list[float]] = []
        timings: list[float] = []
        checkpoint: object | None = None
        expected_replay: list[list[float]] = []
        for step in range(5):
            started = time.perf_counter()
            step_result = training(case.microbatches)
            torch.cuda.current_stream().synchronize()
            timings.append(time.perf_counter() - started)
            values = [float(item) for item in step_result.objectives]
            losses.append(values)
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
        for _step in range(2):
            step_result = training(case.microbatches)
            torch.cuda.current_stream().synchronize()
            replay_losses.append([float(item) for item in step_result.objectives])
        final_state = training.state_dict()
        replay_digest = state_digest(final_state)
        report = training.plan_report
        training.close()

    reference = torch.load(reference_path, map_location="cpu", weights_only=True)
    if reference.get("family") != family or reference.get(
        "model_implementation"
    ) != model_implementation:
        raise RuntimeError(
            "eager reference identity differs from requested qualification: "
            f"expected={model_implementation}/{family}, "
            f"actual={reference.get('model_implementation')}/{reference.get('family')}"
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
                _LOSS_ABSOLUTE_TOLERANCE
                + _LOSS_RELATIVE_TOLERANCE * abs(expected)
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
    qualification_result = {
        "schema": "shadowspill.numerical_qualification/v2",
        "family": family,
        "model_implementation": model_implementation,
        "device_budget_bytes": device_budget,
        "tolerances": {
            "loss_rtol": _LOSS_RELATIVE_TOLERANCE,
            "loss_atol": _LOSS_ABSOLUTE_TOLERANCE,
            "minimum_cosine": _MINIMUM_COSINE,
            "maximum_relative_l2": _MAXIMUM_RELATIVE_L2,
            "minimum_sign_agreement": _MINIMUM_SIGN_AGREEMENT,
        },
        "planning_seconds": planning_seconds,
        "phase_seconds": {
            name: nanoseconds / 1e9 for name, nanoseconds in report.phase_timings_ns
        },
        "planned_step_seconds": timings,
        "eager_step_seconds": reference["step_seconds"],
        "planned_losses": losses,
        "eager_losses": reference["losses"],
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
        "profile_cache_hits": report.profile_cache_hits,
        "profile_cache_misses": report.profile_cache_misses,
        "recomputation_cache_hits": report.recomputation_cache_hits,
        "recomputation_cache_misses": report.recomputation_cache_misses,
    }
    qualification_result["passed"] = bool(
        not loss_failures
        and not metric_failures
        and not exact_failures
        and qualification_result["checkpoint_replay_bitwise"]
        and report.transfer_bytes_to_host > 0
        and report.transfer_bytes_to_device > 0
        and qualification_result["selected_recomputation"]
        and report.predicted_device_peak_bytes <= device_budget
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
) -> None:
    result_directory.mkdir(parents=True, exist_ok=True)
    prefix = f"{model_implementation}_{family}"
    reference = result_directory / f"{prefix}_eager.pt"
    result = result_directory / f"{prefix}.json"
    base = [sys.executable, "-m", "qualification.numerical.run"]
    environment = dict(os.environ)
    subprocess.run(
        [
            *base,
            "_eager",
            family,
            str(reference),
            "--model-implementation",
            model_implementation,
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
        ],
        check=True,
        env=environment,
    )
    print(result.read_text(), end="")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("run", "_eager", "_planned"))
    parser.add_argument("family", choices=tuple(DEFAULT_DEVICE_BUDGETS))
    parser.add_argument("paths", nargs="*")
    parser.add_argument("--device-budget", type=int)
    parser.add_argument(
        "--model-implementation",
        choices=("pytorch", "mlops"),
        default="pytorch",
        help="pure PyTorch is the formal numerical authority",
    )
    arguments = parser.parse_args()
    family = str(arguments.family)
    model_implementation = arguments.model_implementation
    if arguments.mode == "run":
        if len(arguments.paths) != 1:
            parser.error("run requires one result directory")
        _orchestrate(
            family,
            model_implementation,
            Path(arguments.paths[0]),
            arguments.device_budget or DEFAULT_DEVICE_BUDGETS[family],
        )
    elif arguments.mode == "_eager":
        if len(arguments.paths) != 1:
            parser.error("_eager requires one output path")
        _eager_worker(family, model_implementation, Path(arguments.paths[0]))
    else:
        if len(arguments.paths) != 3:
            parser.error("_planned requires reference, result, and device budget")
        _planned_worker(
            family,
            model_implementation,
            Path(arguments.paths[0]),
            Path(arguments.paths[1]),
            int(arguments.paths[2]),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
