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
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

import torch

from qualification.model_state import externalize_case_model, relocate_case_model
from shadowspill.ir import RecomputationGroup, RecomputationSelection
from shadowspill.memory import device, pinned_host
from shadowspill.pytorch import (
    Runtime,
    plan_step,
)
from shadowspill.pytorch.runtime_adapter.abi import AdapterStatistics
from shadowspill.pytorch.runtime_adapter.allocator import installed_allocator

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


def _recomputation_savings_bytes(
    groups: Sequence[RecomputationGroup],
    selections: Sequence[RecomputationSelection],
    alias_sizes: Mapping[str, int],
) -> tuple[int, int]:
    """Report maximum available and selected retained-byte savings."""

    selected_by_group = {item.group_id: item.option_id for item in selections}
    available = 0
    selected = 0
    for group in groups:
        reference = next(
            (item for item in group.options if item.option_id == "save"), None
        )
        if reference is None:
            continue
        reference_bytes = sum(
            alias_sizes[alias_id]
            for alias_id in set(reference.retained_alias_group_ids)
        )
        savings = {
            item.option_id: max(
                0,
                reference_bytes
                - sum(
                    alias_sizes[alias_id]
                    for alias_id in set(item.retained_alias_group_ids)
                ),
            )
            for item in group.options
        }
        available += max(savings.values(), default=0)
        selected += savings.get(selected_by_group.get(group.group_id, "save"), 0)
    return available, selected


def _transfer_pressure_gate_passed(
    *, required: bool, evicted_bytes: int, fetched_bytes: int
) -> bool:
    """Require real bidirectional movement, never a planner policy choice."""

    return bool(not required or (evicted_bytes > 0 and fetched_bytes > 0))


def _state_tensor_at_path(state: object, path: str) -> torch.Tensor:
    """Resolve one compare_states() tensor path for failure diagnostics."""

    components = path.split("/")
    if not components or components[0] != "state":
        raise ValueError(f"invalid state metric path {path!r}")
    value = state
    for component in components[1:]:
        if isinstance(value, dict):
            if component in value:
                value = value[component]
            elif component.isdecimal() and int(component) in value:
                # Optimizer state_dict() keys are integer parameter ordinals,
                # while compare_states() renders every path component as text.
                value = value[int(component)]
            else:
                raise KeyError(
                    f"state metric path component {component!r} is absent"
                )
        elif isinstance(value, (list, tuple)):
            value = value[int(component)]
        else:
            raise ValueError(f"state metric path stops before {component!r}")
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"state metric path {path!r} does not resolve to a tensor")
    return value


def _failure_tensor_values(
    names: list[str], reference: object, actual: object
) -> dict[str, dict[str, object]]:
    """Keep bounded concrete values for failed numerical comparisons."""

    result: dict[str, dict[str, object]] = {}
    for name in names:
        expected = _state_tensor_at_path(reference, name).detach().cpu().reshape(-1)
        observed = _state_tensor_at_path(actual, name).detach().cpu().reshape(-1)
        limit = min(64, expected.numel())
        result[name] = {
            "numel": expected.numel(),
            "truncated": expected.numel() > limit,
            "reference": expected[:limit].tolist(),
            "actual": observed[:limit].tolist(),
        }
    return result


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


def _profiling_metadata(
    case: Any,
    supplied: list[object] | None,
) -> list[object]:
    """Return explicit value-sensitive workload classes for task profiling.

    The built-in qualification cases place packed sequence lengths in the third
    microbatch position.  Custom cases can provide an arbitrary JSON list with
    ``--profiling-metadata`` instead of relying on that convenience.
    """

    if supplied is not None:
        if len(supplied) != len(case.microbatches):
            raise ValueError("profiling metadata must have one entry per microbatch")
        return supplied
    result: list[object] = []
    for microbatch in case.microbatches:
        sequence_lengths = microbatch[2] if len(microbatch) > 2 else None
        if isinstance(sequence_lengths, (list, tuple)) and all(
            isinstance(value, int) for value in sequence_lengths
        ):
            result.append({"sequence_lengths": list(sequence_lengths)})
        else:
            result.append(None)
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
    optimizer_ordering: str = "stage_interleaved",
    steps: int = 5,
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
        "optimizer_ordering": optimizer_ordering,
        "steps": steps,
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
    profiling = phase_seconds.get(
        "unique_stage_warmup_profiling",
        phase_seconds.get("structural_profiling", 0.0),
    )
    compilation = phase_seconds.get(
        "compiled_entrypoint_construction",
        phase_seconds.get("compilation", 0.0),
    )
    cached_warmup = phase_seconds.get("cached_entrypoint_warmup", 0.0)
    profile_orchestration = phase_seconds.get(
        "profile_cache_and_entrypoint_orchestration", 0.0
    )
    program_lowering = phase_seconds.get("program_lowering", 0.0)
    pressurefit = phase_seconds.get("pressurefit_simulation", 0.0)
    admission = phase_seconds.get("host_admission", 0.0) + phase_seconds.get(
        "slab_admission", 0.0
    )
    classified = (
        lowering_aot
        + profiling
        + compilation
        + cached_warmup
        + profile_orchestration
        + program_lowering
        + pressurefit
        + admission
    )
    return {
        "lowering_aot": lowering_aot,
        "profiling": profiling,
        "compiled_entrypoint_construction": compilation,
        "cached_entrypoint_warmup": cached_warmup,
        "profile_cache_and_entrypoint_orchestration": profile_orchestration,
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
    optimizer_ordering: str,
    steps: int,
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
    execution_timings: list[dict[str, object]] = []
    with case.implementations():
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            event_factory: Any = torch.cuda.Event
            compute_start = event_factory(enable_timing=True)
            compute_end = event_factory(enable_timing=True)
            task_events: list[
                tuple[str, int | None, torch.cuda.Event, torch.cuda.Event]
            ] = []
            started = time.perf_counter()
            compute_start.record(torch.cuda.current_stream())
            step_losses: list[float] = []
            for microbatch_index, microbatch in enumerate(microbatches):
                forward_start = event_factory(enable_timing=True)
                forward_end = event_factory(enable_timing=True)
                forward_start.record(torch.cuda.current_stream())
                loss = compiled_objective(*microbatch)
                forward_end.record(torch.cuda.current_stream())
                task_events.append(
                    ("forward", microbatch_index, forward_start, forward_end)
                )
                backward_start = event_factory(enable_timing=True)
                backward_end = event_factory(enable_timing=True)
                backward_start.record(torch.cuda.current_stream())
                loss.backward()
                backward_end.record(torch.cuda.current_stream())
                task_events.append(
                    ("backward", microbatch_index, backward_start, backward_end)
                )
                step_losses.append(float(loss.detach()))
            optimizer_start = event_factory(enable_timing=True)
            optimizer_end = event_factory(enable_timing=True)
            optimizer_start.record(torch.cuda.current_stream())
            optimizer.step()
            optimizer_end.record(torch.cuda.current_stream())
            task_events.append(("optimizer", None, optimizer_start, optimizer_end))
            compute_end.record(torch.cuda.current_stream())
            torch.cuda.current_stream().synchronize()
            elapsed = time.perf_counter() - started
            timings.append(elapsed)
            compute_timings.append(float(compute_start.elapsed_time(compute_end)) / 1e3)
            task_records: list[dict[str, object]] = []
            phase_seconds: dict[str, float] = {}
            for phase, recorded_microbatch, task_start, task_end in task_events:
                duration = float(task_start.elapsed_time(task_end)) / 1e3
                phase_seconds[phase] = phase_seconds.get(phase, 0.0) + duration
                task_records.append(
                    {
                        "phase": phase,
                        "microbatch": recorded_microbatch,
                        "gpu_start_seconds": (
                            float(compute_start.elapsed_time(task_start)) / 1e3
                        ),
                        "gpu_end_seconds": (
                            float(compute_start.elapsed_time(task_end)) / 1e3
                        ),
                        "gpu_duration_seconds": duration,
                    }
                )
            execution_timings.append(
                {
                    "compute_seconds": compute_timings[-1],
                    "optimizer_seconds": (
                        float(optimizer_start.elapsed_time(optimizer_end)) / 1e3
                    ),
                    "host_call_seconds": elapsed,
                    "phase_gpu_seconds": phase_seconds,
                    "tasks": task_records,
                }
            )
            losses.append(step_losses)
            print(
                f"reference {model_implementation}/{family} "
                f"step {step + 1}/{steps}: {elapsed:.3f}s",
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
            optimizer_ordering=optimizer_ordering,
            steps=steps,
        ),
        "losses": losses,
        "step_seconds": timings,
        "compute_step_seconds": compute_timings,
        "execution_timings": execution_timings,
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
    optimizer_ordering: Literal["stage_interleaved", "tail"],
    steps: int,
    checkpoint_step: int,
    require_pressure: bool,
    planning_cachedir: Path | None,
    profiling_metadata: list[object] | None,
    save_plan: bool,
    force_fresh: bool,
    overwrite_plan: bool,
    implementation_revision: str | None,
) -> None:
    identity = _case_identity(
        model_name=family,
        model_implementation=model_implementation,
        seed=seed,
        model_config=model_config,
        data_geometry=data_geometry,
        case_factory=case_factory,
        case_options=case_options,
        optimizer_ordering=optimizer_ordering,
        steps=steps,
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
        workload_metadata = _profiling_metadata(case, profiling_metadata)
        runtime = Runtime(
            pools={
                "execution": device(physical_capacity=device_budget),
                "spill": pinned_host(capacity=_HOST_BUDGET),
            }
        )
        case = relocate_case_model(case, runtime=runtime)
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
            optimizer_ordering=optimizer_ordering,
            planning_cachedir=planning_cachedir,
            profiling_metadata=workload_metadata,
            save_plan=save_plan,
            force_fresh=force_fresh,
            overwrite_plan=overwrite_plan,
            implementation_revision=implementation_revision,
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
            "compilation="
            f"{planning_phases.get('compiled_entrypoint_construction', 0.0):.3f}s, "
            "profiling="
            f"{planning_phases.get('unique_stage_warmup_profiling', 0.0):.3f}s, "
            "pressurefit="
            f"{planning_phases.get('pressurefit_simulation', 0.0):.3f}s",
            flush=True,
        )
        plan_report_path = result_path.with_name(f"{result_path.stem}_plan_report.pt")
        # Planning evidence is valuable even when the first runtime step finds
        # a contract violation. Persist the immutable report before execution
        # rather than losing an expensive cold-plan artifact on failure.
        torch.save(training.plan_report, plan_report_path)
        pressurefit_fixtures = write_pressurefit_fixtures(
            results=training.plan_report.pressurefit_results,
            directory=result_path.parent / f"{result_path.stem}_pressurefit",
        )
        physical_statuses = [_check_physical_budget()]
        execution_baseline = _adapter_statistics()
        losses: list[list[float]] = []
        timings: list[float] = []
        compute_timings: list[float] = []
        execution_timings: list[dict[str, object]] = []
        step_diagnostics: list[dict[str, object]] = []
        checkpoint: object | None = None
        expected_replay: list[list[float]] = []
        for step in range(steps):
            started = time.perf_counter()
            step_result = training(case.microbatches, runtime_trace=True)
            if step_result.diagnostics is None:
                raise AssertionError("runtime_trace=True omitted execution diagnostics")
            diagnostics = step_result.diagnostics.result()
            execution_timing = diagnostics.timing
            physical_statuses.append(_check_physical_budget())
            timings.append(time.perf_counter() - started)
            compute_timings.append(execution_timing.compute_seconds)
            execution_timings.append(execution_timing.as_dict())
            step_diagnostics.append(diagnostics.as_dict())
            values = [float(item) for item in step_result.objectives]
            losses.append(values)
            print(
                f"shadowspill {model_implementation}/{family} "
                f"step {step + 1}/{steps}: {timings[-1]:.3f}s",
                flush=True,
            )
            if step + 1 == checkpoint_step:
                checkpoint = copy.deepcopy(training.state_dict())
            elif step + 1 > checkpoint_step:
                expected_replay.append(values)
        uninterrupted_state = training.state_dict()
        uninterrupted_digest = state_digest(uninterrupted_state)
        if checkpoint is None:
            raise AssertionError(f"step-{checkpoint_step} checkpoint was not captured")
        training.load_state_dict(checkpoint)
        replay_losses: list[list[float]] = []
        replay_steps = steps - checkpoint_step
        for replay_step in range(replay_steps):
            replay_started = time.perf_counter()
            step_result = training(case.microbatches, runtime_trace=True)
            if step_result.diagnostics is None:
                raise AssertionError("runtime_trace=True omitted replay diagnostics")
            step_result.diagnostics.result()
            physical_statuses.append(_check_physical_budget())
            replay_losses.append([float(item) for item in step_result.objectives])
            print(
                f"shadowspill {model_implementation}/{family} replay "
                f"{replay_step + 1}/{replay_steps}: "
                f"{time.perf_counter() - replay_started:.3f}s",
                flush=True,
            )
        final_state = training.state_dict()
        replay_digest = state_digest(final_state)
        report = training.plan_report
        runtime_statistics = _adapter_statistics()
        training.close()
        externalize_case_model(case, runtime=runtime)
        runtime.close()

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
    available_recomputation_savings, selected_recomputation_savings = (
        _recomputation_savings_bytes(
            report.execution_plan.program.recomputation_groups,
            report.execution_plan.selections,
            {
                group.alias_group_id: group.size_bytes
                for group in report.execution_plan.program.alias_groups
            },
        )
    )
    phase_seconds = {
        name: nanoseconds / 1e9 for name, nanoseconds in report.phase_timings_ns
    }
    qualification_result = {
        "schema": "shadowspill.numerical_qualification/v5",
        "reference_execution": _REFERENCE_EXECUTION,
        "family": family,
        "model_implementation": model_implementation,
        "case_identity": identity,
        "steps": steps,
        "checkpoint_step": checkpoint_step,
        "require_pressure": require_pressure,
        "case_request": {
            "model_name": family,
            "model_implementation": model_implementation,
            "seed": seed,
            "model_config": model_config,
            "data_geometry": data_geometry,
            "case_factory": case_factory,
            "case_options": case_options,
            "optimizer_ordering": optimizer_ordering,
            "profiling_metadata": workload_metadata,
        },
        "planning_cache_request": {
            "directory": (
                None if planning_cachedir is None else str(planning_cachedir.resolve())
            ),
            "save_plan": save_plan,
            "force_fresh": force_fresh,
            "overwrite_plan": overwrite_plan,
            "implementation_revision": implementation_revision,
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
        "plan_diagnostics": report.diagnostics.as_dict(),
        "pressurefit_seconds": phase_seconds.get("pressurefit_simulation", 0.0),
        "planned_step_seconds": timings,
        "planned_compute_seconds": compute_timings,
        "planned_execution_timings": execution_timings,
        "planned_step_diagnostics": step_diagnostics,
        "reference_step_seconds": reference["step_seconds"],
        "reference_compute_seconds": reference.get("compute_step_seconds", []),
        "reference_execution_timings": reference.get("execution_timings", []),
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
        "metric_failures": {
            name: asdict(tensor_results[name]) for name in metric_failures
        },
        "metric_failure_values": _failure_tensor_values(
            metric_failures,
            {"model": reference["model"], "optimizer": reference["optimizer"]},
            {"model": final_state["model"], "optimizer": final_state["optimizer"]},
        ),
        "exact_failures": exact_failures,
        "checkpoint_replay_bitwise": (
            uninterrupted_digest == replay_digest and expected_replay == replay_losses
        ),
        "checkpoint_steps": _optimizer_steps(checkpoint),
        "uninterrupted_steps": _optimizer_steps(uninterrupted_state),
        "replay_steps": _optimizer_steps(final_state),
        "transfer_bytes_evicted": report.transfer_bytes_evicted,
        "transfer_bytes_fetched": report.transfer_bytes_fetched,
        "selected_recomputation": any(
            option != "save" for _group, option in selections
        ),
        "recomputation_memory_saving_available": bool(
            available_recomputation_savings
        ),
        "maximum_recomputation_savings_bytes": available_recomputation_savings,
        "selected_recomputation_savings_bytes": selected_recomputation_savings,
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
        "execution_pool_bytes": int(runtime_statistics.runtime.execution_pool_bytes),
        "slab_peak_allocated_bytes": int(
            runtime_statistics.runtime.peak_allocated_bytes
        ),
        "spill_pool_bytes": int(runtime_statistics.runtime.spill_pool_bytes),
        "spill_peak_allocated_bytes": int(
            runtime_statistics.runtime.spill_peak_allocated_bytes
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
        "cold_cache_requested": force_fresh,
        "pressurefit_fixtures": pressurefit_fixtures,
        "plan_report_artifact": {
            "path": str(plan_report_path),
            "size_bytes": plan_report_path.stat().st_size,
            "sha256": hashlib.sha256(plan_report_path.read_bytes()).hexdigest(),
        },
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
            and qualification_result["aot_graph_pair_cache_misses"]
            == qualification_result["aot_unique_stage_abis"]
            and qualification_result["recomputation_cache_hits"] == 0
        )
    )
    qualification_result["recomputation_selection_required"] = False
    transfer_pressure_passed = _transfer_pressure_gate_passed(
        required=require_pressure,
        evicted_bytes=report.transfer_bytes_evicted,
        fetched_bytes=report.transfer_bytes_fetched,
    )
    qualification_result["transfer_pressure_gate_passed"] = (
        transfer_pressure_passed
    )
    qualification_result["pressure_gate_passed"] = transfer_pressure_passed
    qualification_result["passed"] = bool(
        not loss_failures
        and not metric_failures
        and not exact_failures
        and qualification_result["checkpoint_replay_bitwise"]
        and transfer_pressure_passed
        and report.predicted_device_peak_bytes <= device_budget
        and not any(physical_statuses)
        and qualification_result["physical_budget_sealed"]
        and qualification_result["peak_process_physical_bytes"] <= device_budget
        and qualification_result["slab_peak_allocated_bytes"]
        <= qualification_result["execution_pool_bytes"]
        and qualification_result["spill_peak_allocated_bytes"]
        <= qualification_result["spill_pool_bytes"]
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
            f"loss_failures={len(loss_failures)}, "
            f"metric_failures={len(metric_failures)}, "
            f"exact_failures={len(exact_failures)}, "
            "checkpoint_replay_bitwise="
            f"{qualification_result['checkpoint_replay_bitwise']}, "
            f"transfer_pressure_gate_passed={transfer_pressure_passed}, "
            f"physical_statuses={physical_statuses}, artifact={result_path}"
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
    optimizer_ordering: Literal["stage_interleaved", "tail"],
    steps: int,
    checkpoint_step: int,
    require_pressure: bool,
    planning_cachedir: Path | None,
    profiling_metadata_argument: str | None,
    save_plan: bool,
    force_fresh: bool,
    overwrite_plan: bool,
    implementation_revision: str | None,
    reuse_reference: bool,
) -> None:
    result_directory.mkdir(parents=True, exist_ok=True)
    prefix = f"{model_implementation}_{family}"
    reference = result_directory / f"{prefix}_reference.pt"
    result = result_directory / f"{prefix}.json"
    base = [sys.executable, "-m", "qualification.numerical.run"]
    options = [
        "--seed",
        str(seed),
        "--model-config",
        model_config_argument,
        "--optimizer-ordering",
        optimizer_ordering,
    ]
    options.extend(("--steps", str(steps)))
    if not require_pressure:
        options.append("--allow-fully-resident")
    if data_geometry_argument is not None:
        options.extend(("--data-geometry", data_geometry_argument))
    if profiling_metadata_argument is not None:
        options.extend(("--profiling-metadata", profiling_metadata_argument))
    if case_factory is not None:
        options.extend(("--case-factory", case_factory))
    for value in case_option_arguments:
        options.extend(("--case-option", value))
    environment = dict(os.environ)
    if not reuse_reference or not reference.is_file():
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
    planned_options: list[str] = []
    selected_cache = planning_cachedir or result_directory / "planning_cache"
    planned_options.extend(("--planning-cachedir", str(selected_cache)))
    if not save_plan:
        planned_options.append("--no-save-plan")
    if force_fresh:
        planned_options.append("--force-fresh")
    if overwrite_plan:
        planned_options.append("--overwrite-plan")
    if implementation_revision is not None:
        planned_options.extend(("--implementation-revision", implementation_revision))
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
            *planned_options,
            "--checkpoint-step",
            str(checkpoint_step),
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
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument(
        "--optimizer-ordering",
        choices=("stage_interleaved", "tail"),
        default="stage_interleaved",
        help="place grouped optimizer stages as soon as their gradients are final",
    )
    parser.add_argument("--checkpoint-step", type=int)
    parser.add_argument(
        "--allow-fully-resident",
        action="store_true",
        help="do not require real FETCH/EVICT activity",
    )
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
        "--profiling-metadata",
        metavar="JSON|@FILE",
        help=(
            "one JSON-compatible workload descriptor per microbatch; used only "
            "for value-sensitive profile/cache identity"
        ),
    )
    parser.add_argument(
        "--planning-cachedir",
        type=Path,
        help="shared planning artifact root (run mode defaults below the result dir)",
    )
    parser.add_argument(
        "--no-save-plan",
        action="store_true",
        help="do not write reusable planning artifacts",
    )
    parser.add_argument(
        "--force-fresh",
        action="store_true",
        help="bypass every planning-cache read",
    )
    parser.add_argument(
        "--overwrite-plan",
        action="store_true",
        help="replace matching artifacts; requires --force-fresh",
    )
    parser.add_argument(
        "--implementation-revision",
        help="explicit implementation identity for custom-kernel invalidation",
    )
    parser.add_argument(
        "--reuse-reference",
        action="store_true",
        help="reuse an existing compiled-reference artifact in run mode",
    )
    parser.add_argument(
        "--case-factory",
        metavar="MODULE:FUNCTION",
        help="factory for a model not in the built-in qualification registry",
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
        if arguments.steps < 2:
            raise ValueError("steps must be at least two")
        checkpoint_step = arguments.checkpoint_step or max(1, arguments.steps - 2)
        if checkpoint_step < 1 or checkpoint_step >= arguments.steps:
            raise ValueError("checkpoint step must be between one and steps - 1")
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
        profiling_metadata_value = None
        if arguments.profiling_metadata is not None:
            decoded_metadata = _json_argument(
                arguments.profiling_metadata,
                description="profiling metadata",
            )
            if not isinstance(decoded_metadata, list):
                raise ValueError("profiling metadata must decode to a list")
            profiling_metadata_value = decoded_metadata
        if arguments.overwrite_plan and (
            arguments.no_save_plan or not arguments.force_fresh
        ):
            raise ValueError(
                "--overwrite-plan requires saved artifacts and --force-fresh"
            )
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
            optimizer_ordering=arguments.optimizer_ordering,
            steps=arguments.steps,
            checkpoint_step=checkpoint_step,
            require_pressure=not arguments.allow_fully_resident,
            planning_cachedir=arguments.planning_cachedir,
            profiling_metadata_argument=arguments.profiling_metadata,
            save_plan=not arguments.no_save_plan,
            force_fresh=arguments.force_fresh,
            overwrite_plan=arguments.overwrite_plan,
            implementation_revision=arguments.implementation_revision,
            reuse_reference=arguments.reuse_reference,
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
            optimizer_ordering=arguments.optimizer_ordering,
            steps=arguments.steps,
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
            optimizer_ordering=arguments.optimizer_ordering,
            steps=arguments.steps,
            checkpoint_step=checkpoint_step,
            require_pressure=not arguments.allow_fully_resident,
            planning_cachedir=arguments.planning_cachedir,
            profiling_metadata=profiling_metadata_value,
            save_plan=not arguments.no_save_plan,
            force_fresh=arguments.force_fresh,
            overwrite_plan=arguments.overwrite_plan,
            implementation_revision=arguments.implementation_revision,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
