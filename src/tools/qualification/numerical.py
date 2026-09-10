"""Fresh-process compiled-reference/planned numerical qualification."""

from __future__ import annotations

import argparse
import copy
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

from shadowspill.ir import TaskAlternativeChoice, TaskAlternativeGroup
from shadowspill.memory import device, pinned_host, transfer_route
from shadowspill.planner import (
    GenericPlanningOptions,
    SearchOptions,
    StepDataOrdering,
)
from shadowspill.pytorch import (
    Runtime,
    plan_step,
)
from shadowspill.pytorch.accelerator import DEVICE_TYPE
from shadowspill.schema import artifact_schema
from shadowspill.store import StorePolicy
from tools.qualification.model_state import import_case_model, release_case_model
from tools.qualification.runtime_evidence import (
    adapter_statistics,
    check_physical_budget,
)
from workloads.common.training import LEARNING_RATE, optimizer_state_init
from workloads.numerical import (
    DEFAULT_DEVICE_BUDGETS,
    ModelImplementation,
    build_case,
)

from .numerical_metrics import compare_states, cpu_state, state_digest
from .plan_record import write_plan_records
from .references import (
    DEFAULT_APPROXIMATELY_1B_REFERENCE_DIRECTORY,
    REFERENCE_SCHEMA,
    canonical_reference_path,
    reference_artifact_exists,
    reference_inputs_path,
)

# Room for the whole training state of every cell -- parameters, gradients
# and both optimizer moments -- plus the activations that spill, without
# asking a shared machine to reserve more than the measurement needs. The
# gate proves the real bound itself: it fails unless the measured spill peak
# fits the pool.
_SPILL_BUDGET = 32 << 30
_LOSS_RELATIVE_TOLERANCE = 0.01
_LOSS_ABSOLUTE_TOLERANCE = 2e-5
_MINIMUM_COSINE = 0.999
_MAXIMUM_RELATIVE_L2 = 0.025
# An optimizer moment is an accumulator of small values, and the same step
# run under two plans is the same arithmetic in two reduction orders. That
# alone moves a second-moment estimate past the bound above while every
# weight agrees, so the moments get twice the room; the weights keep the
# bound, being what training produces.
_MAXIMUM_RELATIVE_L2_OPTIMIZER = 0.05
_MINIMUM_SIGN_AGREEMENT = 0.99
_REFERENCE_EXECUTION = "torch.compile.inductor.fullgraph"


def _state_half(key: str) -> str:
    """Which half of the training state a comparison key names."""
    parts = str(key).split("/")
    return parts[1] if len(parts) > 1 and parts[0] == "state" else "other"


def _meets_tensor_tolerance(metric: Any, *, key: str = "") -> bool:
    bound = (
        _MAXIMUM_RELATIVE_L2_OPTIMIZER
        if _state_half(key) == "optimizer"
        else _MAXIMUM_RELATIVE_L2
    )
    return bool(
        metric.cosine >= _MINIMUM_COSINE
        and metric.relative_l2 <= bound
        and metric.sign_agreement >= _MINIMUM_SIGN_AGREEMENT
    )


def _recomputation_savings_bytes(
    groups: Sequence[TaskAlternativeGroup],
    selections: Sequence[TaskAlternativeChoice],
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


def _failures_by_state(keys: Sequence[str]) -> dict[str, int]:
    """Count failing tensors by which half of the training state they are in.

    Weights disagreeing and optimizer moments disagreeing are different
    findings: the first says the step computed something else, the second is
    usually an accumulator whose small values are ill-conditioned for a
    relative comparison. An aggregate count cannot be read either way, so the
    split is reported even when one side is zero.
    """
    counts = {"model": 0, "optimizer": 0}
    for key in keys:
        half = _state_half(key)
        counts[half] = counts.get(half, 0) + 1
    return counts


def _state_split(keys: Sequence[str]) -> str:
    """Render the split for a message, always naming both halves."""
    counts = _failures_by_state(keys)
    named = [f"model {counts['model']}", f"optimizer {counts['optimizer']}"]
    named.extend(
        f"{half} {count}"
        for half, count in counts.items()
        if half not in ("model", "optimizer")
    )
    return ", ".join(named)


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
                raise KeyError(f"state metric path component {component!r} is absent")
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


def _device_microbatches(values: list[list[Any]]) -> list[list[Any]]:
    return [
        [
            item.to(DEVICE_TYPE) if isinstance(item, torch.Tensor) else item
            for item in microbatch
        ]
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


def _data_ordering_arguments(label: str | None) -> dict[str, Any]:
    """The plan_step keyword arguments a ``--data-ordering`` label asks for."""
    if label is None:
        return {}
    ordering = StepDataOrdering.from_label(label)
    return {
        "depth": ordering.depth,
        "breadth": ordering.breadth,
        "reverse_breadth": ordering.reverse_breadth,
        "pair_loss": ordering.pair_loss,
    }


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
    data_ordering: str | None = None,
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
    # The walk is deliberately not part of the identity: the reference is the
    # same step fully torch-compiled without ShadowSpill, which every walk of
    # the step must reproduce.
    del data_ordering
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


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
    search = phase_seconds.get("search", 0.0)
    admission = (
        phase_seconds.get("admission_facts", 0.0)
        + phase_seconds.get("spill_admission", 0.0)
        + phase_seconds.get("slab_admission", 0.0)
    )
    classified = (
        lowering_aot
        + profiling
        + compilation
        + cached_warmup
        + profile_orchestration
        + program_lowering
        + search
        + admission
    )
    return {
        "lowering_aot": lowering_aot,
        "profiling": profiling,
        "compiled_entrypoint_construction": compilation,
        "cached_entrypoint_warmup": cached_warmup,
        "profile_cache_and_entrypoint_orchestration": profile_orchestration,
        "canonical_program_lowering": program_lowering,
        "search": search,
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
    model = case.model.to(DEVICE_TYPE)
    microbatches = _device_microbatches(case.microbatches)
    optimizer = case.optimizer(model.parameters())
    # The planned arm is handed this rate on every step, so the reference has
    # to train at it too; comparing two arms trained differently says nothing.
    for group in optimizer.param_groups:
        group["lr"] = LEARNING_RATE

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
    with case.implementations(deterministic=True):
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
                loss.backward()  # type: ignore[no-untyped-call]
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
            phase_seconds: dict[str, float] = {}
            for phase, _microbatch, task_start, task_end in task_events:
                duration = float(task_start.elapsed_time(task_end)) / 1e3
                phase_seconds[phase] = phase_seconds.get(phase, 0.0) + duration
            execution_timings.append(
                {
                    "compute_seconds": compute_timings[-1],
                    "optimizer_seconds": (
                        float(optimizer_start.elapsed_time(optimizer_end)) / 1e3
                    ),
                    "dispatch_call_seconds": elapsed,
                    "phase_gpu_seconds": phase_seconds,
                }
            )
            losses.append(step_losses)
            print(
                f"reference {model_implementation}/{family} "
                f"step {step + 1}/{steps}: {elapsed:.3f}s",
                flush=True,
            )
    artifact = {
        "schema": REFERENCE_SCHEMA,
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
        "microbatch_digest": state_digest(case.microbatches),
        "model": cpu_state(model.state_dict()),
        "optimizer": cpu_state(optimizer.state_dict()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cpu_state(case.microbatches), reference_inputs_path(output))
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
    data_ordering: str | None,
    steps: int,
    checkpoint_step: int,
    require_pressure: bool,
    artifact_store: Path | None,
    build_store: Path | None,
    plan_store: Path | None,
    profiling_metadata: list[object] | None,
    build_store_mode: str,
    plan_store_mode: str,
    implementation_revision: str | None,
    detailed_artifacts: bool,
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
        data_ordering=data_ordering,
        steps=steps,
    )
    if not reference_artifact_exists(reference_path):
        raise RuntimeError(
            "compiled reference is incomplete; expected both "
            f"{reference_path} and {reference_inputs_path(reference_path)}"
        )
    runtime = Runtime(
        pools={
            "execution": device(physical_capacity=device_budget),
            "spill": pinned_host(capacity=_SPILL_BUDGET),
        },
        routes={
            "fetch": transfer_route(source="spill", destination="execution"),
            "evict": transfer_route(source="execution", destination="spill"),
        },
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
    reference_inputs = torch.load(
        reference_inputs_path(reference_path),
        map_location="cpu",
        weights_only=True,
    )
    requested_input_digest = state_digest(case.microbatches)
    if state_digest(reference_inputs) != requested_input_digest:
        raise RuntimeError(
            "compiled reference inputs differ from requested qualification; "
            "replace them with --regenerate-reference"
        )
    with case.implementations(deterministic=True):
        workload_metadata = _profiling_metadata(case, profiling_metadata)
        case = import_case_model(case, runtime=runtime)
        model = case.model
        planning_started = time.perf_counter()
        training = plan_step(
            model,
            objective=case.objective,
            optimizer=case.optimizer,
            optimizer_state_init=optimizer_state_init,
            hyperparams=("lr",),
            example_inputs=case.microbatches,
            runtime=runtime,
            execution="execution",
            spill="spill",
            optimizer_ordering=optimizer_ordering,
            **_data_ordering_arguments(data_ordering),
            artifact_store=artifact_store,
            build_store=build_store,
            plan_store=plan_store,
            profiling_metadata=workload_metadata,
            build_store_mode=build_store_mode,
            plan_store_mode=plan_store_mode,
            implementation_revision=implementation_revision,
            # One plan per tree: the search's shared placement gate would
            # otherwise settle on a different plan run to run, and a plan is
            # a reduction order the comparison below can see.
            search_options=SearchOptions(
                generic=GenericPlanningOptions(deterministic=True)
            ),
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
            "search="
            f"{planning_phases.get('search', 0.0):.3f}s",
            flush=True,
        )
        plan_report_artifact: dict[str, object] | None = None
        plan_records: list[dict[str, object]] = []
        if detailed_artifacts:
            plan_report_path = result_path.with_name(
                f"{result_path.stem}_plan_report.pt"
            )
            # Detailed evidence is opt-in because PlanReport and trace payloads
            # are intentionally comprehensive and can dwarf the compact
            # numerical qualification result.
            torch.save(training.plan_report, plan_report_path)
            plan_records = write_plan_records(
                results=training.plan_report.search_results,
                directory=result_path.parent / f"{result_path.stem}_plan_records",
            )
            plan_report_artifact = {
                "path": str(plan_report_path),
                "size_bytes": plan_report_path.stat().st_size,
                "sha256": hashlib.sha256(plan_report_path.read_bytes()).hexdigest(),
            }
        physical_statuses = [check_physical_budget()]
        execution_baseline = adapter_statistics()
        losses: list[list[float]] = []
        timings: list[float] = []
        compute_timings: list[float] = []
        step_diagnostics: list[dict[str, object]] = []
        step_summaries: list[dict[str, object]] = []
        checkpoint: Mapping[str, object] | None = None
        expected_replay: list[list[float]] = []
        for step in range(steps):
            started = time.perf_counter()
            step_result = training(
                case.microbatches,
                hyperparams={"lr": LEARNING_RATE},
                runtime_trace=True,
            )
            if step_result.diagnostics is None:
                raise AssertionError("runtime_trace=True omitted execution diagnostics")
            diagnostics = step_result.diagnostics.result()
            physical_statuses.append(check_physical_budget())
            timings.append(time.perf_counter() - started)
            compute_timings.append(diagnostics.summary.real_selected_span_seconds)
            step_summaries.append(diagnostics.summary.as_dict())
            if detailed_artifacts:
                step_diagnostics.append(diagnostics.as_dict())
            values = [float(item) for item in step_result.objectives]
            losses.append(values)
            # Objective tensors are caller-owned device outputs.  The scalar
            # evidence above is sufficient for qualification, so release each
            # StepResult before a later explicit Runtime.close().
            del step_result
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
            step_result = training(
                case.microbatches,
                hyperparams={"lr": LEARNING_RATE},
                runtime_trace=True,
            )
            if step_result.diagnostics is None:
                raise AssertionError("runtime_trace=True omitted replay diagnostics")
            step_result.diagnostics.result()
            physical_statuses.append(check_physical_budget())
            replay_losses.append([float(item) for item in step_result.objectives])
            del step_result
            print(
                f"shadowspill {model_implementation}/{family} replay "
                f"{replay_step + 1}/{replay_steps}: "
                f"{time.perf_counter() - replay_started:.3f}s",
                flush=True,
            )
        stage_started = time.perf_counter()
        final_state = training.state_dict()
        replay_digest = state_digest(final_state)
        print(
            f"shadowspill {model_implementation}/{family} final state captured "
            f"and digested: {time.perf_counter() - stage_started:.3f}s",
            flush=True,
        )
        report = training.plan_report
        runtime_statistics = adapter_statistics()
        stage_started = time.perf_counter()
        training.close()
        # Reference parity, checkpoint replay, and transfer evidence all read
        # the state_dict() copies captured above; the model itself is never
        # used again, so release it without another anonymous copy.
        release_case_model(case, runtime=runtime)
        runtime.close()
        print(
            f"shadowspill {model_implementation}/{family} callable and runtime "
            f"closed: {time.perf_counter() - stage_started:.3f}s",
            flush=True,
        )

    stage_started = time.perf_counter()
    reference = torch.load(reference_path, map_location="cpu", weights_only=True)
    print(
        f"shadowspill {model_implementation}/{family} reference loaded from "
        f"{reference_path.resolve()}: "
        f"{time.perf_counter() - stage_started:.3f}s",
        flush=True,
    )
    if (
        reference.get("schema") != REFERENCE_SCHEMA
        or reference.get("reference_execution") != _REFERENCE_EXECUTION
        or reference.get("family") != family
        or reference.get("model_implementation") != model_implementation
        or reference.get("case_identity") != identity
        or (
            reference.get("schema") == REFERENCE_SCHEMA
            and reference.get("microbatch_digest") != requested_input_digest
        )
    ):
        raise RuntimeError(
            "compiled reference identity differs from requested qualification; "
            "replace it with --regenerate-reference"
        )
    stage_started = time.perf_counter()
    print(
        f"shadowspill {model_implementation}/{family} comparing model and "
        "optimizer state against the reference, tensor by tensor",
        flush=True,
    )
    tensor_results, exact_failures, structure_failures = compare_states(
        {"model": reference["model"], "optimizer": reference["optimizer"]},
        {"model": final_state["model"], "optimizer": final_state["optimizer"]},
    )
    print(
        f"shadowspill {model_implementation}/{family} compared "
        f"{len(tensor_results)} tensors: "
        f"{time.perf_counter() - stage_started:.3f}s",
        flush=True,
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
        if not _meets_tensor_tolerance(metric, key=name)
    ]
    # The replayed run has to agree with the uninterrupted one, but it cannot
    # be required to agree bit for bit: a step is only bitwise reproducible if
    # every kernel under it is, and the mlops path's are not on every
    # accelerator. Hold the replay to the same per-tensor tolerance the
    # reference comparison uses, and keep the bitwise answer as evidence.
    stage_started = time.perf_counter()
    print(
        f"shadowspill {model_implementation}/{family} comparing the "
        "checkpoint-replayed state against the uninterrupted run, tensor by "
        "tensor",
        flush=True,
    )
    replay_results, replay_exact_failures, replay_structure_failures = (
        compare_states(
            {
                "model": uninterrupted_state["model"],
                "optimizer": uninterrupted_state["optimizer"],
            },
            {
                "model": final_state["model"],
                "optimizer": final_state["optimizer"],
            },
        )
    )
    print(
        f"shadowspill {model_implementation}/{family} compared "
        f"{len(replay_results)} replayed tensors: "
        f"{time.perf_counter() - stage_started:.3f}s",
        flush=True,
    )
    replay_metric_failures = [
        name
        for name, metric in replay_results.items()
        if not _meets_tensor_tolerance(metric, key=name)
    ]
    stage_started = time.perf_counter()
    print(
        f"shadowspill {model_implementation}/{family} assembling the case "
        "artifact: diagnostics, plan records, and the state digests",
        flush=True,
    )
    selections = tuple(
        (item.group_id, item.option_id) for item in report.execution_plan.selections
    )
    available_recomputation_savings, selected_recomputation_savings = (
        _recomputation_savings_bytes(
            report.execution_plan.program.task_alternative_groups,
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
        "schema": artifact_schema("numerical_qualification"),
        "reference_execution": _REFERENCE_EXECUTION,
        "family": family,
        "model_implementation": model_implementation,
        "case_identity": identity,
        "reference_artifact": {
            "path": str(reference_path.resolve()),
            "schema": reference["schema"],
            "inputs_path": str(reference_inputs_path(reference_path).resolve()),
            "microbatch_digest": requested_input_digest,
        },
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
            "data_ordering": data_ordering,
            "profiling_metadata": workload_metadata,
        },
        "planning_cache_request": {
            "directory": (
                None
                if artifact_store is None
                else str(artifact_store.resolve())
            ),
            "build_store_mode": build_store_mode,
            "plan_store_mode": plan_store_mode,
            "implementation_revision": implementation_revision,
        },
        "device_budget_bytes": device_budget,
        "tolerances": {
            "loss_rtol": _LOSS_RELATIVE_TOLERANCE,
            "loss_atol": _LOSS_ABSOLUTE_TOLERANCE,
            "minimum_cosine": _MINIMUM_COSINE,
            "maximum_relative_l2": _MAXIMUM_RELATIVE_L2,
            "maximum_relative_l2_optimizer": _MAXIMUM_RELATIVE_L2_OPTIMIZER,
            "minimum_sign_agreement": _MINIMUM_SIGN_AGREEMENT,
        },
        "planning_seconds": planning_seconds,
        "phase_seconds": phase_seconds,
        "planning_breakdown_seconds": _planning_breakdown(
            phase_seconds, planning_seconds=planning_seconds
        ),
        "artifact_detail": "detailed" if detailed_artifacts else "compact",
        "search_seconds": phase_seconds.get("search", 0.0),
        "planned_step_seconds": timings,
        "planned_compute_seconds": compute_timings,
        "planned_step_summaries": step_summaries,
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
        "metric_failures_by_state": _failures_by_state(metric_failures),
        "exact_failures_by_state": _failures_by_state(exact_failures),
        "structure_failures_by_state": _failures_by_state(structure_failures),
        "metric_failures": {
            name: asdict(tensor_results[name]) for name in metric_failures
        },
        "metric_failure_values": _failure_tensor_values(
            metric_failures,
            {"model": reference["model"], "optimizer": reference["optimizer"]},
            {"model": final_state["model"], "optimizer": final_state["optimizer"]},
        ),
        "exact_failures": exact_failures,
        "structure_failures": structure_failures,
        "checkpoint_replay_bitwise": (
            uninterrupted_digest == replay_digest and expected_replay == replay_losses
        ),
        "checkpoint_replay_within_tolerance": (
            not replay_exact_failures and not replay_metric_failures
        ),
        "checkpoint_replay_exact_failures": replay_exact_failures,
        "checkpoint_replay_metric_failure_keys": replay_metric_failures,
        "checkpoint_replay_metric_failures_by_state": _failures_by_state(
            replay_metric_failures
        ),
        "checkpoint_replay_metric_failures": {
            name: asdict(replay_results[name]) for name in replay_metric_failures
        },
        "checkpoint_replay_minimum_cosine": min(
            (item.cosine for item in replay_results.values()), default=1.0
        ),
        "checkpoint_replay_maximum_relative_l2": max(
            (item.relative_l2 for item in replay_results.values()), default=0.0
        ),
        "checkpoint_replay_maximum_absolute_error": max(
            (item.maximum_absolute_error for item in replay_results.values()),
            default=0.0,
        ),
        "checkpoint_replay_tensors_compared": len(replay_results),
        "checkpoint_replay_tensors_differing": sum(
            1 for item in replay_results.values() if item.difference_norm > 0.0
        ),
        "checkpoint_steps": _optimizer_steps(checkpoint),
        "uninterrupted_steps": _optimizer_steps(uninterrupted_state),
        "replay_steps": _optimizer_steps(final_state),
        "transfer_bytes_evicted": report.transfer_bytes_evicted,
        "transfer_bytes_fetched": report.transfer_bytes_fetched,
        "selected_recomputation": any(
            option != "save" for _group, option in selections
        ),
        "recomputation_memory_saving_available": bool(available_recomputation_savings),
        "maximum_recomputation_savings_bytes": available_recomputation_savings,
        "selected_recomputation_savings_bytes": selected_recomputation_savings,
        "task_alternative_group_count": len(selections),
        "task_count": len(
            report.execution_plan.program.selected_tasks(
                report.execution_plan.selections
            )
        ),
        "action_count": len(report.transfer_actions),
        "predicted_makespan_seconds": report.predicted_makespan_ns / 1e9,
        "predicted_device_peak_bytes": report.predicted_device_peak_bytes,
        "predicted_spill_peak_bytes": report.predicted_spill_peak_bytes,
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
        "neutral_event_lease_capacity": int(
            runtime_statistics.runtime.event_lease_capacity
        ),
        "neutral_event_lease_peak_in_use": int(
            runtime_statistics.runtime.event_lease_peak_in_use
        ),
        "neutral_event_lease_growth_rejections": int(
            runtime_statistics.runtime.event_lease_growth_rejections
        ),
        "retirement_record_capacity": int(
            runtime_statistics.runtime.retirement_record_capacity
        ),
        "retirement_record_peak_in_use": int(
            runtime_statistics.runtime.retirement_record_peak_in_use
        ),
        "retirement_record_growth_rejections": int(
            runtime_statistics.runtime.retirement_record_growth_rejections
        ),
        "memory_lease_record_capacity": int(
            runtime_statistics.runtime.memory_lease_record_capacity
        ),
        "memory_lease_record_peak_in_use": int(
            runtime_statistics.runtime.memory_lease_record_peak_in_use
        ),
        "memory_lease_record_growth_rejections": int(
            runtime_statistics.runtime.memory_lease_record_growth_rejections
        ),
        "lease_use_record_capacity": int(
            runtime_statistics.runtime.lease_use_record_capacity
        ),
        "lease_use_record_peak_in_use": int(
            runtime_statistics.runtime.lease_use_record_peak_in_use
        ),
        "lease_use_record_growth_rejections": int(
            runtime_statistics.runtime.lease_use_record_growth_rejections
        ),
        "backend_device_allocations": int(
            runtime_statistics.backend.device_allocations
        ),
        "steady_state_backend_device_allocations": int(
            runtime_statistics.backend.device_allocations
            - execution_baseline.backend.device_allocations
        ),
        "steady_state_pinned_host_registrations": int(
            runtime_statistics.backend.pinned_host_registrations
            - execution_baseline.backend.pinned_host_registrations
        ),
        "event_pool_capacity": int(runtime_statistics.runtime.event_lease_capacity),
        "event_pool_peak_in_use": int(
            runtime_statistics.runtime.event_lease_peak_in_use
        ),
        "event_pool_driver_creates": int(
            runtime_statistics.runtime.event_lease_driver_creates
        ),
        "steady_state_event_pool_driver_creates": int(
            runtime_statistics.runtime.event_lease_driver_creates
            - execution_baseline.runtime.event_lease_driver_creates
        ),
        "event_pool_growth_rejections": int(
            runtime_statistics.runtime.event_lease_growth_rejections
        ),
        "event_pool_sealed": bool(runtime_statistics.runtime.event_lease_sealed),
        "profile_cache_hits": report.profile_cache_hits,
        "profile_cache_misses": report.profile_cache_misses,
        "profile_unique_keys": report.profile_unique_keys,
        "captured_stage_count": report.captured_stage_count,
        "aot_unique_stage_contracts": report.aot_unique_stage_contracts,
        "aot_graph_pair_cache_hits": report.aot_graph_pair_cache_hits,
        "aot_graph_pair_cache_misses": report.aot_graph_pair_cache_misses,
        "planned_program_cache_hits": report.planned_program_cache_hits,
        "planned_program_cache_misses": report.planned_program_cache_misses,
        "build_store_mode": build_store_mode,
        "plan_store_mode": plan_store_mode,
        "plan_records": plan_records,
        "plan_report_artifact": plan_report_artifact,
        "reference_state_digest": state_digest(
            {"model": reference["model"], "optimizer": reference["optimizer"]}
        ),
        "planned_state_digest": state_digest(
            {"model": final_state["model"], "optimizer": final_state["optimizer"]}
        ),
    }
    if detailed_artifacts:
        qualification_result["plan_diagnostics"] = report.diagnostics.as_dict()
        qualification_result["planned_step_diagnostics"] = step_diagnostics
    qualification_result["reference_bitwise_equal"] = bool(
        qualification_result["reference_state_digest"]
        == qualification_result["planned_state_digest"]
    )
    # A store mode that disables reads must actually have served nothing. The
    # policy the mode implies is the authority on whether reads were allowed,
    # so there is no second flag to disagree with it; a tree that was allowed
    # to read is not asked to prove anything.
    build_reads = StorePolicy.for_mode(build_store_mode).read_enabled
    plan_reads = StorePolicy.for_mode(plan_store_mode).read_enabled
    qualification_result["store_modes_honoured"] = bool(
        (
            build_reads
            or (
                qualification_result["profile_cache_hits"] == 0
                and qualification_result["aot_graph_pair_cache_misses"]
                == qualification_result["aot_unique_stage_contracts"]
            )
        )
        and (plan_reads or qualification_result["planned_program_cache_hits"] == 0)
    )
    qualification_result["graph_pair_selection_required"] = False
    transfer_pressure_passed = _transfer_pressure_gate_passed(
        required=require_pressure,
        evicted_bytes=report.transfer_bytes_evicted,
        fetched_bytes=report.transfer_bytes_fetched,
    )
    qualification_result["transfer_pressure_gate_passed"] = transfer_pressure_passed
    qualification_result["pressure_gate_passed"] = transfer_pressure_passed
    # Every check names the kind of failure it reports, because "the cell
    # failed" is not actionable: disagreeing with a reference recorded on
    # other hardware, failing to reproduce its own replay, and exceeding a
    # budget are different problems with different owners.
    checks: tuple[tuple[str, object, str], ...] = (
        ("reference", not loss_failures, f"{len(loss_failures)} losses differ"),
        # First, because it decides whether the rest means anything. States
        # that do not have the same shape are not two answers to one question,
        # so the tolerance numbers below are taken over tensors that do not
        # correspond and describe nothing.
        (
            "reference",
            not structure_failures,
            f"{len(structure_failures)} tensors do not match the reference's "
            f"structure [{_state_split(structure_failures)}]: the reference "
            "records a different shape, so the comparison is not a numerical "
            f"one -- first is {structure_failures[0] if structure_failures else ''}",
        ),
        (
            "reference",
            not metric_failures,
            f"{len(metric_failures)} tensors outside tolerance "
            f"[{_state_split(metric_failures)}] "
            f"(min cosine {qualification_result['minimum_cosine']:.6f}, "
            f"max relative l2 "
            f"{qualification_result['maximum_relative_l2']:.6f})",
        ),
        (
            "reference",
            not exact_failures,
            f"{len(exact_failures)} tensors differ that must match exactly "
            f"[{_state_split(exact_failures)}]",
        ),
        (
            "replay",
            not replay_structure_failures,
            f"{len(replay_structure_failures)} tensors do not match the "
            "replay's structure "
            f"[{_state_split(replay_structure_failures)}]",
        ),
        (
            "replay",
            qualification_result["checkpoint_replay_within_tolerance"],
            f"{qualification_result['checkpoint_replay_tensors_differing']} of "
            f"{qualification_result['checkpoint_replay_tensors_compared']} "
            "tensors differ from the checkpoint replay "
            f"[{_state_split(replay_metric_failures)}] (min cosine "
            f"{qualification_result['checkpoint_replay_minimum_cosine']:.6f})",
        ),
        ("pressure", transfer_pressure_passed, "transfer pressure gate"),
        (
            "budget",
            report.predicted_device_peak_bytes <= device_budget,
            f"predicted device peak {report.predicted_device_peak_bytes} "
            f"over budget {device_budget}",
        ),
        (
            "budget",
            not any(physical_statuses),
            f"physical statuses {physical_statuses}",
        ),
        ("budget", qualification_result["physical_budget_sealed"], "budget not sealed"),
        (
            "budget",
            qualification_result["peak_process_physical_bytes"] <= device_budget,
            f"process peak {qualification_result['peak_process_physical_bytes']} "
            f"over budget {device_budget}",
        ),
        (
            "budget",
            qualification_result["slab_peak_allocated_bytes"]
            <= qualification_result["execution_pool_bytes"],
            "slab peak over the execution pool",
        ),
        (
            "budget",
            qualification_result["spill_peak_allocated_bytes"]
            <= qualification_result["spill_pool_bytes"]
            <= _SPILL_BUDGET,
            "spill peak over the pool, or the pool over its budget",
        ),
        (
            "runtime",
            qualification_result["callback_failures"] == 0,
            "callback failures",
        ),
        (
            "runtime",
            qualification_result["pointer_lookup_failures"] == 0,
            "pointer lookup failures",
        ),
        (
            "runtime",
            not qualification_result["allocation_event_overflow"],
            "allocation event overflow",
        ),
        (
            "runtime",
            qualification_result["backend_device_allocations"] == 1,
            f"{qualification_result['backend_device_allocations']} backend device "
            "allocations, expected one",
        ),
        (
            "runtime",
            qualification_result["steady_state_backend_device_allocations"] == 0,
            "device allocations in steady state",
        ),
        (
            "runtime",
            qualification_result["steady_state_pinned_host_registrations"] == 0,
            "pinned host registrations in steady state",
        ),
        (
            "runtime",
            qualification_result["steady_state_event_pool_driver_creates"] == 0,
            "event pool driver creates in steady state",
        ),
        (
            "runtime",
            qualification_result["event_pool_growth_rejections"] == 0,
            "event pool growth rejections",
        ),
        ("runtime", qualification_result["event_pool_sealed"], "event pool not sealed"),
        (
            "store",
            qualification_result["store_modes_honoured"],
            "a store whose mode disables reads still served a hit",
        ),
    )
    failures = [
        {"category": category, "detail": detail}
        for category, satisfied, detail in checks
        if not satisfied
    ]
    failure_categories = sorted({failure["category"] for failure in failures})
    qualification_result["failures"] = failures
    qualification_result["failure_categories"] = failure_categories
    qualification_result["passed"] = not failures
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(qualification_result, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"shadowspill {model_implementation}/{family} wrote "
        f"{result_path.name}, {len(failures)} failure(s): "
        f"{time.perf_counter() - stage_started:.3f}s",
        flush=True,
    )
    if not qualification_result["passed"]:
        raise AssertionError(
            f"{model_implementation} {family} numerical qualification failed "
            f"({', '.join(failure_categories)}): "
            + "; ".join(
                f"{failure['category']}: {failure['detail']}" for failure in failures
            )
            + f", artifact={result_path}"
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
    data_ordering: str | None,
    steps: int,
    checkpoint_step: int,
    require_pressure: bool,
    artifact_store: Path | None,
    build_store: Path | None,
    plan_store: Path | None,
    profiling_metadata_argument: str | None,
    build_store_mode: str,
    plan_store_mode: str,
    implementation_revision: str | None,
    reference_directory: Path,
    regenerate_reference: bool,
    detailed_artifacts: bool,
) -> None:
    result_directory.mkdir(parents=True, exist_ok=True)
    prefix = f"{model_implementation}_{family}"
    reference = canonical_reference_path(
        reference_directory,
        model_name=family,
        implementation=model_implementation,
    )
    result = result_directory / f"{prefix}.json"
    base = [sys.executable, "-m", "tools.qualification.numerical"]
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
    if data_ordering is not None:
        options.extend(("--data-ordering", data_ordering))
    if profiling_metadata_argument is not None:
        options.extend(("--profiling-metadata", profiling_metadata_argument))
    if case_factory is not None:
        options.extend(("--case-factory", case_factory))
    for value in case_option_arguments:
        options.extend(("--case-option", value))
    environment = dict(os.environ)
    if regenerate_reference or not reference_artifact_exists(reference):
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
    selected_cache = artifact_store or result_directory / "artifact_store"
    planned_options.extend(("--artifact-store", str(selected_cache)))
    for flag, path in (("--build-store", build_store), ("--plan-store", plan_store)):
        if path is not None:
            planned_options.extend((flag, str(path)))
    for tree, mode in (("build", build_store_mode), ("plan", plan_store_mode)):
        if mode != "contribute":
            planned_options.extend((f"--{tree}-store-mode", mode))
    if implementation_revision is not None:
        planned_options.extend(("--implementation-revision", implementation_revision))
    if detailed_artifacts:
        planned_options.append("--detailed-artifacts")
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
        "--data-ordering",
        help="how the step walks its microbatches, as <depth>x<breadth> with"
        " r for the reversed backward walk and p for the paired loss, for"
        " example 2x4rp; omitted plans depth-first as every step did before",
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
        "--artifact-store",
        type=Path,
        help="roots both stores (run mode defaults below the result dir)",
    )
    parser.add_argument("--build-store", type=Path)
    parser.add_argument("--plan-store", type=Path)
    for tree in ("build", "plan"):
        parser.add_argument(
            f"--{tree}-store-mode",
            choices=("contribute", "reuse", "require"),
            default="contribute",
            help=f"what this run may do about a {tree} artifact the store does"
            " not hold: contribute writes it back, reuse persists nothing,"
            " require refuses",
        )
    parser.add_argument(
        "--implementation-revision",
        help="explicit implementation identity for custom-kernel invalidation",
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=DEFAULT_APPROXIMATELY_1B_REFERENCE_DIRECTORY,
        help="canonical compiled-reference root used by run mode",
    )
    parser.add_argument(
        "--regenerate-reference",
        action="store_true",
        help="replace the canonical compiled reference in run mode",
    )
    parser.add_argument(
        "--detailed-artifacts",
        action="store_true",
        help=(
            "persist the complete PlanReport, plan records, and per-task "
            "step traces; compact correctness evidence is the default"
        ),
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
            data_ordering=arguments.data_ordering,
            steps=arguments.steps,
            checkpoint_step=checkpoint_step,
            require_pressure=not arguments.allow_fully_resident,
            artifact_store=arguments.artifact_store,
            build_store=arguments.build_store,
            plan_store=arguments.plan_store,
            profiling_metadata_argument=arguments.profiling_metadata,
            build_store_mode=arguments.build_store_mode,
            plan_store_mode=arguments.plan_store_mode,
            implementation_revision=arguments.implementation_revision,
            reference_directory=arguments.reference_dir,
            regenerate_reference=arguments.regenerate_reference,
            detailed_artifacts=arguments.detailed_artifacts,
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
            data_ordering=arguments.data_ordering,
            steps=arguments.steps,
            checkpoint_step=checkpoint_step,
            require_pressure=not arguments.allow_fully_resident,
            artifact_store=arguments.artifact_store,
            build_store=arguments.build_store,
            plan_store=arguments.plan_store,
            profiling_metadata=profiling_metadata_value,
            build_store_mode=arguments.build_store_mode,
            plan_store_mode=arguments.plan_store_mode,
            implementation_revision=arguments.implementation_revision,
            detailed_artifacts=arguments.detailed_artifacts,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
