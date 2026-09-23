"""The planned arm: one case planned, stepped, checkpointed and replayed.

Everything the comparison and the artifact read is captured here, because
the runtime and the model are released before either of them runs: the state
dictionaries, the timings, the diagnostics and the pool statistics outlive
the run that produced them, and nothing else does.
"""

from __future__ import annotations

import copy
import hashlib
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from shadowspill.memory import device, pinned_host, transfer_route
from shadowspill.planner import GenericPlanningOptions, SearchOptions
from shadowspill.pytorch import Runtime, plan_step
from workloads.common.training import LEARNING_RATE, optimizer_state_init

from ..model_state import import_case_model, release_case_model
from ..plan_record import write_plan_records
from ..planning_phases import planning_breakdown, planning_summary
from ..runtime_evidence import (
    adapter_statistics,
    check_physical_budget,
    measured_rate_clause,
)
from .metrics import state_digest
from .references import reference_artifact_exists, reference_inputs_path
from .request import PlannedRequest, workload_metadata_for
from .tolerances import SPILL_BUDGET


@dataclass(frozen=True, slots=True)
class PlannedRun:
    """What one planned run produced, after its runtime has been closed."""

    workload_metadata: list[object]
    requested_input_digest: str
    planning_seconds: float
    plan_records: list[dict[str, object]]
    plan_report_artifact: dict[str, object] | None
    losses: list[list[float]]
    timings: list[float]
    compute_timings: list[float]
    step_summaries: list[dict[str, object]]
    step_diagnostics: list[dict[str, object]]
    checkpoint: Mapping[str, object]
    expected_replay: list[list[float]]
    replay_losses: list[list[float]]
    uninterrupted_state: Mapping[str, Any]
    uninterrupted_digest: str
    final_state: Mapping[str, Any]
    replay_digest: str
    physical_statuses: list[int]
    execution_baseline: Any
    runtime_statistics: Any
    spill_statistics: Any
    report: Any

    @property
    def phase_seconds(self) -> dict[str, float]:
        """The plan report's phases in seconds, as the artifact records them."""

        return {
            name: nanoseconds / 1e9
            for name, nanoseconds in self.report.phase_timings_ns
        }


@dataclass(frozen=True, slots=True)
class _Steps:
    """What the measured steps produced, and the checkpoint taken among them."""

    losses: list[list[float]]
    timings: list[float]
    compute_timings: list[float]
    step_summaries: list[dict[str, object]]
    step_diagnostics: list[dict[str, object]]
    checkpoint: Mapping[str, object]
    expected_replay: list[list[float]]


def _open_runtime(request: PlannedRequest) -> Runtime:
    """The two pools and the two routes every numerical case is planned on.

    The spill pool is the case's if it named one and a pinned-host pool
    otherwise. That single substitution is the whole of what the remote gate
    changes: same programs, same references, same tolerances, same routes --
    only the memory the spill pool is made of, which is exactly the variable
    under test.
    """

    return Runtime(
        pools={
            "execution": device(physical_capacity=request.device_budget),
            "spill": request.spill_pool or pinned_host(capacity=SPILL_BUDGET),
        },
        routes={
            "fetch": transfer_route(source="spill", destination="execution"),
            "evict": transfer_route(source="execution", destination="spill"),
        },
    )


def _checked_case(request: PlannedRequest) -> tuple[Any, str]:
    """Build the case, and refuse a reference that was not recorded for it."""

    if not reference_artifact_exists(request.reference_path):
        raise RuntimeError(
            "compiled reference is incomplete; expected both "
            f"{request.reference_path} and "
            f"{reference_inputs_path(request.reference_path)}"
        )
    case = request.case.build()
    reference_inputs = torch.load(
        reference_inputs_path(request.reference_path),
        map_location="cpu",
        weights_only=True,
    )
    requested_input_digest = state_digest(case.microbatches)
    if state_digest(reference_inputs) != requested_input_digest:
        raise RuntimeError(
            "compiled reference inputs differ from requested qualification; "
            "replace them with --regenerate-reference"
        )
    return case, requested_input_digest


def _tokens_per_step(case: Any) -> int | None:
    """Tokens one step consumes, read from the case's own microbatches.

    The performance matrix takes this from a manifest; this path has none, so
    it comes from the inputs: the first value in each microbatch is the token
    block, and a step is every microbatch it accumulates over. `None` where a
    case's inputs are not shaped that way, and the line below then reports
    seconds without a rate rather than a rate that is wrong.
    """

    total = 0
    for microbatch in getattr(case, "microbatches", None) or ():
        block = next(
            (item for item in microbatch if isinstance(item, torch.Tensor)), None
        )
        if block is None:
            return None
        total += int(block.numel())
    return total or None


def _announce_prediction(
    case: Any, training: Any, runtime: Runtime, case_name: str
) -> None:
    """State the prediction, the rates it was made from, and the measured ones.

    Said after planning and before the first step, so the steps below can be
    read against it. The measured rates are what the runtime calibrated at
    construction -- every runtime does, this matrix simply never said what it
    found. On a remote run they are the link's, which is the only place a
    degraded link would otherwise show up as nothing more than a slow step.
    """

    report = getattr(training, "plan_report", None)
    planned = getattr(report, "summary", None)
    if report is None or planned is None:
        return
    predicted_seconds = report.predicted_makespan_ns / 1e9
    unconstrained_seconds = planned.unconstrained_step_seconds
    tokens = _tokens_per_step(case)

    def rate(seconds: float) -> str:
        if tokens is None or seconds <= 0.0:
            return ""
        return f", {tokens / seconds:.2f} tokens/s"

    measured = measured_rate_clause(runtime)

    print(
        f"simulator predicts {case_name}: "
        f"{predicted_seconds:.4f} s/step{rate(predicted_seconds)} "
        f"(planned with: fetch "
        f"{planned.fetch_bandwidth_bytes_per_second / 1e9:.1f} GB/s, evict "
        f"{planned.evict_bandwidth_bytes_per_second / 1e9:.1f} GB/s"
        f"{measured})"
        f"; unconstrained throughput "
        f"{unconstrained_seconds:.4f} s/step{rate(unconstrained_seconds)}",
        flush=True,
    )


def _plan_case(
    case: Any,
    request: PlannedRequest,
    *,
    runtime: Runtime,
    workload_metadata: list[object],
    case_name: str,
) -> tuple[Any, float]:
    """Plan the step deterministically, and say what each phase cost."""

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
        optimizer_ordering=request.case.optimizer_ordering,
        **request.case.data_ordering_arguments(),
        artifact_store=request.artifact_store,
        build_store=request.build_store,
        plan_store=request.plan_store,
        profiling_metadata=workload_metadata,
        build_store_mode=request.build_store_mode,
        plan_store_mode=request.plan_store_mode,
        export_bypass_key=request.export_bypass_key,
        # One plan per tree: the search's shared placement gate would
        # otherwise settle on a different plan run to run, and a plan is
        # a reduction order the comparison below can see.
        search_options=SearchOptions(
            generic=GenericPlanningOptions(deterministic=True)
        ),
    )
    planning_seconds = time.perf_counter() - planning_started
    phases = {
        name: nanoseconds / 1e9
        for name, nanoseconds in training.plan_report.phase_timings_ns
    }
    print(
        planning_summary(
            case_name,
            planning_breakdown(phases, planning_seconds=planning_seconds),
        ),
        flush=True,
    )
    _announce_prediction(case, training, runtime, case_name)
    return training, planning_seconds


def _detailed_artifacts(
    training: Any, request: PlannedRequest
) -> tuple[list[dict[str, object]], dict[str, object] | None]:
    """Save the whole plan report and its records, when they were asked for.

    Detailed evidence is opt-in because PlanReport and trace payloads are
    intentionally comprehensive and can dwarf the compact numerical result.
    """

    if not request.detailed_artifacts:
        return [], None
    plan_report_path = request.result_path.with_name(
        f"{request.result_path.stem}_plan_report.pt"
    )
    torch.save(training.plan_report, plan_report_path)
    plan_records = write_plan_records(
        results=training.plan_report.search_results,
        directory=request.result_path.parent
        / f"{request.result_path.stem}_plan_records",
    )
    return plan_records, {
        "path": str(plan_report_path),
        "size_bytes": plan_report_path.stat().st_size,
        "sha256": hashlib.sha256(plan_report_path.read_bytes()).hexdigest(),
    }


def _measured_steps(
    training: Any,
    case: Any,
    request: PlannedRequest,
    *,
    case_name: str,
    physical_statuses: list[int],
) -> _Steps:
    """Step the case, checkpointing at the step the request names."""

    losses: list[list[float]] = []
    timings: list[float] = []
    compute_timings: list[float] = []
    step_diagnostics: list[dict[str, object]] = []
    step_summaries: list[dict[str, object]] = []
    checkpoint: Mapping[str, object] | None = None
    expected_replay: list[list[float]] = []
    for step in range(request.case.steps):
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
        if request.detailed_artifacts:
            step_diagnostics.append(diagnostics.as_dict())
        values = [float(item) for item in step_result.objectives]
        losses.append(values)
        # Objective tensors are caller-owned device outputs.  The scalar
        # evidence above is sufficient for qualification, so release each
        # StepResult before a later explicit Runtime.close().
        del step_result
        print(
            f"shadowspill {case_name} "
            f"step {step + 1}/{request.case.steps}: {timings[-1]:.3f}s",
            flush=True,
        )
        if step + 1 == request.checkpoint_step:
            checkpoint = copy.deepcopy(training.state_dict())
        elif step + 1 > request.checkpoint_step:
            expected_replay.append(values)
    if checkpoint is None:
        raise AssertionError(
            f"step-{request.checkpoint_step} checkpoint was not captured"
        )
    return _Steps(
        losses=losses,
        timings=timings,
        compute_timings=compute_timings,
        step_summaries=step_summaries,
        step_diagnostics=step_diagnostics,
        checkpoint=checkpoint,
        expected_replay=expected_replay,
    )


def _replayed_steps(
    training: Any,
    case: Any,
    request: PlannedRequest,
    *,
    case_name: str,
    physical_statuses: list[int],
) -> list[list[float]]:
    """Step again from the restored checkpoint, to the end of the case."""

    replay_losses: list[list[float]] = []
    replay_steps = request.case.steps - request.checkpoint_step
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
            f"shadowspill {case_name} replay "
            f"{replay_step + 1}/{replay_steps}: "
            f"{time.perf_counter() - replay_started:.3f}s",
            flush=True,
        )
    return replay_losses


def run_planned_case(request: PlannedRequest) -> PlannedRun:
    """Plan the case, step it, replay it from a checkpoint, and close."""

    case_name = f"{request.case.model_implementation}/{request.case.family}"
    runtime = _open_runtime(request)
    case, requested_input_digest = _checked_case(request)
    with case.implementations(deterministic=True):
        workload_metadata = workload_metadata_for(case, request.profiling_metadata)
        case = import_case_model(case, runtime=runtime)
        training, planning_seconds = _plan_case(
            case,
            request,
            runtime=runtime,
            workload_metadata=workload_metadata,
            case_name=case_name,
        )
        plan_records, plan_report_artifact = _detailed_artifacts(training, request)
        physical_statuses = [check_physical_budget()]
        execution_baseline = adapter_statistics()
        steps = _measured_steps(
            training,
            case,
            request,
            case_name=case_name,
            physical_statuses=physical_statuses,
        )
        uninterrupted_state = training.state_dict()
        uninterrupted_digest = state_digest(uninterrupted_state)
        training.load_state_dict(steps.checkpoint)
        replay_losses = _replayed_steps(
            training,
            case,
            request,
            case_name=case_name,
            physical_statuses=physical_statuses,
        )
        stage_started = time.perf_counter()
        final_state = training.state_dict()
        replay_digest = state_digest(final_state)
        print(
            f"shadowspill {case_name} final state captured "
            f"and digested: {time.perf_counter() - stage_started:.3f}s",
            flush=True,
        )
        report = training.plan_report
        runtime_statistics = adapter_statistics()
        # The adapter reports the pool it allocates from; the spill pool is asked
        # for its own numbers, before close releases what it holds.
        spill_statistics = runtime.pool_statistics("spill")
        stage_started = time.perf_counter()
        training.close()
        # Reference parity, checkpoint replay, and transfer evidence all read
        # the state_dict() copies captured above; the model itself is never
        # used again, so release it without another anonymous copy.
        release_case_model(case, runtime=runtime)
        runtime.close()
        print(
            f"shadowspill {case_name} callable and runtime "
            f"closed: {time.perf_counter() - stage_started:.3f}s",
            flush=True,
        )

    return PlannedRun(
        workload_metadata=workload_metadata,
        requested_input_digest=requested_input_digest,
        planning_seconds=planning_seconds,
        plan_records=plan_records,
        plan_report_artifact=plan_report_artifact,
        losses=steps.losses,
        timings=steps.timings,
        compute_timings=steps.compute_timings,
        step_summaries=steps.step_summaries,
        step_diagnostics=steps.step_diagnostics,
        checkpoint=steps.checkpoint,
        expected_replay=steps.expected_replay,
        replay_losses=replay_losses,
        uninterrupted_state=uninterrupted_state,
        uninterrupted_digest=uninterrupted_digest,
        final_state=final_state,
        replay_digest=replay_digest,
        physical_statuses=physical_statuses,
        execution_baseline=execution_baseline,
        runtime_statistics=runtime_statistics,
        spill_statistics=spill_statistics,
        report=report,
    )
