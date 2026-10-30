"""Fresh-worker numerical execution of a saved Plan; never build or profile."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlops.dispatch import deterministic_kernels

from qualification.pipeline.observe import Observe
from qualification.pipeline.state import initialize, load_inputs, snapshot
from qualification.runtime_evidence import adapter_statistics, check_physical_budget
from shadowspill import Executor, Plan, load_checkpoint, save_checkpoint

from .metrics import state_digest
from .request import PlannedRequest


@dataclass(frozen=True)
class PlannedRun:
    plan: Plan
    phase_seconds: dict[str, float]
    workload_metadata: list[object]
    requested_input_digest: str
    plan_path: Path
    losses: list[list[float]]
    timings: list[float]
    device_timings: list[float]
    step_summaries: list[dict[str, Any]]
    step_diagnostics: list[dict[str, Any]]
    checkpoint: dict[str, Any]
    expected_replay: list[list[float]]
    replay_losses: list[list[float]]
    uninterrupted_state: dict[str, Any]
    uninterrupted_digest: str
    final_state: dict[str, Any]
    replay_digest: str
    physical_statuses: list[int]
    execution_baseline: Any
    runtime_statistics: Any
    spill_statistics: Any

    @property
    def planning_seconds(self) -> float:
        return sum(self.phase_seconds.values())


def run_planned_case(request: PlannedRequest, directory: Path) -> PlannedRun:
    plan = Plan.load(directory / "plan.ssp")
    inputs = load_inputs(directory)
    metadata = json.loads((directory / "case.json").read_text())
    recorded = json.loads((directory / "phases.json").read_text())
    phases = {key: recorded[key] for key in ("build", "profile", "plan")}
    losses, timings, compute, summaries, diagnostics, statuses = [], [], [], [], [], []
    checkpoint_path = directory / "checkpoint.ssp"
    with (
        deterministic_kernels(True),
        Executor(
            plan,
            external_headroom_bytes=request.external_headroom_mib << 20,
            spill=request.spill_pool,
            reject_overbudget=request.reject_overbudget,
        ) as executor,
    ):
        initialize(executor, inputs)
        observer = Observe(executor, steps=1)
        try:
            # Exercise all lazy execution setup, then restore every tensor/control.
            save_checkpoint(executor, checkpoint_path)
            warm, _, _ = observer.traced(inputs["args"], step=0)
            del warm
            executor.synchronize()
            load_checkpoint(executor, checkpoint_path)
            observer.idle()
            baseline = adapter_statistics()
            statuses.append(check_physical_budget())
            checkpoint = None
            for step in range(request.case.steps):
                result, summary, trace = observer.traced(inputs["args"], step=step + 1)
                losses.append([float(value) for value in result.losses])
                del result
                executor.synchronize()
                timings.append(summary["host_seconds"])
                compute.append(summary["gpu_seconds"])
                summaries.append(summary)
                if request.detailed_artifacts:
                    diagnostics.append(trace)
                statuses.append(check_physical_budget())
                print(
                    f"shadowspill step {step + 1}/{request.case.steps}: "
                    f"{timings[-1]:.3f}s",
                    flush=True,
                )
                if step + 1 == request.checkpoint_step:
                    save_checkpoint(
                        executor, checkpoint_path, metadata={"step": step + 1}
                    )
                    checkpoint = snapshot(executor, inputs)
            assert checkpoint is not None
            uninterrupted = snapshot(executor, inputs)
            restored = load_checkpoint(executor, checkpoint_path)
            assert restored["step"] == request.checkpoint_step
            replay = []
            for _step in range(request.checkpoint_step, request.case.steps):
                result = executor.run(args=inputs["args"]).result()
                replay.append([float(value) for value in result.losses])
                del result
                executor.synchronize()
                statuses.append(check_physical_budget())
            final = snapshot(executor, inputs)
            observer.idle()
            statistics = adapter_statistics()
            spill = executor.runtime.pool_statistics("spill")
        finally:
            observer.close()
    return PlannedRun(
        plan,
        phases,
        metadata["workload_metadata"],
        metadata["input_digest"],
        directory / "plan.ssp",
        losses,
        timings,
        compute,
        summaries,
        diagnostics,
        checkpoint,
        losses[request.checkpoint_step :],
        replay,
        uninterrupted,
        state_digest(uninterrupted),
        final,
        state_digest(final),
        statuses,
        baseline,
        statistics,
        spill,
    )
