"""Rank-local saved-Plan execution, with no model or preparation dependency."""

from __future__ import annotations

import json
import math
import os
import statistics
from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from orchestration.lifecycle import Lifecycle

from .monitor import GpuMonitor

if TYPE_CHECKING:
    from shadowspill import Plan


@dataclass(frozen=True)
class Context:
    """Workload hook context; ``rank`` is the Plan's logical rank."""

    plan: Plan
    rank: int
    worker_rank: int
    world_size: int


def validate(steps, warmup_steps, units_per_step, trace_final_step):
    if type(steps) is not int or steps < 1:
        raise ValueError("steps must be a positive integer")
    if type(warmup_steps) is not int or not 0 <= warmup_steps < steps:
        raise ValueError("warmup_steps must leave at least one measured update")
    if type(trace_final_step) is not bool:
        raise TypeError("trace_final_step must be a bool")
    if trace_final_step and steps <= warmup_steps + 1:
        raise ValueError(
            "final-step tracing must leave at least one untraced measured update"
        )
    if units_per_step is not None and (
        isinstance(units_per_step, bool)
        or not math.isfinite(units_per_step)
        or units_per_step <= 0
    ):
        raise ValueError("units_per_step must be finite and positive")


def run(
    plan,
    *,
    initializer,
    inputs,
    output_dir,
    steps=7,
    warmup_steps=1,
    trainer_options=None,
    trainer_type=None,
    rank_map=None,
    units_per_step=None,
    unit_label="items",
    monitor=True,
    trace_final_step=True,
):
    """Execute a saved training Plan in each participating fresh worker.

    ``initializer(context)`` is a context manager yielding Trainer's bounded
    ``fill_tensor(chunk, destination)`` callback. ``inputs(context)`` returns an
    iterable of Trainer.step keyword mappings. Both run after Executor installs
    its allocator. Trainer options may be a mapping or a context-taking factory.

    Launch distributed workers with torchrun and orchestration.rank_device; each
    process must see its assigned GPU as cuda:0. WORLD_SIZE must match the Plan.
    No model construction, build, profiling, planning, evaluation, checkpoints,
    or startup diagnostic runs are added here. By default the final real update
    collects a runtime trace on every rank. Warmup and traced updates still
    train, but are excluded from the reported steady median. Trace files are
    written after runtime cleanup; set ``trace_final_step=False`` to disable.
    """
    validate(steps, warmup_steps, units_per_step, trace_final_step)
    worker_rank = int(os.environ.get("RANK", "0"))
    root = Path(output_dir)
    local = root / f"rank-{worker_rank:05d}"
    local.mkdir(parents=True, exist_ok=False)
    with (
        Lifecycle(local / "phases.jsonl", rank=worker_rank) as timing,
        ExitStack() as stack,
    ):
        if monitor:
            # The rank_device launcher exposes exactly one physical GPU. A
            # standalone worker also uses visible ordinal zero.
            visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
            stack.enter_context(GpuMonitor(local, devices=visible))
        with timing.phase("execution"):
            return _execute(
                Path(plan),
                initializer,
                inputs,
                root,
                local,
                timing,
                steps,
                warmup_steps,
                trainer_options,
                trainer_type,
                rank_map,
                units_per_step,
                unit_label,
                trace_final_step,
            )


def _execute(
    path,
    initializer,
    inputs,
    root,
    local,
    timing,
    steps,
    warmup_steps,
    trainer_options,
    trainer_type,
    rank_map,
    units_per_step,
    unit_label,
    trace_final_step,
):
    with timing.phase("imports"):
        from orchestration.workers import workers
        from shadowspill import Plan, Trainer

    with timing.phase("plans.load"):
        plan = Plan.load(path)
        if plan.program.bindings.get("kind") != "step":
            raise ValueError("plan_runner requires a saved training Plan (build_step)")
        count = (
            1
            if plan.program.parallelism is None
            else plan.program.parallelism.world_size
        )
    records = []
    diagnostics = None

    def completed(owner, result):
        nonlocal diagnostics
        if not math.isfinite(result.loss) or not math.isfinite(result.seconds):
            raise FloatingPointError("training returned a nonfinite loss or duration")
        timing.event(
            "training.step_completed", step=result.step, seconds=result.seconds
        )
        records.append(
            dict(
                step=result.step,
                seconds=result.seconds,
                loss=result.loss,
                scalars=dict(result.summary.scalars),
            )
        )
        if trace_final_step and result.step == steps:
            diagnostics = result.diagnostics
            if diagnostics is None:
                raise RuntimeError(
                    "final update did not return the requested runtime trace"
                )

    with timing.managed("control_group", workers(count, rank_map=rank_map)) as group:
        context = Context(plan, group.logical_rank, group.rank, group.count)
        with timing.phase("configuration"):
            options = (
                trainer_options(context)
                if callable(trainer_options)
                else trainer_options
            )
            options = dict(options or {})
            forbidden = {
                "executor",
                "evaluation_plan",
                "rank_map",
                "device",
            } & options.keys()
            if forbidden:
                raise ValueError(
                    f"runner owns these Trainer options: {sorted(forbidden)}"
                )
        with ExitStack() as runtime:
            with timing.phase("runtime.create"):
                trainer = (trainer_type or Trainer)(
                    plan, rank_map=group.rank_map, **options
                )
                runtime.enter_context(timing.managed("runtime", trainer))
            with timing.phase("weights.initialize"), initializer(context) as initialize:
                trainer.initialize(initialize)
            with timing.phase("inputs.create"):
                source = inputs(context)
            with timing.phase("training.fit"):
                trainer.fit(
                    _traced_inputs(source, steps, trace_final_step),
                    steps=steps,
                    run_dir=local,
                    callbacks=[completed],
                    checkpoint_every=0,
                    tables_every=0,
                    startup_diagnostics=False,
                )
            capacity = trainer.executor.spill_capacity_bytes
        trace = None
        if diagnostics is not None:
            with timing.phase("trace.write"):
                from shadowspill.diagnostics.step_file import write_step_diagnostics

                trace_path = local / "trace" / "step.json"
                trace_path.parent.mkdir()
                events = write_step_diagnostics(trace_path, diagnostics)
                trace = dict(
                    step=steps,
                    diagnostics_path=str(trace_path.relative_to(root)),
                    events_path=str(events.relative_to(root)),
                    runtime_event_overflow=diagnostics.runtime.event_overflow,
                    allocation_event_overflow=diagnostics.runtime.allocation_event_overflow,
                    allocator_overflow=diagnostics.allocator.overflow,
                )
            del diagnostics
        # Gather only after runtime teardown, so cleanup cannot hide behind a
        # successful result file. Small host records never retain GPU outputs.
        with timing.phase("results.gather"):
            per_rank = group.gather(
                dict(
                    rank=group.logical_rank,
                    worker_rank=group.rank,
                    steps=records,
                    spill_capacity_bytes=capacity,
                    trace=trace,
                )
            )
    if group.rank != 0:
        return None
    with timing.phase("results.write"):
        result = summarize(
            per_rank,
            steps=steps,
            warmup_steps=warmup_steps,
            units_per_step=units_per_step,
            unit_label=unit_label,
            trace_final_step=trace_final_step,
        )
        result.update(
            plan=str(path.resolve()),
            program_digest=plan.program.digest,
            schedule_digest=plan.schedule.digest,
            predicted_seconds=plan.timeline.makespan_ns / 1e9,
            execution_budget_bytes=plan.memory_budgets.execution_bytes,
            spill_budget_bytes=plan.memory_budgets.spill_bytes,
        )
        result["measured_over_predicted"] = (
            result["median_seconds"] / result["predicted_seconds"]
            if result["predicted_seconds"] > 0
            else None
        )
        (root / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def _traced_inputs(source, steps, trace_final_step):
    for index, data in enumerate(source, 1):
        if not isinstance(data, Mapping):
            raise TypeError(
                "each source item must contain keyword arguments for Trainer.step"
            )
        if "runtime_trace" in data:
            raise ValueError("plan_runner owns runtime_trace; use trace_final_step")
        yield dict(data, runtime_trace=trace_final_step and index == steps)


def summarize(
    per_rank, *, steps, warmup_steps, units_per_step, unit_label, trace_final_step=False
):
    """Compare synchronous updates by max-rank time before taking the median."""
    validate(steps, warmup_steps, units_per_step, trace_final_step)
    per_rank = sorted(per_rank, key=lambda row: row["rank"])
    if not per_rank or len({row["rank"] for row in per_rank}) != len(per_rank):
        raise ValueError("expected one result for every distinct logical rank")
    expected = list(range(1, steps + 1))
    if any([r["step"] for r in rank["steps"]] != expected for rank in per_rank):
        raise ValueError("workers did not complete the requested updates")
    combined = [
        dict(step=index + 1, seconds=max(row["seconds"] for row in rows))
        for index, rows in enumerate(
            zip(*(rank["steps"] for rank in per_rank), strict=True)
        )
    ]
    steady = combined[warmup_steps : steps - int(trace_final_step)]
    median = statistics.median(row["seconds"] for row in steady)
    return dict(
        status="PASS",
        steps=combined,
        per_rank=per_rank,
        warmup_steps=warmup_steps,
        trace_step=steps if trace_final_step else None,
        steady_steps=[row["step"] for row in steady],
        median_seconds=median,
        units_per_step=units_per_step,
        unit_label=unit_label,
        units_per_second=None if units_per_step is None else units_per_step / median,
        timing_definition=(
            "median of per-update maximum rank Trainer.step seconds, "
            "excluding warmup and traced updates"
        ),
    )
