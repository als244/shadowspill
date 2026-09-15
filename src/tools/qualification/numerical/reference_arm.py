"""The reference arm: one step, fully compiled, without ShadowSpill.

This is the numerical authority the planned arm is compared against, so it
trains at the same rate and records the same losses, the same state, and the
per-phase device timings a comparison of speed needs.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from shadowspill.pytorch.accelerator import DEVICE_TYPE
from workloads.common.training import LEARNING_RATE

from .metrics import cpu_state, state_digest
from .references import REFERENCE_SCHEMA, reference_inputs_path
from .request import REFERENCE_EXECUTION, CaseRequest


def device_microbatches(values: list[list[Any]]) -> list[list[Any]]:
    """Move every tensor in the case's microbatches onto the device."""

    return [
        [
            item.to(DEVICE_TYPE) if isinstance(item, torch.Tensor) else item
            for item in microbatch
        ]
        for microbatch in values
    ]


def reference_worker(request: CaseRequest, output: Path) -> None:
    """Train the reference arm and write its artifact and inputs."""

    case = request.build()
    model = case.model.to(DEVICE_TYPE)
    microbatches = device_microbatches(case.microbatches)
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
        for step in range(request.steps):
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
                f"reference {request.model_implementation}/{request.family} "
                f"step {step + 1}/{request.steps}: {elapsed:.3f}s",
                flush=True,
            )
    artifact = {
        "schema": REFERENCE_SCHEMA,
        "reference_execution": REFERENCE_EXECUTION,
        "family": request.family,
        "model_implementation": request.model_implementation,
        "case_identity": request.identity(),
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
