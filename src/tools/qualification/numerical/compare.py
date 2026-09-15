"""Two comparisons: against the compiled reference, and against the replay.

The reference decides whether the planned arm computed the right answer; the
replay decides whether the arm reproduces itself from a checkpoint. Both are
per-tensor and hold to the same tolerance, because a step is only bitwise
reproducible if every kernel under it is, and not every accelerator's are.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch

from .metrics import compare_states
from .references import REFERENCE_SCHEMA
from .request import REFERENCE_EXECUTION, PlannedRequest
from .run import PlannedRun
from .tolerances import (
    LOSS_ABSOLUTE_TOLERANCE,
    LOSS_RELATIVE_TOLERANCE,
    meets_tensor_tolerance,
)


@dataclass(frozen=True, slots=True)
class Comparison:
    """What the planned state and losses agreed with, and where they did not."""

    reference: dict[str, Any]
    tensor_results: dict[str, Any]
    exact_failures: tuple[str, ...]
    structure_failures: tuple[str, ...]
    metric_failures: list[str]
    loss_failures: list[str]
    worst_loss_relative: float
    replay_results: dict[str, Any]
    replay_exact_failures: tuple[str, ...]
    replay_structure_failures: tuple[str, ...]
    replay_metric_failures: list[str]


def compare_planned_run(request: PlannedRequest, run: PlannedRun) -> Comparison:
    """Load the reference, refuse a mismatched one, and compare both ways."""

    case_name = f"{request.case.model_implementation}/{request.case.family}"
    stage_started = time.perf_counter()
    reference = torch.load(
        request.reference_path, map_location="cpu", weights_only=True
    )
    print(
        f"shadowspill {case_name} reference loaded from "
        f"{request.reference_path.resolve()}: "
        f"{time.perf_counter() - stage_started:.3f}s",
        flush=True,
    )
    if (
        reference.get("schema") != REFERENCE_SCHEMA
        or reference.get("reference_execution") != REFERENCE_EXECUTION
        or reference.get("family") != request.case.family
        or reference.get("model_implementation") != request.case.model_implementation
        or reference.get("case_identity") != request.case.identity()
        or (
            reference.get("schema") == REFERENCE_SCHEMA
            and reference.get("microbatch_digest") != run.requested_input_digest
        )
    ):
        raise RuntimeError(
            "compiled reference identity differs from requested qualification; "
            "replace it with --regenerate-reference"
        )
    stage_started = time.perf_counter()
    print(
        f"shadowspill {case_name} comparing model and "
        "optimizer state against the reference, tensor by tensor",
        flush=True,
    )
    tensor_results, exact_failures, structure_failures = compare_states(
        {"model": reference["model"], "optimizer": reference["optimizer"]},
        {"model": run.final_state["model"], "optimizer": run.final_state["optimizer"]},
    )
    print(
        f"shadowspill {case_name} compared "
        f"{len(tensor_results)} tensors: "
        f"{time.perf_counter() - stage_started:.3f}s",
        flush=True,
    )
    loss_failures: list[str] = []
    worst_loss_relative = 0.0
    for step, (expected_step, actual_step) in enumerate(
        zip(reference["losses"], run.losses, strict=True), start=1
    ):
        for microbatch, (expected, actual) in enumerate(
            zip(expected_step, actual_step, strict=True), start=1
        ):
            relative = abs(actual - expected) / max(abs(expected), 1e-30)
            worst_loss_relative = max(worst_loss_relative, relative)
            if abs(actual - expected) > (
                LOSS_ABSOLUTE_TOLERANCE + LOSS_RELATIVE_TOLERANCE * abs(expected)
            ):
                loss_failures.append(
                    f"step {step} microbatch {microbatch}: "
                    f"expected={expected}, actual={actual}"
                )
    metric_failures = [
        name
        for name, metric in tensor_results.items()
        if not meets_tensor_tolerance(metric, key=name)
    ]
    # The replayed run has to agree with the uninterrupted one, but it cannot
    # be required to agree bit for bit: a step is only bitwise reproducible if
    # every kernel under it is, and the mlops path's are not on every
    # accelerator. Hold the replay to the same per-tensor tolerance the
    # reference comparison uses, and keep the bitwise answer as evidence.
    stage_started = time.perf_counter()
    print(
        f"shadowspill {case_name} comparing the "
        "checkpoint-replayed state against the uninterrupted run, tensor by "
        "tensor",
        flush=True,
    )
    replay_results, replay_exact_failures, replay_structure_failures = compare_states(
        {
            "model": run.uninterrupted_state["model"],
            "optimizer": run.uninterrupted_state["optimizer"],
        },
        {
            "model": run.final_state["model"],
            "optimizer": run.final_state["optimizer"],
        },
    )
    print(
        f"shadowspill {case_name} compared "
        f"{len(replay_results)} replayed tensors: "
        f"{time.perf_counter() - stage_started:.3f}s",
        flush=True,
    )
    replay_metric_failures = [
        name
        for name, metric in replay_results.items()
        if not meets_tensor_tolerance(metric, key=name)
    ]

    return Comparison(
        reference=reference,
        tensor_results=tensor_results,
        exact_failures=exact_failures,
        structure_failures=structure_failures,
        metric_failures=metric_failures,
        loss_failures=loss_failures,
        worst_loss_relative=worst_loss_relative,
        replay_results=replay_results,
        replay_exact_failures=replay_exact_failures,
        replay_structure_failures=replay_structure_failures,
        replay_metric_failures=replay_metric_failures,
    )
