"""Reconcile ShadowSpill's execution-ordered frontend task boundaries.

This tool consumes either a standalone ``StepDiagnostics`` JSON object or a
qualification artifact containing one. It deliberately reports host boundary
work separately from compute-stream gaps: CUDA work launched by postprocessing
(for example gradient accumulation) can occur between two task timing events
and is not necessarily idle time.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

_BEFORE_COMPONENTS = (
    "dispatch_stream_resolution_seconds",
    "dispatch_readiness_marker_seconds",
    "dispatch_input_lookup_seconds",
    "dispatch_storage_rebind_seconds",
    "dispatch_argument_assembly_seconds",
)
_AFTER_COMPONENTS = (
    "dispatch_output_flatten_seconds",
    "dispatch_output_publish_seconds",
    "dispatch_dematerialize_seconds",
    "dispatch_cleanup_seconds",
)
_OUTPUT_SUBCOMPONENTS = (
    "dispatch_output_classification_seconds",
    "dispatch_output_adoption_seconds",
    "dispatch_output_state_publish_seconds",
)
_ALL_COMPONENTS = (*_BEFORE_COMPONENTS, "dispatch_invoke_seconds", *_AFTER_COMPONENTS)


def _diagnostics(payload: Mapping[str, Any], sample: int) -> Mapping[str, Any]:
    if payload.get("schema") == "shadowspill.step_diagnostics/v4":
        return payload
    trace = payload.get("trace")
    if isinstance(trace, Mapping):
        return trace
    samples = payload.get("planned_step_diagnostics")
    if isinstance(samples, list) and samples:
        try:
            selected = samples[sample]
        except IndexError as exc:
            raise ValueError(
                f"sample index {sample} is outside {len(samples)} diagnostics"
            ) from exc
        if isinstance(selected, Mapping):
            return selected
    raise ValueError("input contains no ShadowSpill StepDiagnostics object")


def _seconds(task: Mapping[str, Any], key: str) -> float:
    value = task.get(key, 0.0)
    return float(value) if value is not None else 0.0


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(fraction * len(ordered)))
    return ordered[index]


def _summary(values: Iterable[float]) -> dict[str, float | int]:
    samples = list(values)
    return {
        "count": len(samples),
        "total_seconds": sum(samples),
        "median_seconds": statistics.median(samples) if samples else 0.0,
        "p95_seconds": _percentile(samples, 0.95),
        "maximum_seconds": max(samples, default=0.0),
    }


def _dispatch_timestamp(task: Mapping[str, Any], boundary: str, edge: str) -> int:
    timestamps = task["boundary_timestamps"]
    return int(timestamps["host"][boundary][edge])


def analyze(payload: Mapping[str, Any], *, sample: int = -1) -> dict[str, object]:
    """Return complete task and transition attribution in execution order."""

    diagnostics = _diagnostics(payload, sample)
    task_mapping = diagnostics.get("tasks")
    if not isinstance(task_mapping, Mapping):
        raise ValueError("StepDiagnostics has no task mapping")
    tasks = sorted(
        (dict(value) for value in task_mapping.values()),
        key=lambda task: int(task["execution_ordinal"]),
    )

    by_phase: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        by_phase[str(task["phase"])].append(task)
    phase_components = {
        phase: {
            key: _summary(_seconds(task, key) for task in selected)
            for key in (
                "dispatch_before_task_seconds",
                *_BEFORE_COMPONENTS,
                "dispatch_invoke_seconds",
                "dispatch_after_task_seconds",
                *_AFTER_COMPONENTS,
                *_OUTPUT_SUBCOMPONENTS,
            )
        }
        for phase, selected in sorted(by_phase.items())
    }

    task_rows: list[dict[str, object]] = []
    for task in tasks:
        before_accounted = sum(_seconds(task, key) for key in _BEFORE_COMPONENTS)
        after_accounted = sum(_seconds(task, key) for key in _AFTER_COMPONENTS)
        task_rows.append(
            {
                "execution_task_id": task["execution_task_id"],
                "semantic_name": task["semantic_name"],
                "phase": task["phase"],
                "microbatch": task.get("microbatch"),
                "expected_profile_seconds": task["expected_profile_seconds"],
                "compute_duration_seconds": task["compute_duration_seconds"],
                "dispatch_before_task_seconds": task["dispatch_before_task_seconds"],
                "dispatch_before_accounted_seconds": before_accounted,
                "dispatch_before_unattributed_seconds": max(
                    0.0,
                    _seconds(task, "dispatch_before_task_seconds") - before_accounted,
                ),
                "dispatch_invoke_seconds": task["dispatch_invoke_seconds"],
                "dispatch_after_task_seconds": task["dispatch_after_task_seconds"],
                "dispatch_after_accounted_seconds": after_accounted,
                "dispatch_after_unattributed_seconds": max(
                    0.0,
                    _seconds(task, "dispatch_after_task_seconds") - after_accounted,
                ),
                "components": {
                    key: _seconds(task, key)
                    for key in (*_ALL_COMPONENTS, *_OUTPUT_SUBCOMPONENTS)
                },
            }
        )

    transitions: list[dict[str, object]] = []
    for previous, current in pairwise(tasks):
        outer_gap = (
            max(
                0,
                _dispatch_timestamp(current, "before_task", "enter")
                - _dispatch_timestamp(previous, "after_task", "exit"),
            )
            / 1e9
        )
        prior_after = _seconds(previous, "dispatch_after_task_seconds")
        next_before = _seconds(current, "dispatch_before_task_seconds")
        dispatch_boundary = prior_after + outer_gap + next_before
        compute_gap = max(
            0.0,
            _seconds(current, "compute_started_seconds")
            - _seconds(previous, "compute_finished_seconds"),
        )
        previous_components = {
            key: _seconds(previous, key) for key in _AFTER_COMPONENTS
        }
        current_components = {key: _seconds(current, key) for key in _BEFORE_COMPONENTS}
        accounted = sum(previous_components.values()) + sum(current_components.values())
        transitions.append(
            {
                "previous_execution_task_id": previous["execution_task_id"],
                "previous_semantic_name": previous["semantic_name"],
                "previous_phase": previous["phase"],
                "next_execution_task_id": current["execution_task_id"],
                "next_semantic_name": current["semantic_name"],
                "next_phase": current["phase"],
                "dispatch_boundary_seconds": dispatch_boundary,
                "dispatch_accounted_seconds": accounted,
                "dispatch_unattributed_seconds": max(
                    0.0, dispatch_boundary - accounted
                ),
                "dispatch_outer_loop_gap_seconds": outer_gap,
                "compute_stream_gap_seconds": compute_gap,
                "next_input_readiness_wait_seconds": _seconds(
                    current, "input_readiness_wait_seconds"
                ),
                "next_allocation_reuse_wait_seconds": _seconds(
                    current, "allocation_reuse_wait_seconds"
                ),
                "previous_after_components": previous_components,
                "next_before_components": current_components,
            }
        )

    timing = diagnostics.get("timing", {})
    return {
        "schema": "shadowspill.inter_task_analysis/v1",
        "note": (
            "compute_stream_gap_seconds includes CUDA work launched during "
            "frontend postprocessing and must be reconciled with a CUDA/NSYS "
            "timeline before it is classified as idle"
        ),
        "task_count": len(tasks),
        "selected_task_span_seconds": float(timing.get("compute_seconds", 0.0)),
        "task_interval_sum_seconds": sum(
            _seconds(task, "compute_duration_seconds") for task in tasks
        ),
        "dispatch_boundary_sum_seconds": sum(
            _seconds(task, "dispatch_before_task_seconds")
            + _seconds(task, "dispatch_after_task_seconds")
            for task in tasks
        ),
        "phase_components": phase_components,
        "tasks": task_rows,
        "transitions": transitions,
        "largest_spill_boundaries": sorted(
            transitions,
            key=lambda row: cast(float, row["dispatch_boundary_seconds"]),
            reverse=True,
        )[:20],
        "largest_compute_stream_gaps": sorted(
            transitions,
            key=lambda row: cast(float, row["compute_stream_gap_seconds"]),
            reverse=True,
        )[:20],
    }


def _print_summary(result: Mapping[str, Any]) -> None:
    print(
        "task span={:.3f} ms, task sum={:.3f} ms, host boundaries={:.3f} ms".format(
            1e3 * float(result["selected_task_span_seconds"]),
            1e3 * float(result["task_interval_sum_seconds"]),
            1e3 * float(result["dispatch_boundary_sum_seconds"]),
        )
    )
    print("largest host boundaries:")
    for row in result["largest_spill_boundaries"][:10]:
        print(
            "  {} -> {}: {:.1f} us host, {:.1f} us stream gap; {}".format(
                row["previous_execution_task_id"],
                row["next_execution_task_id"],
                1e6 * float(row["dispatch_boundary_seconds"]),
                1e6 * float(row["compute_stream_gap_seconds"]),
                row["next_semantic_name"],
            )
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Attribute ShadowSpill frontend task-boundary overhead."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--sample", type=int, default=-1)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    payload = json.loads(arguments.input.read_text())
    result = analyze(payload, sample=arguments.sample)
    _print_summary(result)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
