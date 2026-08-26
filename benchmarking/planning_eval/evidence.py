"""Compact comparison records derived from complete annotated plans."""

from __future__ import annotations

import hashlib
import json
import traceback
from collections import Counter
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from benchmarking.program_collection.corpus import SavedProgramCase
from shadowspill.ir import MemoryActionKind
from shadowspill.planner.program import (
    AnnotatedProgramPlan,
)
from shadowspill.simulator import TransferDirection

from .matrix import FrontierPointRequest
from .source import CorpusProgramCase


def successful_point_evidence(
    *,
    case: SavedProgramCase,
    request: FrontierPointRequest,
    plan: AnnotatedProgramPlan,
    annotated_plan_directory: Path,
    started_at: str,
    completed_at: str,
    elapsed_seconds: float,
) -> dict[str, object]:
    """Build the small row used for frontier comparisons."""

    simulation = plan.simulation
    task_duration = sum(
        item.end_ns - item.start_ns for item in simulation.task_intervals
    )
    task_stall = sum(item.stall_ns for item in simulation.task_intervals)
    if simulation.task_intervals:
        task_span = (
            max(item.end_ns for item in simulation.task_intervals)
            - min(item.start_ns for item in simulation.task_intervals)
        )
    else:
        task_span = 0
    transfer_summary = {
        direction.value: _transfer_summary(plan, direction)
        for direction in TransferDirection
    }
    alias_sizes = {
        item.alias_group_id: item.size_bytes
        for item in plan.program.program.alias_groups
    }
    actions = plan.result.schedule.actions
    action_counts = Counter(item.kind.value for item in actions)
    action_bytes = Counter(
        {
            kind.value: sum(
                alias_sizes[item.alias_group_id]
                for item in actions
                if item.kind is kind
            )
            for kind in MemoryActionKind
        }
    )
    selection_value = [item.to_dict() for item in plan.result.selections]
    selection_digest = _digest(selection_value)
    option_counts = Counter(item.option_id for item in plan.result.selections)
    pressurefit_diagnostics = plan.result.diagnostics
    peak = simulation.device_peaks[0]
    manifest = json.loads((annotated_plan_directory / "manifest.json").read_text())
    artifact = manifest["annotated_program_plan"]
    tokens_per_optimizer_step = case.identity.data_geometry.tokens_per_optimizer_step
    return {
        "case": {
            "directory": str(case.directory),
            "identity": case.identity.to_dict(),
            "data_geometry": case.identity.data_geometry.to_dict(),
            "step_program_digest": case.program_digest,
            "pressurefit_program_digest": request.program_digest,
        },
        "request": request.to_dict(),
        "timing": {
            "started_at": started_at,
            "completed_at": completed_at,
            "attempt_elapsed_seconds": elapsed_seconds,
            "total_selection_wall_time_ns": plan.wall_time_ns,
            "pressurefit_wall_time_ns": plan.pressurefit_wall_time_ns,
            "physical_admission_wall_time_ns": (
                plan.physical_admission_wall_time_ns
            ),
            "orchestration_wall_time_ns": plan.orchestration_wall_time_ns,
        },
        "result": {
            "annotated_plan": {
                "directory": str(annotated_plan_directory),
                "path": str(
                    annotated_plan_directory / "annotated_program_plan.json"
                ),
                "plan_digest": plan.digest,
                "artifact_sha256": artifact["artifact_sha256"],
            },
            "throughput": {
                "tokens_per_optimizer_step": tokens_per_optimizer_step,
                "tokens_per_second": (
                    tokens_per_optimizer_step
                    * 1_000_000_000
                    / simulation.makespan_ns
                ),
            },
            "simulation": {
                "makespan_ns": simulation.makespan_ns,
                "task_duration_sum_ns": task_duration,
                "task_ready_stall_sum_ns": task_stall,
                "selected_task_span_ns": task_span,
                "compute_idle_within_selected_span_ns": max(
                    0, task_span - task_duration
                ),
                "transfers": transfer_summary,
                "device_peak": asdict(peak),
                "spill_peak_bytes": simulation.spill_peak_bytes,
            },
            "selection": {
                "selection_digest": selection_digest,
                "option_counts": dict(sorted(option_counts.items())),
                "selections": selection_value,
                "selected_candidate_id": (
                    pressurefit_diagnostics.selected_candidate_id
                ),
                "selected_selection_id": (
                    pressurefit_diagnostics.selected_selection_id
                ),
                "recomputation_problem_count": (
                    pressurefit_diagnostics.recomputation_problem_count
                ),
                "valid_recomputation_problem_count": (
                    pressurefit_diagnostics.valid_recomputation_problem_count
                ),
                "candidate_policy_count": (
                    pressurefit_diagnostics.candidate_policy_count
                ),
                "candidate_evaluation_count": (
                    pressurefit_diagnostics.candidate_evaluation_count
                ),
                "valid_candidate_evaluation_count": (
                    pressurefit_diagnostics.valid_candidate_evaluation_count
                ),
                "candidate_status_counts": (
                    pressurefit_diagnostics.candidate_status_counts
                ),
                "pressurefit_diagnostics": pressurefit_diagnostics.to_dict(),
            },
            "schedule": {
                "digest": plan.result.schedule.digest,
                "action_count": len(actions),
                "action_counts": dict(sorted(action_counts.items())),
                "action_bytes": dict(sorted(action_bytes.items())),
            },
            "physical_admission": {
                "facts_digest": plan.effective_facts.digest,
                "layout_digest": plan.fixed_layout.digest,
                "pool_capacity_bytes": plan.fixed_layout.pool_capacity_bytes,
                "required_bytes": plan.fixed_layout.required_bytes,
                "slack_bytes": plan.fixed_layout.slack_bytes,
                "fixed_slice_bytes": plan.fixed_layout.fixed_slice_bytes,
                "dynamic_reserve_bytes": plan.fixed_layout.dynamic_reserve_bytes,
                "scratch_reserve_bytes": plan.fixed_layout.scratch_reserve_bytes,
                "reuse_dependency_count": len(
                    plan.fixed_layout.reuse_dependencies
                ),
                "attempts": [
                    {
                        "requested_object_capacity_bytes": (
                            item.requested_object_capacity_bytes
                        ),
                        "effective_object_capacity_bytes": (
                            item.effective_object_capacity_bytes
                        ),
                        "required_bytes": item.required_bytes,
                        "pool_capacity_bytes": item.pool_capacity_bytes,
                        "accepted": item.accepted,
                        "pressurefit_wall_time_ns": (
                            item.pressurefit_wall_time_ns
                        ),
                        "physical_admission_wall_time_ns": (
                            item.physical_admission_wall_time_ns
                        ),
                        "pressurefit_diagnostics": (
                            None
                            if item.pressurefit_diagnostics is None
                            else item.pressurefit_diagnostics.to_dict()
                        ),
                    }
                    for item in plan.attempts
                ],
                "effective_object_capacity_bytes": (
                    plan.result.diagnostics.effective_object_capacity_bytes
                ),
            },
        },
    }


def failed_point_evidence(
    *,
    case: CorpusProgramCase,
    request: FrontierPointRequest,
    error: BaseException,
    started_at: str,
    completed_at: str,
    elapsed_seconds: float,
) -> dict[str, object]:
    return {
        "case": case.to_dict(),
        "request": request.to_dict(),
        "timing": {
            "started_at": started_at,
            "completed_at": completed_at,
            "attempt_elapsed_seconds": elapsed_seconds,
        },
        "error": exception_details(error),
    }


def exception_details(error: BaseException) -> dict[str, object]:
    """Preserve machine-readable planner fields and the complete traceback."""

    fields: dict[str, object] = {}
    for name in (
        "kind",
        "device_id",
        "boundary_task_id",
        "task_id",
        "required_bytes",
        "capacity_bytes",
        "requested_bytes",
        "used_bytes",
        "time_ns",
        "location",
        "alias_group_ids",
        "diagnostics",
    ):
        if hasattr(error, name):
            fields[name] = _json_value(getattr(error, name))
    return {
        "type": type(error).__name__,
        "module": type(error).__module__,
        "message": str(error),
        "notes": list(getattr(error, "__notes__", ())),
        "fields": fields,
        "traceback": "".join(traceback.format_exception(error)),
    }


def _transfer_summary(
    plan: AnnotatedProgramPlan,
    direction: TransferDirection,
) -> dict[str, int]:
    intervals = tuple(
        item
        for item in plan.simulation.transfer_intervals
        if item.direction is direction
    )
    return {
        "count": len(intervals),
        "bytes": sum(item.bytes for item in intervals),
        "busy_ns": sum(item.end_ns - item.start_ns for item in intervals),
        "ready_stall_ns": sum(item.stall_ns for item in intervals),
    }


def _digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return _json_value(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return repr(value)


__all__ = [
    "exception_details",
    "failed_point_evidence",
    "successful_point_evidence",
]
