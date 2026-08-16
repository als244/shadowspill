"""Aggregate comparison tables over independently committed point records."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .storage import BaselinePaths, atomic_json, atomic_text, read_object, utc_now

_CSV_FIELDS = (
    "case_id",
    "status",
    "family",
    "provider",
    "tokens_per_microbatch",
    "sequence_length",
    "accumulation_steps",
    "tokens_per_step",
    "execution_budget_bytes",
    "spill_budget_bytes",
    "bandwidth_scale_numerator",
    "bandwidth_scale_denominator",
    "fetch_bytes_per_second",
    "evict_bytes_per_second",
    "makespan_ns",
    "tokens_per_second",
    "total_selection_wall_time_ns",
    "pressurefit_wall_time_ns",
    "physical_admission_wall_time_ns",
    "orchestration_wall_time_ns",
    "task_duration_sum_ns",
    "compute_idle_ns",
    "fetch_bytes",
    "evict_bytes",
    "device_peak_bytes",
    "host_peak_bytes",
    "candidate_count",
    "valid_candidate_count",
    "action_count",
    "admission_refinement_count",
    "layout_required_bytes",
    "layout_slack_bytes",
    "plan_digest",
    "schedule_digest",
    "selection_digest",
    "option_counts_json",
    "annotated_plan_path",
    "error_type",
    "error_kind",
    "error_message",
    "point_path",
)


def write_frontier_summary(
    paths: BaselinePaths,
    *,
    expected_programs: int,
    expected_points_per_program: int,
    case_failures: dict[str, dict[str, object]] | None = None,
) -> dict[str, object]:
    """Regenerate JSONL, CSV, and aggregate counts from point source-of-truth."""

    point_paths = tuple(sorted(paths.cases_directory.glob("*/points/*/point.json")))
    points = tuple(read_object(path) for path in point_paths)
    rows = tuple(
        _csv_row(path, point) for path, point in zip(point_paths, points, strict=True)
    )
    atomic_text(
        paths.jsonl_path,
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in points),
    )
    _write_csv(paths.csv_path, rows)
    statuses = Counter(str(item.get("status", "invalid")) for item in points)
    transfer_pairs = sorted(
        {
            (
                int(_request(item)["fetch_bytes_per_second"]),
                int(_request(item)["evict_bytes_per_second"]),
            )
            for item in points
        }
    )
    expected_points = expected_programs * expected_points_per_program
    summary: dict[str, object] = {
        "schema": "shadowspill.pressurefit_frontier_summary/v1",
        "baseline_id": paths.directory.name,
        "updated_at": utc_now(),
        "expected_programs": expected_programs,
        "expected_points_per_program": expected_points_per_program,
        "expected_points": expected_points,
        "completed_points": len(points),
        "pending_points": expected_points - len(points),
        "status_counts": dict(sorted(statuses.items())),
        "observed_transfer_bandwidth_combinations": [
            {
                "fetch_bytes_per_second": fetch,
                "evict_bytes_per_second": evict,
            }
            for fetch, evict in transfer_pairs
        ],
        "case_failures": dict(sorted((case_failures or {}).items())),
        "artifacts": {
            "csv": str(paths.csv_path),
            "jsonl": str(paths.jsonl_path),
            "point_root": str(paths.cases_directory),
        },
    }
    atomic_json(paths.summary_path, summary)
    return summary


def _csv_row(path: Path, point: dict[str, Any]) -> dict[str, object]:
    case = _mapping(point.get("case"))
    identity = _mapping(case.get("identity"))
    request = _mapping(point.get("request"))
    budgets = _mapping(request.get("memory_budgets"))
    bandwidth = _mapping(request.get("transfer_bandwidths"))
    result = _mapping(point.get("result"))
    simulation = _mapping(result.get("simulation"))
    transfers = _mapping(simulation.get("transfers"))
    fetch = _mapping(transfers.get("fetch"))
    evict = _mapping(transfers.get("evict"))
    throughput = _mapping(result.get("throughput"))
    selection = _mapping(result.get("selection"))
    schedule = _mapping(result.get("schedule"))
    admission = _mapping(result.get("physical_admission"))
    plan = _mapping(result.get("annotated_plan"))
    timing = _mapping(point.get("timing"))
    error = _mapping(point.get("error"))
    error_fields = _mapping(error.get("fields"))
    return {
        "case_id": path.parents[2].name,
        "status": point.get("status"),
        "family": identity.get("family"),
        "provider": identity.get("provider"),
        "tokens_per_microbatch": identity.get("tokens_per_microbatch"),
        "sequence_length": identity.get("sequence_length"),
        "accumulation_steps": identity.get("accumulation_steps"),
        "tokens_per_step": throughput.get("tokens_per_step"),
        "execution_budget_bytes": budgets.get("execution_bytes"),
        "spill_budget_bytes": budgets.get("spill_bytes"),
        "bandwidth_scale_numerator": bandwidth.get("scale_numerator"),
        "bandwidth_scale_denominator": bandwidth.get("scale_denominator"),
        "fetch_bytes_per_second": bandwidth.get("fetch_bytes_per_second"),
        "evict_bytes_per_second": bandwidth.get("evict_bytes_per_second"),
        "makespan_ns": simulation.get("makespan_ns"),
        "tokens_per_second": throughput.get("tokens_per_second"),
        "total_selection_wall_time_ns": timing.get("total_selection_wall_time_ns"),
        "pressurefit_wall_time_ns": timing.get("pressurefit_wall_time_ns"),
        "physical_admission_wall_time_ns": timing.get(
            "physical_admission_wall_time_ns"
        ),
        "orchestration_wall_time_ns": timing.get("orchestration_wall_time_ns"),
        "task_duration_sum_ns": simulation.get("task_duration_sum_ns"),
        "compute_idle_ns": simulation.get("compute_idle_within_selected_span_ns"),
        "fetch_bytes": fetch.get("bytes"),
        "evict_bytes": evict.get("bytes"),
        "device_peak_bytes": _mapping(simulation.get("device_peak")).get(
            "total_bytes"
        ),
        "host_peak_bytes": simulation.get("host_peak_bytes"),
        "candidate_count": selection.get("candidate_count"),
        "valid_candidate_count": selection.get("valid_candidate_count"),
        "action_count": schedule.get("action_count"),
        "admission_refinement_count": len(
            _list(admission.get("admission_refinements"))
        ),
        "layout_required_bytes": admission.get("required_bytes"),
        "layout_slack_bytes": admission.get("slack_bytes"),
        "plan_digest": plan.get("plan_digest"),
        "schedule_digest": schedule.get("digest"),
        "selection_digest": selection.get("selection_digest"),
        "option_counts_json": json.dumps(
            _mapping(selection.get("option_counts")), sort_keys=True
        ),
        "annotated_plan_path": plan.get("path"),
        "error_type": error.get("type"),
        "error_kind": error_fields.get("kind"),
        "error_message": error.get("message"),
        "point_path": str(path),
    }


def _request(point: dict[str, Any]) -> dict[str, Any]:
    request = _mapping(point.get("request"))
    return _mapping(request.get("transfer_bandwidths"))


def _write_csv(path: Path, rows: tuple[dict[str, object], ...]) -> None:
    import io

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=_CSV_FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(path, buffer.getvalue())


def _mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: object) -> list[object]:
    return value if isinstance(value, list) else []


__all__ = ["write_frontier_summary"]
