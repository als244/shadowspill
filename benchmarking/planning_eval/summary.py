"""Aggregate comparison tables over independently committed point records."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .matrix import FrontierPointRequest
from .storage import BaselinePaths, atomic_json, atomic_text, read_object, utc_now

_CSV_FIELDS = (
    "case_id",
    "status",
    "family",
    "provider",
    "tokens_per_microbatch",
    "sequence_length",
    "accumulation_rounds",
    "tokens_per_optimizer_step",
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
    "spill_peak_bytes",
    "recomputation_problem_count",
    "valid_recomputation_problem_count",
    "candidate_policy_count",
    "candidate_evaluation_count",
    "valid_candidate_evaluation_count",
    "repair_attempt_count",
    "pressure_boundary_repair_attempt_count",
    "residency_evaluations",
    "residency_cache_hits",
    "schedule_emissions",
    "schedule_cache_hits",
    "simulation_requests",
    "simulation_search_calls",
    "simulation_cache_hits",
    "simulation_result_materialization_calls",
    "simulation_total_calls",
    "admission_search_calls",
    "admission_result_materialization_calls",
    "admission_total_calls",
    "pressurefit_problem_work_time_ns",
    "residency_work_time_ns",
    "schedule_work_time_ns",
    "simulation_work_time_ns",
    "admission_work_time_ns",
    "digest_work_time_ns",
    "action_count",
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
    requests = tuple(
        _point_request(path, point)
        for path, point in zip(point_paths, points, strict=True)
    )
    rows = tuple(
        _csv_row(path, point, request)
        for path, point, request in zip(
            point_paths, points, requests, strict=True
        )
    )
    atomic_text(
        paths.jsonl_path,
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in points),
    )
    _write_csv(paths.csv_path, rows)
    statuses = Counter(str(item.get("status", "invalid")) for item in points)
    transfer_pairs = sorted(
        {
            (_bandwidth(request, "fetch"), _bandwidth(request, "evict"))
            for request in requests
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


def _csv_row(
    path: Path,
    point: dict[str, Any],
    request: dict[str, Any],
) -> dict[str, object]:
    case = _point_case(path, point)
    identity = _mapping(case.get("identity"))
    geometry = _mapping(case.get("data_geometry"))
    budgets = _mapping(request.get("memory_budgets"))
    bandwidth = _mapping(request.get("transfer_bandwidths"))
    result = _mapping(point.get("result"))
    simulation = _mapping(result.get("simulation"))
    transfers = _mapping(simulation.get("transfers"))
    fetch = _mapping(transfers.get("fetch"))
    evict = _mapping(transfers.get("evict"))
    throughput = _mapping(result.get("throughput"))
    selection = _mapping(result.get("selection"))
    pressurefit = _mapping(selection.get("pressurefit_diagnostics"))
    pressurefit_summary = _mapping(pressurefit.get("summary"))
    pressurefit_repairs = _mapping(pressurefit.get("repairs"))
    pressurefit_work = _mapping(pressurefit.get("work"))
    evaluation_work = _mapping(pressurefit_work.get("evaluation"))
    residency_work = _mapping(pressurefit_work.get("residency"))
    schedule_work = _mapping(pressurefit_work.get("schedule"))
    simulation_work = _mapping(pressurefit_work.get("simulation"))
    admission_work = _mapping(pressurefit_work.get("admission"))
    digest_work = _mapping(pressurefit_work.get("digest"))
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
        "tokens_per_microbatch": geometry.get(
            "tokens_per_microbatch", identity.get("tokens_per_microbatch")
        ),
        "sequence_length": geometry.get(
            "sequence_length", identity.get("sequence_length")
        ),
        "accumulation_rounds": geometry.get(
            "accumulation_rounds",
            identity.get("accumulation_rounds", identity.get("accumulation_steps")),
        ),
        "tokens_per_optimizer_step": throughput.get(
            "tokens_per_optimizer_step", throughput.get("tokens_per_step")
        ),
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
        "spill_peak_bytes": simulation.get("spill_peak_bytes"),
        "recomputation_problem_count": pressurefit_summary.get(
            "recomputation_problem_count"
        ),
        "valid_recomputation_problem_count": pressurefit_summary.get(
            "valid_recomputation_problem_count"
        ),
        "candidate_policy_count": pressurefit_summary.get(
            "candidate_policy_count"
        ),
        "candidate_evaluation_count": pressurefit_summary.get(
            "candidate_evaluation_count"
        ),
        "valid_candidate_evaluation_count": pressurefit_summary.get(
            "valid_candidate_evaluation_count"
        ),
        "repair_attempt_count": pressurefit_repairs.get("total_attempts"),
        "pressure_boundary_repair_attempt_count": pressurefit_repairs.get(
            "pressure_boundary_attempts"
        ),
        "residency_evaluations": residency_work.get("evaluations"),
        "residency_cache_hits": residency_work.get("cache_hits"),
        "schedule_emissions": schedule_work.get("emissions"),
        "schedule_cache_hits": schedule_work.get("cache_hits"),
        "simulation_requests": simulation_work.get("requests"),
        "simulation_search_calls": simulation_work.get("search_calls"),
        "simulation_cache_hits": simulation_work.get("cache_hits"),
        "simulation_result_materialization_calls": simulation_work.get(
            "result_materialization_calls"
        ),
        "simulation_total_calls": simulation_work.get("total_calls"),
        "admission_search_calls": admission_work.get("search_calls"),
        "admission_result_materialization_calls": admission_work.get(
            "result_materialization_calls"
        ),
        "admission_total_calls": admission_work.get("total_calls"),
        "pressurefit_problem_work_time_ns": evaluation_work.get(
            "summed_wall_time_ns"
        ),
        "residency_work_time_ns": residency_work.get("summed_work_time_ns"),
        "schedule_work_time_ns": schedule_work.get("summed_work_time_ns"),
        "simulation_work_time_ns": (
            int(simulation_work.get("summed_work_time_ns") or 0)
            + int(simulation_work.get("result_materialization_time_ns") or 0)
        ),
        "admission_work_time_ns": (
            int(admission_work.get("summed_work_time_ns") or 0)
            + int(admission_work.get("result_materialization_time_ns") or 0)
        ),
        "digest_work_time_ns": digest_work.get("summed_work_time_ns"),
        "action_count": schedule.get("action_count"),
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


def _point_request(path: Path, point: dict[str, Any]) -> dict[str, Any]:
    """Read the complete request embedded in one current point record."""

    request = _mapping(point.get("request"))
    if not request:
        raise ValueError(f"frontier point has no embedded request at {path}")
    expected_digest = point.get("request_digest")
    actual = FrontierPointRequest.from_value(request)
    if actual.digest != expected_digest:
        raise ValueError(f"frontier point request changed at {path}")
    return request


def _point_case(path: Path, point: dict[str, Any]) -> dict[str, Any]:
    """Read the complete case identity embedded in one current point record."""

    case = _mapping(point.get("case"))
    if case:
        return case
    raise ValueError(f"frontier point has no embedded case identity at {path}")


def _bandwidth(request: dict[str, Any], direction: str) -> int:
    bandwidths = _mapping(request.get("transfer_bandwidths"))
    key = f"{direction}_bytes_per_second"
    value = bandwidths.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"frontier point has invalid {key}")
    return value


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
