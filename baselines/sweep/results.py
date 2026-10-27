"""Classify completed attempts and select cases for resumption."""

from __future__ import annotations

import json

from baselines.common.results import host_peak_bytes, validate_row
from baselines.common.results import row as result_row

from .config import flat


def classify(attempt, returncode, world_size, settings):
    """What an attempt produced: an out-of-memory, a failure, or the
    measurements of a row the engine wrote for exactly this case."""

    failures = [
        json.loads(path.read_text())
        for path in sorted(attempt.glob("rank-*/failure.json"))
    ]
    if any(item.get("status") == "cuda_out_of_memory" for item in failures):
        return {"status": "cuda_out_of_memory", "failures": failures}
    if returncode:
        return {
            "status": "failed",
            "failures": failures,
            "reason": f"exit status {returncode}",
        }
    path = attempt / "row.json"
    if not path.is_file():
        return {"status": "failed", "reason": "the process exited without a row"}
    try:
        row = json.loads(path.read_text())
        validate_row(row)
    except ValueError as error:
        return {"status": "failed", "reason": f"invalid row: {error}"}
    if row["status"] != "completed":
        return {"status": "failed", "reason": f"the row says {row['status']}"}
    expected = settings.describe(world_size)
    for key in ("model", "geometry", "precision", "configuration"):
        if row[key] != expected[key]:
            return {
                "status": "failed",
                "reason": f"the executed {key} differs from the request",
            }
    measurement, memory = row["measurement"], row["memory"]
    return {
        "status": "completed",
        "step_seconds": measurement["step_seconds"],
        "global_tokens_per_second": measurement["global_tokens_per_second"],
        "useful_tflops_per_gpu": measurement.get("useful_tflops_per_gpu"),
        "performance_result": bool(measurement.get("performance_result", True)),
        "peak_allocated_bytes": memory["peak_allocated_bytes"],
        "peak_reserved_bytes": memory["peak_reserved_bytes"],
        "host_peak_bytes": host_peak_bytes(row),
        "host_peak_rss_bytes": memory["host_peak_rss_bytes"],
    }


def to_attempt(previous, retry_failed):
    """Whether a case gets a new attempt: one never attempted or interrupted
    does, a completed one never does, and one that failed, ran out of
    memory or timed out does on a retry."""

    status = previous["status"]
    return status in ("pending", "interrupted") or (
        status != "completed" and retry_failed
    )


def latest(case):
    attempts = sorted(case.glob("attempt-*"))
    if not attempts:
        return {"status": "pending"}
    path = attempts[-1] / "outcome.json"
    if not path.exists():
        return {"status": "interrupted", "attempt": attempts[-1].name}
    outcome = json.loads(path.read_text())
    if outcome["status"] == "completed" and "host_peak_bytes" not in outcome:
        row_path = path.with_name("row.json")
        outcome["host_peak_bytes"] = (
            host_peak_bytes(json.loads(row_path.read_text()))
            if row_path.is_file()
            else outcome["host_peak_rss_bytes"]
        )
    return outcome


def failure_row(attempt, outcome, settings, world_size, engine, manifest):
    """The row this launcher writes for an attempt that produced none: the
    request as described by the settings, the hardware the first rank
    reported before it failed, and what went wrong."""

    hardware = {}
    request = attempt / "rank-00000" / "request.json"
    if request.is_file():
        hardware = json.loads(request.read_text()).get("hardware") or {}
    failure = None
    if outcome["status"] in ("failed", "timeout", "cuda_out_of_memory"):
        failure = {
            key: outcome.get(key) for key in ("returncode", "reason", "failures")
        }
    return result_row(
        engine=engine,
        status=outcome["status"],
        **settings.describe(world_size),
        hardware=hardware,
        provenance={
            "source": manifest["source"],
            "environment": manifest["environment"],
        },
        failure=failure,
    )


def described(cases, world_size):
    """Per case, the geometry and configuration its settings describe."""

    return {name: flat(settings.describe(world_size)) for name, settings in cases}
