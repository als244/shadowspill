"""Per-point attempt transitions and final evidence validation."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from benchmarking._serialization import atomic_json, canonical_json_value
from benchmarking.journals import read_object, utc_now
from shadowspill.schema import artifact_schema

from .manifest import BaselinePaths, write_or_match
from .matrix import FrontierPointRequest
from .source import CorpusProgramCase

_STATUS_SCHEMA = artifact_schema("search_frontier_point_status")
_POINT_SCHEMA = artifact_schema("search_frontier_point")


def initialize_point(
    paths: BaselinePaths,
    case: CorpusProgramCase,
    request: FrontierPointRequest,
) -> Path:
    directory = paths.point_directory(case, request)
    directory.mkdir(parents=True, exist_ok=True)
    write_or_match(directory / "request.json", request.to_dict(), "point request")
    status_path = directory / "status.json"
    if not status_path.exists():
        atomic_json(
            status_path,
            {
                "schema": _STATUS_SCHEMA,
                "request_digest": request.digest,
                "point_id": request.point_id,
                "status": "pending",
                "attempts": [],
            },
        )
    else:
        _validate_status(read_object(status_path), request, status_path)
    return directory


def point_complete(directory: Path, request: FrontierPointRequest) -> bool:
    status = read_object(directory / "status.json")
    _validate_status(status, request, directory / "status.json")
    if status.get("status") not in {
        "succeeded",
        "infeasible",
        "search_exhausted",
        "error",
    }:
        return False
    point_path = directory / "point.json"
    if not point_path.exists():
        raise ValueError(f"completed point lacks evidence at {directory}")
    point = read_object(point_path)
    if point.get("schema") != _POINT_SCHEMA:
        raise ValueError(f"unsupported point evidence schema at {point_path}")
    if point.get("request_digest") != request.digest:
        raise ValueError(f"point evidence request changed at {point_path}")
    if point.get("status") != status.get("status"):
        raise ValueError(f"point evidence and status disagree at {point_path}")
    return True


def point_attempt_count(directory: Path, request: FrontierPointRequest) -> int:
    """Return attempts charged against the configured retry budget."""

    status = read_object(directory / "status.json")
    _validate_status(status, request, directory / "status.json")
    attempts = status["attempts"]
    assert isinstance(attempts, list)
    return sum(item.get("status") != "interrupted" for item in attempts)


def recover_interrupted_attempt(
    directory: Path,
    request: FrontierPointRequest,
) -> bool:
    """Close one orphaned running attempt without consuming its retry budget."""

    status_path = directory / "status.json"
    status = read_object(status_path)
    _validate_status(status, request, status_path)
    attempts = status["attempts"]
    assert isinstance(attempts, list)
    if not attempts or not isinstance(attempts[-1], dict):
        return False
    current = attempts[-1]
    if current.get("status") != "running":
        return False
    current.update(
        {
            "status": "interrupted",
            "completed_at": utc_now(),
            "elapsed_seconds": None,
            "error": {
                "type": "FrontierControllerInterrupted",
                "message": "controller stopped before the point produced evidence",
            },
        }
    )
    status["status"] = "retryable"
    atomic_json(status_path, status)
    return True


def begin_point_attempt(
    directory: Path,
    request: FrontierPointRequest,
    *,
    revision: str,
) -> int:
    status_path = directory / "status.json"
    status = read_object(status_path)
    _validate_status(status, request, status_path)
    attempts = status["attempts"]
    assert isinstance(attempts, list)
    if (
        attempts
        and isinstance(attempts[-1], dict)
        and attempts[-1].get("status") == "running"
    ):
        raise RuntimeError("cannot begin a point while its prior attempt is running")
    prior_ordinals = tuple(
        int(item["attempt"])
        for item in attempts
        if isinstance(item, dict) and isinstance(item.get("attempt"), int)
    )
    attempt = max(prior_ordinals, default=0) + 1
    attempts.append(
        {
            "attempt": attempt,
            "status": "running",
            "revision": revision,
            "started_at": utc_now(),
            "completed_at": None,
            "elapsed_seconds": None,
            "error": None,
        }
    )
    status["status"] = "running"
    atomic_json(status_path, status)
    return attempt


def finish_point_attempt(
    directory: Path,
    request: FrontierPointRequest,
    *,
    attempt: int,
    status_name: str,
    elapsed_seconds: float,
    evidence: Mapping[str, object] | None,
    error: Mapping[str, object] | None,
    final: bool,
) -> None:
    status_path = directory / "status.json"
    status = read_object(status_path)
    _validate_status(status, request, status_path)
    attempts = status["attempts"]
    assert isinstance(attempts, list)
    if not attempts or not isinstance(attempts[-1], dict):
        raise RuntimeError("point has no current attempt")
    current = attempts[-1]
    if current.get("attempt") != attempt or current.get("status") != "running":
        raise RuntimeError("point attempt identity changed")
    current.update(
        {
            "status": status_name,
            "completed_at": utc_now(),
            "elapsed_seconds": elapsed_seconds,
            "error": None if error is None else dict(error),
        }
    )
    if final:
        if evidence is None:
            raise ValueError("a final point requires evidence")
        final_evidence = dict(evidence)
        recorded_request = final_evidence.get("request")
        if recorded_request is not None and canonical_json_value(
            recorded_request
        ) != canonical_json_value(request.to_dict()):
            raise ValueError("point evidence request differs from its journal")
        final_evidence["request"] = request.to_dict()
        point = {
            "schema": _POINT_SCHEMA,
            "request_digest": request.digest,
            "point_id": request.point_id,
            "status": status_name,
            "revision": current.get("revision"),
            **final_evidence,
        }
        atomic_json(directory / "point.json", point)
        status["status"] = status_name
    else:
        status["status"] = "retryable"
    atomic_json(status_path, status)


def recover_running_attempt(
    directory: Path,
    request: FrontierPointRequest,
    *,
    case: CorpusProgramCase,
    max_attempts: int,
    elapsed_seconds: float,
    error: Mapping[str, object],
) -> bool:
    """Close a worker-killed attempt; return whether the point is now final."""

    status_path = directory / "status.json"
    status = read_object(status_path)
    _validate_status(status, request, status_path)
    attempts = status["attempts"]
    assert isinstance(attempts, list)
    if not attempts or not isinstance(attempts[-1], dict):
        return point_complete(directory, request)
    current = attempts[-1]
    if current.get("status") != "running":
        return point_complete(directory, request)
    attempt = int(current["attempt"])
    final = attempt >= max_attempts
    completed_at = utc_now()
    evidence = {
        "case": case.to_dict(),
        "request": request.to_dict(),
        "timing": {
            "started_at": current.get("started_at"),
            "completed_at": completed_at,
            "attempt_elapsed_seconds": elapsed_seconds,
        },
        "error": dict(error),
    }
    finish_point_attempt(
        directory,
        request,
        attempt=attempt,
        status_name="error",
        elapsed_seconds=elapsed_seconds,
        error=error,
        evidence=evidence if final else None,
        final=final,
    )
    return final


def _validate_status(
    status: dict[str, Any],
    request: FrontierPointRequest,
    path: Path,
) -> None:
    if status.get("schema") != _STATUS_SCHEMA:
        raise ValueError(f"unsupported point status schema at {path}")
    if status.get("request_digest") != request.digest:
        raise ValueError(f"point status request changed at {path}")
    attempts = status.get("attempts")
    if not isinstance(attempts, list) or any(
        not isinstance(item, dict) for item in attempts
    ):
        raise ValueError(f"point status attempts are invalid at {path}")
