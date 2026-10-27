"""Per-program attempt transitions; reporting does not mutate these journals."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from benchmarking._serialization import atomic_json
from benchmarking.journals import read_object, utc_now

from .matrix import ProgramRequest
from .paths import CollectionPaths, path_reference, portable_artifact

_STATUS_SCHEMA = "shadowspill.program_corpus_collection.case/v1"


def load_case_status(paths: CollectionPaths, request: ProgramRequest) -> dict[str, Any]:
    """Load one case journal, returning a new empty journal when absent."""

    path = paths.status_path(request)
    if not path.exists():
        return {
            "schema": _STATUS_SCHEMA,
            "case_id": request.case_id,
            "request_digest": request.digest,
            "status": "pending",
            "artifact": None,
            "attempts": [],
        }
    value = read_object(path)
    if value.get("schema") != _STATUS_SCHEMA:
        raise ValueError(f"unsupported status schema at {path}")
    if value.get("request_digest") != request.digest:
        raise ValueError(f"request digest mismatch at {path}")
    attempts = value.get("attempts")
    if not isinstance(attempts, list):
        raise ValueError(f"invalid attempts list at {path}")
    return value


def begin_attempt(
    paths: CollectionPaths,
    request: ProgramRequest,
    *,
    command: Iterable[str],
    revision: str,
) -> tuple[int, Path, Path]:
    """Append and persist a running attempt before spawning its worker."""

    directory = paths.case_directory(request)
    directory.mkdir(parents=True, exist_ok=True)
    atomic_json(paths.request_path(request), request.to_dict())
    status = load_case_status(paths, request)
    attempts = _attempts(status)
    if attempts and attempts[-1].get("status") == "running":
        attempts[-1].update(
            {
                "status": "interrupted",
                "completed_at": utc_now(),
                "error": {
                    "type": "InterruptedCollection",
                    "message": "controller ended before recording worker completion",
                },
            }
        )
    attempt = len(attempts) + 1
    log_path = paths.log_path(request, attempt)
    result_path = paths.worker_result_path(request, attempt)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    attempts.append(
        {
            "attempt": attempt,
            "status": "running",
            "revision": revision,
            "started_at": utc_now(),
            "completed_at": None,
            "elapsed_seconds": None,
            "command": list(command),
            "log_path": path_reference(paths, log_path),
            "worker_result_path": path_reference(paths, result_path),
            "return_code": None,
            "error": None,
        }
    )
    status["status"] = "running"
    atomic_json(paths.status_path(request), status)
    return attempt, log_path, result_path


def finish_attempt(
    paths: CollectionPaths,
    request: ProgramRequest,
    *,
    attempt: int,
    status_name: str,
    elapsed_seconds: float,
    return_code: int | None,
    error: Mapping[str, object] | None,
    artifact: Mapping[str, object] | None,
) -> None:
    """Atomically finish exactly the current running attempt."""

    status = load_case_status(paths, request)
    attempts = _attempts(status)
    if not attempts or attempts[-1].get("attempt") != attempt:
        raise RuntimeError(f"attempt ordering changed for {request.case_id}")
    current = attempts[-1]
    if current.get("status") != "running":
        raise RuntimeError(f"attempt {attempt} is not running")
    current.update(
        {
            "status": status_name,
            "completed_at": utc_now(),
            "elapsed_seconds": elapsed_seconds,
            "return_code": return_code,
            "error": None if error is None else dict(error),
        }
    )
    status["status"] = status_name
    status["artifact"] = (
        None if artifact is None else portable_artifact(paths, artifact)
    )
    atomic_json(paths.status_path(request), status)


def attempt_count(paths: CollectionPaths, request: ProgramRequest) -> int:
    return len(_attempts(load_case_status(paths, request)))


def _attempts(status: dict[str, Any]) -> list[dict[str, Any]]:
    value = status.get("attempts")
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError("case status has an invalid attempts list")
    return value
