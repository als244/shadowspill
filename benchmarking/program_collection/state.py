"""Atomic per-Program journals and resume validation."""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from contextlib import AbstractContextManager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchmarking.program_collection.corpus import (
    ProgramCaseIdentity,
    load_step_program,
)

from .config import CollectionConfig
from .matrix import ProgramRequest

_STATUS_SCHEMA = "shadowspill.program_corpus_collection.case/v1"
_SUMMARY_SCHEMA = "shadowspill.program_corpus_collection.summary/v1"


@dataclass(frozen=True, slots=True)
class CollectionPaths:
    """Stable filesystem paths for one configuration digest."""

    output_root: Path
    run_directory: Path
    config_path: Path
    cases_directory: Path
    controller_log: Path
    summary_path: Path
    lock_path: Path

    @classmethod
    def initialize(cls, output_root: Path, config: CollectionConfig) -> CollectionPaths:
        root = output_root.expanduser().resolve()
        run = root / "_collections" / f"{config.name}-{config.digest[:12]}"
        result = cls(
            output_root=root,
            run_directory=run,
            config_path=run / "config.json",
            cases_directory=run / "cases",
            controller_log=run / "collection.log",
            summary_path=run / "summary.json",
            lock_path=run / "collection.lock",
        )
        result.cases_directory.mkdir(parents=True, exist_ok=True)
        payload = config.to_dict()
        if result.config_path.exists():
            existing = _read_object(result.config_path)
            if _canonical(existing) != _canonical(payload):
                raise ValueError("collection config snapshot changed for this digest")
        else:
            atomic_json(result.config_path, payload)
        return result

    def case_directory(self, request: ProgramRequest) -> Path:
        return self.cases_directory / request.case_id

    def status_path(self, request: ProgramRequest) -> Path:
        return self.case_directory(request) / "status.json"

    def request_path(self, request: ProgramRequest) -> Path:
        return self.case_directory(request) / "request.json"

    def worker_result_path(self, request: ProgramRequest, attempt: int) -> Path:
        return self.case_directory(request) / f"worker-result-{attempt:04d}.json"

    def log_path(self, request: ProgramRequest, attempt: int) -> Path:
        return self.case_directory(request) / "logs" / f"attempt-{attempt:04d}.log"


class CollectionLock(AbstractContextManager[None]):
    """Prevent two controllers from mutating one collection journal."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._descriptor: int | None = None

    def __enter__(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self._path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            os.close(descriptor)
            raise RuntimeError(f"another collector owns {self._path.parent}") from error
        os.ftruncate(descriptor, 0)
        os.write(descriptor, f"pid={os.getpid()}\n".encode())
        self._descriptor = descriptor
        return None

    def __exit__(self, *_errors: object) -> None:
        if self._descriptor is not None:
            fcntl.flock(self._descriptor, fcntl.LOCK_UN)
            os.close(self._descriptor)
            self._descriptor = None


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
    value = _read_object(path)
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
            "started_at": utc_now(),
            "completed_at": None,
            "elapsed_seconds": None,
            "command": list(command),
            "log_path": _path_reference(paths, log_path),
            "worker_result_path": _path_reference(paths, result_path),
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
        None if artifact is None else _portable_artifact(paths, artifact)
    )
    atomic_json(paths.status_path(request), status)


def completed_artifact(
    paths: CollectionPaths,
    request: ProgramRequest,
) -> dict[str, object] | None:
    """Return validated completed evidence, or fail closed on corruption."""

    status = load_case_status(paths, request)
    raw = status.get("artifact")
    recovered = False
    if status.get("status") != "succeeded" or not isinstance(raw, dict):
        raw = _previous_success_artifact(paths, request, status)
        if raw is None:
            return None
        recovered = True
    directory = _resolve_artifact_directory(paths, request, raw)
    case, program = load_step_program(directory)
    if case.program_digest != raw.get("program_digest"):
        raise ValueError(f"saved Program digest changed for {request.case_id}")
    if program.digest != raw.get("program_digest"):
        raise ValueError(f"StepProgram content changed for {request.case_id}")
    if case.artifact_digest != raw.get("artifact_digest"):
        raise ValueError(f"StepProgram artifact digest changed for {request.case_id}")
    normalized = _portable_artifact(
        paths,
        {
            **raw,
            "directory": str(case.directory),
            "program_path": str(case.program_path),
        },
    )
    if recovered or status.get("artifact") != normalized:
        status["status"] = "succeeded"
        status["artifact"] = normalized
        repairs = status.setdefault("repairs", [])
        if not isinstance(repairs, list):
            raise ValueError(f"invalid repairs list for {request.case_id}")
        repairs.append(
            {
                "kind": "relocated_artifact_reference",
                "repaired_at": utc_now(),
                "program_digest": case.program_digest,
            }
        )
        atomic_json(paths.status_path(request), status)
    return normalized


def _previous_success_artifact(
    paths: CollectionPaths,
    request: ProgramRequest,
    status: Mapping[str, object],
) -> dict[str, object] | None:
    """Recover a validated worker artifact after its parent tree was moved."""

    attempts = status.get("attempts")
    if not isinstance(attempts, list):
        return None
    for item in reversed(attempts):
        if not isinstance(item, dict) or item.get("status") != "succeeded":
            continue
        attempt = item.get("attempt")
        if not isinstance(attempt, int):
            continue
        candidates = [paths.worker_result_path(request, attempt)]
        stored = item.get("worker_result_path")
        if isinstance(stored, str):
            stored_path = Path(stored)
            candidates.append(
                stored_path
                if stored_path.is_absolute()
                else paths.output_root / stored_path
            )
        for result_path in candidates:
            if not result_path.exists():
                continue
            result = _read_object(result_path)
            raw = result.get("artifact")
            if result.get("passed") is True and isinstance(raw, dict):
                return {str(key): value for key, value in raw.items()}
    return None


def _resolve_artifact_directory(
    paths: CollectionPaths,
    request: ProgramRequest,
    artifact: Mapping[str, object],
) -> Path:
    value = artifact.get("directory")
    if isinstance(value, str):
        stored = Path(value)
        candidate = stored if stored.is_absolute() else paths.output_root / stored
        if candidate.exists():
            return candidate
    digest = artifact.get("program_digest")
    if not isinstance(digest, str):
        raise ValueError(f"successful case {request.case_id} has no Program digest")
    identity = ProgramCaseIdentity(
        request.model.family,
        request.model.implementation,
        request.tokens_per_microbatch,
        request.sequence_length,
        request.accumulation_rounds,
    )
    candidate = (
        paths.output_root
        / "cases"
        / identity.case_name
        / identity.geometry_name
        / digest
    )
    if not candidate.exists():
        raise ValueError(
            f"saved Program directory cannot be resolved for {request.case_id}"
        )
    return candidate


def _portable_artifact(
    paths: CollectionPaths,
    artifact: Mapping[str, object],
) -> dict[str, object]:
    result = {str(key): value for key, value in artifact.items()}
    for key in ("directory", "program_path"):
        value = result.get(key)
        if isinstance(value, str):
            result[key] = _path_reference(paths, Path(value))
    return result


def _path_reference(paths: CollectionPaths, path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(paths.output_root))
    except ValueError:
        return str(resolved)


def attempt_count(paths: CollectionPaths, request: ProgramRequest) -> int:
    return len(_attempts(load_case_status(paths, request)))


def write_summary(
    paths: CollectionPaths,
    requests: Iterable[ProgramRequest],
    *,
    started_at: str,
    selected_case_ids: Iterable[str],
) -> dict[str, object]:
    """Reconcile and persist aggregate status after every case."""

    counts = {
        name: 0 for name in ("pending", "running", "succeeded", "failed", "interrupted")
    }
    failed_cases: list[str] = []
    all_requests = tuple(requests)
    for request in all_requests:
        status = str(load_case_status(paths, request).get("status", "pending"))
        if status not in counts:
            status = "failed"
        counts[status] += 1
        if status in {"failed", "interrupted"}:
            failed_cases.append(request.case_id)
    summary: dict[str, object] = {
        "schema": _SUMMARY_SCHEMA,
        "collection_name": all_requests[0].collection_name,
        "config_digest": all_requests[0].config_digest,
        "started_at": started_at,
        "updated_at": utc_now(),
        "total_programs": len(all_requests),
        "selected_case_ids": list(selected_case_ids),
        "counts": counts,
        "failed_cases": failed_cases,
    }
    atomic_json(paths.summary_path, summary)
    return summary


def append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message)
        if not message.endswith("\n"):
            handle.write("\n")
        handle.flush()


def atomic_json(path: Path, value: Mapping[str, object]) -> None:
    atomic_text(path, json.dumps(dict(value), indent=2, sort_keys=True) + "\n")


def atomic_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _attempts(status: dict[str, Any]) -> list[dict[str, Any]]:
    value = status.get("attempts")
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError("case status has an invalid attempts list")
    return value


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON object {path}") from error
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{path} does not contain a JSON object")
    return value


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


__all__ = [
    "CollectionLock",
    "CollectionPaths",
    "append_log",
    "atomic_json",
    "attempt_count",
    "begin_attempt",
    "completed_artifact",
    "finish_attempt",
    "load_case_status",
    "utc_now",
    "write_summary",
]
