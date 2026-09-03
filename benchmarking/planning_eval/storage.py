"""Atomic baseline manifests, point journals, and final evidence."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from shadowspill.schema import artifact_schema

from .config import FrontierConfig
from .matrix import FrontierPointRequest
from .provenance import RepositoryProvenance, environment_provenance
from .source import CorpusProgramCase

_BASELINE_SCHEMA = artifact_schema("pressurefit_frontier_baseline")
_STATUS_SCHEMA = artifact_schema("pressurefit_frontier_point_status")
_POINT_SCHEMA = artifact_schema("pressurefit_frontier_point")


@dataclass(frozen=True, slots=True)
class BaselinePaths:
    """Stable central index for one exact planner implementation."""

    directory: Path
    cases_directory: Path
    config_path: Path
    manifest_path: Path
    summary_path: Path
    csv_path: Path
    jsonl_path: Path
    log_path: Path
    patch_path: Path

    @classmethod
    def open_existing(
        cls,
        directory: Path,
        *,
        config: FrontierConfig,
        corpus_digest: str,
    ) -> BaselinePaths:
        """Open and validate one immutable baseline selected for resume."""

        result = cls._from_directory(directory.expanduser().resolve())
        if not result.directory.is_dir():
            raise ValueError(f"resume baseline does not exist: {result.directory}")
        if _canonical(read_object(result.config_path)) != _canonical(config.to_dict()):
            raise ValueError("resume baseline configuration changed")
        manifest = read_object(result.manifest_path)
        if manifest.get("baseline_id") != result.directory.name:
            raise ValueError("resume baseline identity is invalid")
        if manifest.get("config_digest") != config.digest:
            raise ValueError("resume baseline config digest changed")
        corpus = manifest.get("corpus")
        if (
            not isinstance(corpus, dict)
            or corpus.get("manifest_digest") != corpus_digest
        ):
            raise ValueError("resume baseline corpus digest changed")
        return result

    @classmethod
    def _from_directory(cls, directory: Path) -> BaselinePaths:
        return cls(
            directory=directory,
            cases_directory=directory / "cases",
            config_path=directory / "config.json",
            manifest_path=directory / "manifest.json",
            summary_path=directory / "summary.json",
            csv_path=directory / "frontier.csv",
            jsonl_path=directory / "frontier.jsonl",
            log_path=directory / "collection.log",
            patch_path=directory / "planner.patch",
        )

    @classmethod
    def initialize(
        cls,
        output_root: Path,
        *,
        baseline_id: str,
        config: FrontierConfig,
        provenance: RepositoryProvenance,
        corpus_root: Path,
        corpus_digest: str,
        cases: tuple[CorpusProgramCase, ...],
    ) -> BaselinePaths:
        directory = output_root.expanduser().resolve() / baseline_id
        result = cls._from_directory(directory)
        result.cases_directory.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema": _BASELINE_SCHEMA,
            "baseline_id": baseline_id,
            "created_at": utc_now(),
            "config_digest": config.digest,
            "corpus": {
                "root": str(corpus_root.expanduser().resolve()),
                "manifest_digest": corpus_digest,
                "program_count": len(cases),
                "programs": [item.to_dict() for item in cases],
            },
            "repository": provenance.to_dict(),
            "environment": environment_provenance(),
        }
        _write_or_match(result.config_path, config.to_dict(), "frontier config")
        if result.manifest_path.exists():
            existing = read_object(result.manifest_path)
            stable_existing = dict(existing)
            stable_manifest = dict(manifest)
            stable_existing.pop("created_at", None)
            stable_manifest.pop("created_at", None)
            stable_existing_environment = stable_existing.get("environment")
            stable_manifest_environment = stable_manifest.get("environment")
            if isinstance(stable_existing_environment, dict):
                stable_existing_environment.pop("pid", None)
            if isinstance(stable_manifest_environment, dict):
                stable_manifest_environment.pop("pid", None)
            if _canonical(stable_existing) != _canonical(stable_manifest):
                raise ValueError("baseline manifest identity changed")
        else:
            atomic_json(result.manifest_path, manifest)
        _write_or_match_text(result.patch_path, provenance.diff, "planner patch")
        _write_or_match_text(
            result.directory / "git-status.txt",
            provenance.status,
            "git status",
        )
        _write_or_match_text(result.directory / "README.md", _README, "baseline guide")
        return result

    def case_directory(self, case: CorpusProgramCase) -> Path:
        return self.cases_directory / case.case_id

    def point_directory(
        self,
        case: CorpusProgramCase,
        request: FrontierPointRequest,
    ) -> Path:
        return self.case_directory(case) / "points" / request.point_id


def initialize_point(
    paths: BaselinePaths,
    case: CorpusProgramCase,
    request: FrontierPointRequest,
) -> Path:
    directory = paths.point_directory(case, request)
    directory.mkdir(parents=True, exist_ok=True)
    _write_or_match(directory / "request.json", request.to_dict(), "point request")
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
        if recorded_request is not None and _canonical(recorded_request) != _canonical(
            request.to_dict()
        ):
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


def write_active_point(
    case_directory: Path,
    request: FrontierPointRequest | None,
) -> None:
    atomic_json(
        case_directory / "active.json",
        {
            "schema": artifact_schema("pressurefit_frontier_active"),
            "updated_at": utc_now(),
            "point_id": None if request is None else request.point_id,
            "request_digest": None if request is None else request.digest,
        },
    )


def read_active_point(case_directory: Path) -> tuple[str, str] | None:
    path = case_directory / "active.json"
    if not path.exists():
        return None
    value = read_object(path)
    point_id = value.get("point_id")
    request_digest = value.get("request_digest")
    if point_id is None and request_digest is None:
        return None
    if not isinstance(point_id, str) or not isinstance(request_digest, str):
        raise ValueError(f"invalid active point journal at {path}")
    return point_id, request_digest


def read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON object {path}") from error
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{path} does not contain a JSON object")
    return value


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


def _write_or_match(path: Path, value: Mapping[str, object], name: str) -> None:
    if path.exists():
        if _canonical(read_object(path)) != _canonical(value):
            raise ValueError(f"{name} changed at {path}")
        return
    atomic_json(path, value)


def _write_or_match_text(path: Path, payload: str, name: str) -> None:
    if path.exists():
        if path.read_text() != payload:
            raise ValueError(f"{name} changed at {path}")
        return
    atomic_text(path, payload)


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


_README = """# PressureFit frontier baseline

This directory is an immutable planner-implementation identity plus resumable
per-Program journals. `config.json` defines the exact budget/bandwidth matrix;
`manifest.json` freezes the Program corpus and source revision; `planner.patch`
freezes tracked uncommitted changes. `frontier.csv` and `summary.json` are the
compact comparison index.

Each `cases/<case>/points/<point>/point.json` is the compact outcome for one
Program/budget/bandwidth input. Successful points link to a complete canonical
`AnnotatedProgramPlan` under the corresponding baseline case's
`annotated-plans/` tree. The frozen source corpus is never modified by a
frontier run. Each plan artifact contains the selected schedule, simulator
intervals and memory timeline, physical admission certificate, diagnostics,
and source Program.
"""


__all__ = [
    "BaselinePaths",
    "append_log",
    "atomic_json",
    "atomic_text",
    "begin_point_attempt",
    "finish_point_attempt",
    "initialize_point",
    "point_attempt_count",
    "point_complete",
    "read_active_point",
    "read_object",
    "recover_running_attempt",
    "utc_now",
    "write_active_point",
]
