"""Validate completed artifacts and resolve relocated collection directories."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from benchmarking._serialization import atomic_json
from benchmarking.journals import read_object, utc_now

from .artifacts import load_program
from .corpus import ProgramCaseIdentity
from .journal import load_case_status
from .matrix import ProgramRequest
from .paths import CollectionPaths, portable_artifact


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
    case, program = load_program(directory)
    if case.program_digest != raw.get("program_digest"):
        raise ValueError(f"saved program digest changed for {request.case_id}")
    if program.digest != raw.get("program_digest"):
        raise ValueError(f"ProfiledProgram content changed for {request.case_id}")
    if case.artifact_digest != raw.get("artifact_digest"):
        raise ValueError(
            f"ProfiledProgram artifact digest changed for {request.case_id}"
        )
    normalized = portable_artifact(
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
            result = read_object(result_path)
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
        raise ValueError(f"successful case {request.case_id} has no program digest")
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
            f"saved program directory cannot be resolved for {request.case_id}"
        )
    return candidate
