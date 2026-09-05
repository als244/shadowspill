"""Sequential subprocess controller for long-running Program collection."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from .config import CollectionConfig
from .matrix import ProgramRequest
from .process import WorkerOutcome, execute_worker, load_worker_outcome
from .state import (
    CollectionLock,
    CollectionPaths,
    append_log,
    attempt_count,
    begin_attempt,
    completed_artifact,
    finish_attempt,
    load_case_status,
    utc_now,
    write_summary,
)


@dataclass(frozen=True, slots=True)
class ControllerOptions:
    """Invocation-level controls that do not affect Program identity."""

    #: The revision every case this run produces records.
    revision: str
    artifact_store: Path
    resume: bool
    timeout_seconds: int
    max_attempts: int
    quiet_plan: bool
    force_fresh: bool


def run_collection(
    config: CollectionConfig,
    requests: tuple[ProgramRequest, ...],
    selected: tuple[ProgramRequest, ...],
    *,
    output_root: Path,
    options: ControllerOptions,
) -> dict[str, object]:
    """Collect every selected Program, recording failures and continuing."""

    paths = CollectionPaths.initialize(output_root, config)
    options.artifact_store.mkdir(parents=True, exist_ok=True)
    _preflight_resume(paths, selected, options)
    started_at = utc_now()
    selected_ids = tuple(request.case_id for request in selected)
    with CollectionLock(paths.lock_path):
        _log(
            paths,
            "COLLECTION START "
            f"utc={started_at} config={config.name} digest={config.digest} "
            f"selected={len(selected)} total={len(requests)} resume={options.resume}",
        )
        write_summary(
            paths,
            requests,
            started_at=started_at,
            selected_case_ids=selected_ids,
        )
        for ordinal, request in enumerate(selected, 1):
            prefix = f"[{ordinal}/{len(selected)}] {request.case_id}"
            try:
                _recover_worker_result(paths, request)
                artifact = completed_artifact(paths, request)
            except BaseException as error:
                _record_controller_failure(
                    paths, request, prefix, error, options.revision
                )
                write_summary(
                    paths,
                    requests,
                    started_at=started_at,
                    selected_case_ids=selected_ids,
                )
                continue
            if artifact is not None:
                _log(
                    paths,
                    f"PROGRAM SKIP {prefix} reason=validated_completed "
                    f"digest={artifact['program_digest']}",
                )
                continue
            attempts = attempt_count(paths, request)
            if attempts >= options.max_attempts:
                _log(
                    paths,
                    f"PROGRAM SKIP {prefix} reason=max_attempts attempts={attempts}",
                )
                continue
            try:
                outcome = _run_one(paths, request, prefix=prefix, options=options)
            except KeyboardInterrupt:
                raise
            except BaseException as error:
                _record_controller_failure(
                    paths, request, prefix, error, options.revision
                )
                write_summary(
                    paths,
                    requests,
                    started_at=started_at,
                    selected_case_ids=selected_ids,
                )
                continue
            if outcome.status == "succeeded":
                assert outcome.artifact is not None
                _log(
                    paths,
                    f"PROGRAM SUCCESS {prefix} "
                    f"elapsed_seconds={outcome.elapsed_seconds:.3f} "
                    f"digest={outcome.artifact['program_digest']}",
                )
            else:
                error_type = (
                    None if outcome.error is None else outcome.error.get("type")
                )
                _log(
                    paths,
                    f"PROGRAM FAILURE {prefix} status={outcome.status} "
                    f"return_code={outcome.return_code} error_type={error_type} "
                    f"elapsed_seconds={outcome.elapsed_seconds:.3f}; continuing",
                )
            write_summary(
                paths,
                requests,
                started_at=started_at,
                selected_case_ids=selected_ids,
            )
        summary = write_summary(
            paths,
            requests,
            started_at=started_at,
            selected_case_ids=selected_ids,
        )
        _log(
            paths,
            f"COLLECTION COMPLETE utc={utc_now()} counts={summary['counts']}",
        )
        return summary


def _run_one(
    paths: CollectionPaths,
    request: ProgramRequest,
    *,
    prefix: str,
    options: ControllerOptions,
) -> WorkerOutcome:
    next_attempt = attempt_count(paths, request) + 1
    result_path = paths.worker_result_path(request, next_attempt)
    command = _worker_command(paths, request, result_path, options)
    attempt, log_path, actual_result_path = begin_attempt(
        paths,
        request,
        command=command,
        revision=options.revision,
    )
    if attempt != next_attempt or actual_result_path != result_path:
        raise RuntimeError("case attempt changed while preparing its worker")
    _log(
        paths,
        f"PROGRAM START {prefix} attempt={attempt} utc={utc_now()} "
        f"data_geometry=({request.data_geometry.describe()}) log={log_path}",
    )
    outcome = execute_worker(
        command,
        repository_root=_repository_root(),
        log_path=log_path,
        result_path=result_path,
        timeout_seconds=options.timeout_seconds,
        console_prefix=prefix,
        expected_request=request,
    )
    finish_attempt(
        paths,
        request,
        attempt=attempt,
        status_name=outcome.status,
        elapsed_seconds=outcome.elapsed_seconds,
        return_code=outcome.return_code,
        error=outcome.error,
        artifact=outcome.artifact,
    )
    if outcome.status == "interrupted":
        raise KeyboardInterrupt
    return outcome


def _worker_command(
    paths: CollectionPaths,
    request: ProgramRequest,
    result_path: Path,
    options: ControllerOptions,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "benchmarking.program_collection.worker",
        "--config",
        str(paths.config_path),
        "--case-id",
        request.case_id,
        "--output-dir",
        str(paths.output_root),
        "--artifact-store",
        str(options.artifact_store),
        "--result",
        str(result_path),
    ]
    if options.quiet_plan:
        command.append("--quiet-plan")
    if options.force_fresh:
        command.append("--force-fresh")
    return command


def _preflight_resume(
    paths: CollectionPaths,
    selected: tuple[ProgramRequest, ...],
    options: ControllerOptions,
) -> None:
    if options.resume:
        return
    existing = tuple(
        request.case_id for request in selected if paths.status_path(request).exists()
    )
    if existing:
        preview = ", ".join(existing[:3])
        raise FileExistsError(
            "collection state already exists; pass --resume to validate completed "
            f"Programs and continue ({preview})"
        )


def _record_controller_failure(
    paths: CollectionPaths,
    request: ProgramRequest,
    prefix: str,
    error: BaseException,
    revision: str,
) -> None:
    next_attempt = attempt_count(paths, request) + 1
    command = ["controller-validation"]
    attempt, _log_path, _result_path = begin_attempt(
        paths,
        request,
        command=command,
        revision=revision,
    )
    if attempt != next_attempt:
        raise RuntimeError("controller failure attempt changed")
    finish_attempt(
        paths,
        request,
        attempt=attempt,
        status_name="failed",
        elapsed_seconds=0.0,
        return_code=None,
        error={"type": type(error).__name__, "message": str(error)},
        artifact=None,
    )
    _log(
        paths,
        f"PROGRAM FAILURE {prefix} error_type={type(error).__name__} "
        f"error={error}; continuing",
    )


def _recover_worker_result(
    paths: CollectionPaths,
    request: ProgramRequest,
) -> None:
    """Commit an atomic worker result left by an interrupted controller."""

    status = load_case_status(paths, request)
    if status.get("status") != "running":
        return
    attempts = status.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        raise ValueError("running case has no attempt record")
    current = attempts[-1]
    if not isinstance(current, dict):
        raise ValueError("running case has an invalid attempt record")
    attempt = current.get("attempt")
    result_value = current.get("worker_result_path")
    if not isinstance(attempt, int) or not isinstance(result_value, str):
        raise ValueError("running case has incomplete worker-result identity")
    stored_path = Path(result_value)
    result_path = (
        stored_path if stored_path.is_absolute() else paths.output_root / stored_path
    )
    if not result_path.exists():
        result_path = paths.worker_result_path(request, attempt)
    if not result_path.exists():
        return
    outcome = load_worker_outcome(
        result_path,
        return_code=0,
        elapsed_seconds=0.0,
        expected_request=request,
    )
    finish_attempt(
        paths,
        request,
        attempt=attempt,
        status_name=outcome.status,
        elapsed_seconds=outcome.elapsed_seconds,
        return_code=outcome.return_code,
        error=outcome.error,
        artifact=outcome.artifact,
    )


def _log(paths: CollectionPaths, message: str) -> None:
    line = f"[{utc_now()}] {message}"
    append_log(paths.controller_log, line)
    print(line, flush=True)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


__all__ = ["ControllerOptions", "run_collection"]
