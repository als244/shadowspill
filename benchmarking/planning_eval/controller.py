"""Sequential, crash-isolated orchestration of all Program frontier workers."""

from __future__ import annotations

import fcntl
import os
import signal
import sys
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path

from .config import FrontierConfig
from .matrix import FrontierPointRequest
from .process import WorkerProcessOutcome, execute_case_worker
from .source import CorpusProgramCase
from .storage import (
    BaselinePaths,
    append_log,
    atomic_json,
    read_active_point,
    read_object,
    recover_interrupted_attempt,
    recover_running_attempt,
    utc_now,
    write_active_point,
)
from .summary import write_frontier_summary


@dataclass(frozen=True, slots=True)
class ControllerOptions:
    planning_cache: Path
    resume: bool
    verbose_pressurefit: bool


def run_frontier_collection(
    config: FrontierConfig,
    all_cases: tuple[CorpusProgramCase, ...],
    selected_cases: tuple[CorpusProgramCase, ...],
    *,
    paths: BaselinePaths,
    options: ControllerOptions,
    repository_root: Path,
) -> dict[str, object]:
    """Evaluate selected cases while preserving and continuing past failures."""

    options.planning_cache.mkdir(parents=True, exist_ok=True)
    _preflight_resume(paths, selected_cases, options.resume)
    failures = _load_case_failures(paths)
    with _CollectionLock(paths.directory / "collection.lock"):
        recovered = _recover_interrupted_points(paths, selected_cases)
        _log(
            paths,
            "FRONTIER START "
            f"utc={utc_now()} baseline={paths.directory.name} "
            f"selected_programs={len(selected_cases)} total_programs={len(all_cases)} "
            f"points_per_program={config.expected_points_per_program} "
            f"resume={options.resume}",
        )
        if recovered:
            _log(
                paths,
                f"RESUME RECOVERED interrupted_points={recovered}",
            )
        write_frontier_summary(
            paths,
            expected_programs=len(all_cases),
            expected_points_per_program=config.expected_points_per_program,
            case_failures=failures,
        )
        global_ordinals = {
            item.case_id: index
            for index, item in enumerate(all_cases, 1)
        }
        global_point_count = (
            len(all_cases) * config.expected_points_per_program
        )
        for case in selected_cases:
            program_ordinal = global_ordinals[case.case_id]
            prefix = f"[{program_ordinal}/{len(all_cases)}] {case.case_id}"
            _program_separator(paths)
            if _case_complete(paths, case, config.expected_points_per_program):
                _log(paths, f"PROGRAM SKIP {prefix} reason=validated_completed")
                continue
            try:
                completed = _run_case_until_complete(
                    config,
                    case,
                    paths=paths,
                    options=options,
                    repository_root=repository_root,
                    prefix=prefix,
                    console_prefix=f"[{program_ordinal}/{len(all_cases)}]",
                    global_point_base=(
                        (program_ordinal - 1)
                        * config.expected_points_per_program
                    ),
                    global_point_count=global_point_count,
                )
            except KeyboardInterrupt:
                raise
            except BaseException as error:
                completed = False
                failures[case.case_id] = {
                    "type": type(error).__name__,
                    "message": str(error),
                    "recorded_at": utc_now(),
                }
                _log(
                    paths,
                    f"PROGRAM FAILURE {prefix} error_type={type(error).__name__} "
                    f"error={error}; continuing",
                )
            if completed:
                failures.pop(case.case_id, None)
                _log(paths, f"PROGRAM COMPLETE {prefix}")
            atomic_json(paths.directory / "case-failures.json", failures)
            write_frontier_summary(
                paths,
                expected_programs=len(all_cases),
                expected_points_per_program=config.expected_points_per_program,
                case_failures=failures,
            )
        summary = write_frontier_summary(
            paths,
            expected_programs=len(all_cases),
            expected_points_per_program=config.expected_points_per_program,
            case_failures=failures,
        )
        _log(
            paths,
            f"FRONTIER COMPLETE utc={utc_now()} "
            f"completed={summary['completed_points']} "
            f"counts={summary['status_counts']} failures={len(failures)}",
        )
        return summary


def _recover_interrupted_points(
    paths: BaselinePaths,
    selected_cases: tuple[CorpusProgramCase, ...],
) -> int:
    """Make orphaned running journals resumable after acquiring the sole lock."""

    recovered = 0
    for case in selected_cases:
        case_directory = paths.case_directory(case)
        case_path = case_directory / "case.json"
        if not case_path.is_file():
            continue
        raw_points = read_object(case_path).get("points")
        if not isinstance(raw_points, list):
            raise ValueError(f"frontier case point inventory is invalid at {case_path}")
        for raw in raw_points:
            request = FrontierPointRequest.from_value(raw)
            directory = case_directory / "points" / request.point_id
            if not (directory / "status.json").is_file():
                continue
            if recover_interrupted_attempt(directory, request):
                recovered += 1
        write_active_point(case_directory, None)
    return recovered


def _run_case_until_complete(
    config: FrontierConfig,
    case: CorpusProgramCase,
    *,
    paths: BaselinePaths,
    options: ControllerOptions,
    repository_root: Path,
    prefix: str,
    console_prefix: str,
    global_point_base: int,
    global_point_count: int,
) -> bool:
    case_directory = paths.case_directory(case)
    case_directory.mkdir(parents=True, exist_ok=True)
    first_attempt = _worker_attempt_count(case_directory) + 1
    for restart in range(first_attempt, config.max_worker_restarts_per_program + 1):
        if _case_complete(paths, case, config.expected_points_per_program):
            return True
        command = _worker_command(
            paths,
            case,
            planning_cache=options.planning_cache,
            verbose_pressurefit=options.verbose_pressurefit,
            global_point_base=global_point_base,
            global_point_count=global_point_count,
        )
        log_path = case_directory / "logs" / f"worker-{restart:04d}.log"
        _log(
            paths,
            f"PROGRAM START {prefix} worker_attempt={restart} "
            f"utc={utc_now()} log={log_path}",
        )
        outcome = execute_case_worker(
            command,
            repository_root=repository_root,
            case_run_directory=case_directory,
            log_path=log_path,
            collection_log_path=paths.log_path,
            point_timeout_seconds=config.point_timeout_seconds,
            console_prefix=console_prefix,
        )
        _record_worker_attempt(case_directory, restart, command, log_path, outcome)
        if _case_complete(paths, case, config.expected_points_per_program):
            return True
        _recover_active_point(
            config,
            case,
            paths=paths,
            outcome=outcome,
        )
        reason = _process_failure(outcome)
        _log(
            paths,
            f"PROGRAM RESTART {prefix} worker_attempt={restart} "
            f"return_code={outcome.return_code} reason={reason}",
        )
    raise RuntimeError(
        "Program worker exceeded max restarts before completing its points: "
        f"{case.case_id}"
    )


def _worker_attempt_count(directory: Path) -> int:
    path = directory / "worker-attempts.json"
    if not path.exists():
        return 0
    attempts = read_object(path).get("attempts")
    if not isinstance(attempts, list):
        raise ValueError(f"invalid worker attempt journal at {path}")
    return len(attempts)


def _recover_active_point(
    config: FrontierConfig,
    case: CorpusProgramCase,
    *,
    paths: BaselinePaths,
    outcome: WorkerProcessOutcome,
) -> None:
    active = outcome.active_point or read_active_point(paths.case_directory(case))
    if active is None:
        return
    point_id, request_digest = active
    directory = paths.case_directory(case) / "points" / point_id
    request = FrontierPointRequest.from_value(read_object(directory / "request.json"))
    if request.digest != request_digest:
        raise ValueError("active point and request journal disagree")
    error = {
        "type": (
            "FrontierPointTimeout"
            if outcome.timed_out_point_id is not None
            else "FrontierWorkerProcessFailure"
        ),
        "message": _process_failure(outcome),
        "return_code": outcome.return_code,
    }
    final = recover_running_attempt(
        directory,
        request,
        case=case,
        max_attempts=config.max_point_attempts,
        elapsed_seconds=outcome.active_point_elapsed_seconds or 0.0,
        error=error,
    )
    disposition = "FINAL" if final else "RETRY"
    if outcome.timed_out_point_id is not None:
        _log(
            paths,
            f"POINT TIMEOUT {disposition} {case.case_id} point={point_id} "
            f"timeout_seconds={config.point_timeout_seconds} "
            f"return_code={outcome.return_code}",
        )
    else:
        _log(
            paths,
            f"POINT WORKER_FAILURE {disposition} {case.case_id} "
            f"point={point_id} return_code={outcome.return_code}",
        )
    write_active_point(paths.case_directory(case), None)


def _case_complete(
    paths: BaselinePaths,
    case: CorpusProgramCase,
    expected_points: int,
) -> bool:
    directory = paths.case_directory(case)
    case_path = directory / "case.json"
    if not case_path.exists():
        return False
    value = read_object(case_path)
    points = value.get("points")
    if not isinstance(points, list) or len(points) != expected_points:
        raise ValueError(f"frontier case point inventory is invalid at {case_path}")
    for raw in points:
        request = FrontierPointRequest.from_value(raw)
        point_path = directory / "points" / request.point_id / "point.json"
        if not point_path.exists():
            return False
        point = read_object(point_path)
        if point.get("request_digest") != request.digest:
            raise ValueError(f"frontier point request changed at {point_path}")
    return True


def _worker_command(
    paths: BaselinePaths,
    case: CorpusProgramCase,
    *,
    planning_cache: Path,
    verbose_pressurefit: bool,
    global_point_base: int,
    global_point_count: int,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "benchmarking.planning_eval.worker",
        "--config",
        str(paths.config_path),
        "--baseline-dir",
        str(paths.directory),
        "--case-dir",
        str(case.directory),
        "--planning-cache",
        str(planning_cache),
        "--global-point-base",
        str(global_point_base),
        "--global-point-count",
        str(global_point_count),
    ]
    if verbose_pressurefit:
        command.append("--verbose-pressurefit")
    return command


def _record_worker_attempt(
    directory: Path,
    attempt: int,
    command: list[str],
    log_path: Path,
    outcome: WorkerProcessOutcome,
) -> None:
    path = directory / "worker-attempts.json"
    if path.exists():
        value = read_object(path)
    else:
        value = {
            "schema": "shadowspill.pressurefit_frontier_worker_attempts/v1",
            "attempts": [],
        }
    attempts = value.get("attempts")
    if not isinstance(attempts, list):
        raise ValueError(f"invalid worker attempt journal at {path}")
    attempts.append(
        {
            "attempt": attempt,
            "completed_at": utc_now(),
            "command": command,
            "log_path": str(log_path),
            "return_code": outcome.return_code,
            "elapsed_seconds": outcome.elapsed_seconds,
            "timed_out_point_id": outcome.timed_out_point_id,
            "active_point": outcome.active_point,
        }
    )
    atomic_json(path, value)


def _process_failure(outcome: WorkerProcessOutcome) -> str:
    if outcome.timed_out_point_id is not None:
        return f"point {outcome.timed_out_point_id} exceeded its timeout"
    if outcome.return_code < 0:
        number = -outcome.return_code
        return f"worker died from signal {number} ({signal.strsignal(number)})"
    return f"worker exited with code {outcome.return_code}"


def _preflight_resume(
    paths: BaselinePaths,
    selected: tuple[CorpusProgramCase, ...],
    resume: bool,
) -> None:
    if resume:
        return
    existing = tuple(
        case.case_id for case in selected if paths.case_directory(case).exists()
    )
    if existing:
        raise FileExistsError(
            "frontier state already exists; pass --resume to validate and continue "
            f"({', '.join(existing[:3])})"
        )


def _load_case_failures(paths: BaselinePaths) -> dict[str, dict[str, object]]:
    path = paths.directory / "case-failures.json"
    if not path.exists():
        return {}
    value = read_object(path)
    return {
        key: item
        for key, item in value.items()
        if isinstance(item, dict)
    }


def _log(paths: BaselinePaths, message: str) -> None:
    timestamped = f"[{utc_now()}] {message}"
    append_log(paths.log_path, timestamped)
    print(timestamped, flush=True)


def _program_separator(paths: BaselinePaths) -> None:
    """Make Program boundaries obvious in both the terminal and main log."""

    append_log(paths.log_path, "\n\n")
    print("\n\n", end="", flush=True)


class _CollectionLock(AbstractContextManager[None]):
    def __init__(self, path: Path) -> None:
        self._path = path
        self._descriptor: int | None = None

    def __enter__(self) -> None:
        descriptor = os.open(self._path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            os.close(descriptor)
            raise RuntimeError(
                "another frontier controller owns this baseline"
            ) from error
        os.ftruncate(descriptor, 0)
        os.write(descriptor, f"pid={os.getpid()}\n".encode())
        self._descriptor = descriptor
        return None

    def __exit__(self, *_errors: object) -> None:
        if self._descriptor is not None:
            fcntl.flock(self._descriptor, fcntl.LOCK_UN)
            os.close(self._descriptor)
            self._descriptor = None


__all__ = ["ControllerOptions", "run_frontier_collection"]
