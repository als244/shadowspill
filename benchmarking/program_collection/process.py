"""One-case subprocess execution and result validation."""

from __future__ import annotations

import json
import os
import queue
import signal
import subprocess
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from benchmarking.program_collection.corpus import load_step_program

from .matrix import ProgramRequest
from .state import utc_now


@dataclass(frozen=True, slots=True)
class WorkerOutcome:
    """Normalized completion of a Python exception, timeout, or native exit."""

    status: str
    return_code: int | None
    elapsed_seconds: float
    error: dict[str, object] | None
    artifact: dict[str, object] | None


def execute_worker(
    command: list[str],
    *,
    repository_root: Path,
    log_path: Path,
    result_path: Path,
    timeout_seconds: int,
    console_prefix: str,
    expected_request: ProgramRequest,
) -> WorkerOutcome:
    """Run one isolated worker while streaming and preserving its output."""

    started = time.perf_counter()
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        command,
        cwd=repository_root,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    lines: queue.Queue[str | None] = queue.Queue()
    reader = threading.Thread(
        target=_read_output,
        args=(process, lines),
        name=f"corpus-log-{expected_request.case_id}",
        daemon=True,
    )
    reader.start()
    timed_out, interrupted, return_code = _stream_until_exit(
        process,
        reader,
        lines,
        log_path=log_path,
        command=command,
        deadline=started + timeout_seconds,
        console_prefix=console_prefix,
    )
    elapsed = time.perf_counter() - started
    if interrupted:
        return WorkerOutcome(
            "interrupted",
            return_code,
            elapsed,
            {
                "type": "InterruptedCollection",
                "message": "worker was terminated after a controller interrupt",
            },
            None,
        )
    if timed_out:
        return WorkerOutcome(
            "failed",
            return_code,
            elapsed,
            {
                "type": "ProgramCollectionTimeout",
                "message": f"worker exceeded {timeout_seconds} seconds",
            },
            None,
        )
    return load_worker_outcome(
        result_path,
        return_code=return_code,
        elapsed_seconds=elapsed,
        expected_request=expected_request,
    )


def load_worker_outcome(
    result_path: Path,
    *,
    return_code: int,
    elapsed_seconds: float,
    expected_request: ProgramRequest,
) -> WorkerOutcome:
    """Validate the journal and artifact produced by one worker process."""

    if not result_path.exists():
        description = (
            f"signal {-return_code} ({signal.strsignal(-return_code)})"
            if return_code < 0
            else f"exit code {return_code}"
        )
        return _failure(
            return_code,
            elapsed_seconds,
            "WorkerProcessFailure",
            f"worker produced no result after {description}",
        )
    try:
        value = json.loads(result_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        return _failure(return_code, elapsed_seconds, type(error).__name__, str(error))
    if not isinstance(value, dict):
        return _failure(
            return_code,
            elapsed_seconds,
            "InvalidWorkerResult",
            "result is not an object",
        )
    if return_code != 0 or value.get("passed") is not True:
        raw_error = value.get("error")
        failure_details: dict[str, object]
        if isinstance(raw_error, dict):
            failure_details = {str(key): item for key, item in raw_error.items()}
        else:
            failure_details = {
                "type": "WorkerProcessFailure",
                "message": f"worker returned {return_code}",
            }
        return WorkerOutcome(
            "failed", return_code, elapsed_seconds, failure_details, None
        )
    if value.get("case_id") != expected_request.case_id:
        return _failure(
            return_code,
            elapsed_seconds,
            "InvalidWorkerResult",
            "case identity changed",
        )
    raw_artifact = value.get("artifact")
    if not isinstance(raw_artifact, dict):
        return _failure(
            return_code,
            elapsed_seconds,
            "InvalidWorkerResult",
            "artifact is missing",
        )
    try:
        directory = Path(str(raw_artifact["directory"]))
        saved, program = load_step_program(directory)
        if saved.program_digest != raw_artifact.get("program_digest"):
            raise ValueError("worker artifact Program digest changed")
        if program.digest != saved.program_digest:
            raise ValueError("worker artifact content digest changed")
    except BaseException as error:
        return _failure(return_code, elapsed_seconds, type(error).__name__, str(error))
    return WorkerOutcome(
        "succeeded",
        return_code,
        elapsed_seconds,
        None,
        {str(key): item for key, item in raw_artifact.items()},
    )


def _stream_until_exit(
    process: subprocess.Popen[str],
    reader: threading.Thread,
    lines: queue.Queue[str | None],
    *,
    log_path: Path,
    command: list[str],
    deadline: float,
    console_prefix: str,
) -> tuple[bool, bool, int]:
    timed_out = False
    interrupted = False
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"COMMAND {json.dumps(command)}\n")
        log.flush()
        reader_done = False
        try:
            while not reader_done or process.poll() is None:
                line = _next_line(lines)
                if line is None:
                    reader_done = True
                elif line:
                    _write_line(log, console_prefix, line)
                if process.poll() is None and time.perf_counter() >= deadline:
                    timed_out = True
                    _terminate_process_group(process)
        except KeyboardInterrupt:
            interrupted = True
            _terminate_process_group(process)
        finally:
            return_code = process.wait()
            reader.join(timeout=5.0)
            while True:
                try:
                    remaining = lines.get_nowait()
                except queue.Empty:
                    break
                if remaining:
                    _write_line(log, console_prefix, remaining)
            log.flush()
    return timed_out, interrupted, return_code


def _next_line(lines: queue.Queue[str | None]) -> str | None:
    try:
        return lines.get(timeout=0.25)
    except queue.Empty:
        return ""


def _write_line(log: TextIO, console_prefix: str, line: str) -> None:
    stamped = f"[{utc_now()}] {line}"
    log.write(stamped)
    log.flush()
    print(f"{console_prefix} | {line}", end="", flush=True)


def _read_output(
    process: subprocess.Popen[str],
    output: queue.Queue[str | None],
) -> None:
    assert process.stdout is not None
    try:
        for line in process.stdout:
            output.put(line)
    finally:
        output.put(None)


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)


def _failure(
    return_code: int,
    elapsed_seconds: float,
    error_type: str,
    message: str,
) -> WorkerOutcome:
    return WorkerOutcome(
        "failed",
        return_code,
        elapsed_seconds,
        {"type": error_type, "message": message},
        None,
    )


__all__ = ["WorkerOutcome", "execute_worker", "load_worker_outcome"]
