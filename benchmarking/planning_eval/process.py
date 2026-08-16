"""Stream and monitor one Program worker with a per-active-point timeout."""

from __future__ import annotations

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

from .storage import read_active_point, utc_now


@dataclass(frozen=True, slots=True)
class WorkerProcessOutcome:
    return_code: int
    elapsed_seconds: float
    timed_out_point_id: str | None
    active_point: tuple[str, str] | None
    active_point_elapsed_seconds: float | None


def execute_case_worker(
    command: list[str],
    *,
    repository_root: Path,
    case_run_directory: Path,
    log_path: Path,
    collection_log_path: Path,
    point_timeout_seconds: int,
    console_prefix: str,
) -> WorkerProcessOutcome:
    """Run a worker and kill only when its active PressureFit point times out."""

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
        name=f"frontier-log-{console_prefix}",
        daemon=True,
    )
    reader.start()
    active: tuple[str, str] | None = None
    active_started = time.perf_counter()
    timed_out_point_id: str | None = None
    log_path.parent.mkdir(parents=True, exist_ok=True)
    collection_log_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        log_path.open("a", encoding="utf-8", buffering=1) as log,
        collection_log_path.open("a", encoding="utf-8", buffering=1) as collection_log,
    ):
        log.write(f"COMMAND {command!r}\n")
        log.flush()
        reader_done = False
        try:
            while not reader_done or process.poll() is None:
                line = _next_line(lines)
                if line is None:
                    reader_done = True
                elif line:
                    _write_line(log, collection_log, console_prefix, line)
                observed = read_active_point(case_run_directory)
                if observed != active:
                    active = observed
                    active_started = time.perf_counter()
                if (
                    active is not None
                    and process.poll() is None
                    and time.perf_counter() - active_started >= point_timeout_seconds
                ):
                    timed_out_point_id = active[0]
                    _terminate_process_group(process)
        except KeyboardInterrupt:
            _terminate_process_group(process)
            raise
        finally:
            return_code = process.wait()
            reader.join(timeout=5.0)
            while True:
                try:
                    remaining = lines.get_nowait()
                except queue.Empty:
                    break
                if remaining:
                    _write_line(log, collection_log, console_prefix, remaining)
            log.flush()
            collection_log.flush()
    final_active = read_active_point(case_run_directory)
    return WorkerProcessOutcome(
        return_code,
        time.perf_counter() - started,
        timed_out_point_id,
        final_active,
        None if final_active is None else time.perf_counter() - active_started,
    )


def _next_line(lines: queue.Queue[str | None]) -> str | None:
    try:
        return lines.get(timeout=0.25)
    except queue.Empty:
        return ""


def _write_line(
    log: TextIO,
    collection_log: TextIO,
    prefix: str,
    line: str,
) -> None:
    if not line.strip():
        log.write(line)
        log.flush()
        collection_log.write(line)
        collection_log.flush()
        print(line, end="", flush=True)
        return
    timestamp = utc_now()
    log.write(f"[{timestamp}] {line}")
    log.flush()
    collection_log.write(f"[{timestamp}] {prefix} | {line}")
    collection_log.flush()
    print(f"{prefix} | {line}", end="", flush=True)


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
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)


__all__ = ["WorkerProcessOutcome", "execute_case_worker"]
