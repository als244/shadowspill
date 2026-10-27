"""Run one case with timeout and process-group cleanup."""

from __future__ import annotations

import datetime
import os
import signal
import subprocess
import time

from baselines.common import write_json
from baselines.common.provenance import ROOT

from .results import classify


def stop(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def run_case(
    command_line, attempt, settings, world_size, timeout, *, classify_result=None
):
    """Run one attempt to completion, timeout or interruption, recording its
    console, its process and its outcome beside it."""

    started = time.monotonic()
    environment = os.environ.copy()
    threads = getattr(settings, "cpu_threads", None)
    if threads is not None:
        environment["OMP_NUM_THREADS"] = str(threads)
        environment["MKL_NUM_THREADS"] = str(threads)
    with (attempt / "console.log").open("w") as log:
        process = subprocess.Popen(
            command_line,
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        write_json(
            attempt / "process.json", {"pid": process.pid, "command": command_line}
        )
        try:
            returncode = process.wait(timeout=timeout)
            try:
                outcome = (classify_result or classify)(
                    attempt, returncode, world_size, settings
                )
            except (ValueError, KeyError, TypeError) as error:
                outcome = {"status": "failed", "reason": f"invalid result: {error}"}
            outcome["returncode"] = returncode
        except subprocess.TimeoutExpired:
            stop(process)
            outcome = {"status": "timeout", "reason": f"exceeded {timeout} seconds"}
        except BaseException:
            stop(process)
            write_json(
                attempt / "outcome.json",
                {"status": "interrupted", "attempt": attempt.name},
            )
            raise
    outcome.update(
        {
            "attempt": attempt.name,
            "wall_seconds": time.monotonic() - started,
            "finished_at": datetime.datetime.now().astimezone().isoformat(),
        }
    )
    write_json(attempt / "outcome.json", outcome)
    return outcome
