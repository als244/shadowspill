"""Host orchestration keeps assignments isolated, resumable and verifiable."""

import json
import os
import signal
import sqlite3
import sys
import threading
from pathlib import Path

import pytest

from benchmarking._serialization import atomic_json
from benchmarking.stage_traces import nsys
from benchmarking.stage_traces.process import launch
from benchmarking.stage_traces.queue import run_queue, select_attempts


def trace_options():
    return {
        "nsys": "nsys",
        "gpu_metrics": False,
        "gpu_metrics_frequency": 10000,
        "warmup": 1,
        "iterations": 2,
        "cpu_threads": 1,
        "timeout": 60,
    }


def test_queue_records_failures_and_retries_only_unfinished_cases(
    tmp_path, monkeypatch
):
    cases = [
        {
            "stage": f"stage-{i}",
            "sequence_length": 8,
            "batch_size": 1,
            "variant": "recompute",
            "artifact": f"pair-{i}.pt",
            "directory": str(tmp_path / "results" / f"case-{i}"),
        }
        for i in range(4)
    ]
    # Exercise simultaneous GPU assignment without needing GPUs or Nsight.
    barrier = threading.Barrier(2, timeout=5)
    launched = []

    def fake_launch(command, log_path, *, env, **kwargs):
        output = Path(
            next(v.split("=", 1)[1] for v in command if v.startswith("--output="))
        )
        if command[1] == "profile":
            gpu = env["CUDA_VISIBLE_DEVICES"]
            stage = command[command.index("--stage") + 1]
            launched.append((stage, gpu))
            # Every worker child preserves the interpreter's isolated import mode.
            python = command.index(sys.executable)
            assert ("-S" in command[python:]) == bool(sys.flags.no_site)
            assert env["MLOPS_QUACK_TUNE_DISTRIBUTED"] == "0"
            if stage in {"stage-0", "stage-1"}:
                barrier.wait()
            attempt = output.parent.parent
            if stage == "stage-2":
                atomic_json(attempt / "failure.json", {"status": "out_of_memory"})
                return 1, False
            if stage == "stage-3":
                return 0, False  # A zero exit without a capture is still a failure.
            output.with_suffix(".nsys-rep").write_text("fake report")
            atomic_json(attempt / "result.json", {"device_identity": {}, "samples": []})
        else:
            with sqlite3.connect(output) as db:
                db.execute(
                    "create table NVTX_EVENTS (start integer, end integer, text text)"
                )
                db.execute("create table CUPTI_ACTIVITY_KIND_KERNEL (start integer)")
                for iteration in range(2):
                    db.execute(
                        "insert into CUPTI_ACTIVITY_KIND_KERNEL values (?)",
                        (iteration,),
                    )
                    for text in (f"iteration/{iteration}", "forward", "backward"):
                        db.execute("insert into NVTX_EVENTS values (1, 2, ?)", (text,))
        return 0, False

    monkeypatch.setattr(nsys, "launch", fake_launch)
    with pytest.raises(RuntimeError, match="trace cases failed"):
        run_queue(cases, ["GPU-first", "GPU-second"], trace_options(), tmp_path)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["outcomes"] == {"completed": 2, "out_of_memory": 1, "failed": 1}
    assert len(launched) == 4
    assert len({gpu for stage, gpu in launched if stage in {"stage-0", "stage-1"}}) == 2
    assert "stage-3" in (tmp_path / "README.md").read_text()
    assert not json.loads((tmp_path / "progress.json").read_text())["active"]
    pending, finished = select_attempts(cases, retry_failed=False)
    assert not pending and len(finished) == 4
    pending, finished = select_attempts(cases, retry_failed=True)
    assert {case["stage"] for case in pending} == {"stage-2", "stage-3"}
    assert all(Path(case["attempt"]).name == "attempt-002" for case in pending)
    assert len(finished) == 2


def test_timeout_stops_the_worker_process_group(tmp_path):
    pid_file = tmp_path / "process.json"
    code, timed_out = launch(
        [
            sys.executable,
            *(["-S"] if sys.flags.no_site else []),
            "-c",
            "import time; time.sleep(30)",
        ],
        tmp_path / "worker.log",
        env=os.environ.copy(),
        timeout=0.2,
        pid_path=pid_file,
    )
    assert timed_out and code == -signal.SIGTERM
    with pytest.raises(ProcessLookupError):
        os.killpg(json.loads(pid_file.read_text())["process_group"], 0)
