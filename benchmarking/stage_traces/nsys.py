"""One isolated Nsight capture, export and verification."""

import json
import time
from pathlib import Path

from benchmarking._serialization import atomic_json
from benchmarking.processes import worker_command

from .process import launch, stamp, worker_environment
from .report import verify


def capture_command(case, options):
    attempt = Path(case["attempt"])
    command = [
        options["nsys"],
        "profile",
        "--trace=cuda,nvtx,osrt",
        "--sample=none",
        "--cpuctxsw=none",
        "--osrt-threshold=10000",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop",
        "--force-overwrite=false",
        "--output=" + str(attempt / "nsys/trace"),
    ]
    if options["gpu_metrics"]:
        # Let Nsight resolve the worker's physical device/UUID mask.
        command.extend(
            [
                "--gpu-metrics-devices=cuda-visible",
                f"--gpu-metrics-frequency={options['gpu_metrics_frequency']}",
            ]
        )
    return command + worker_command(
        "benchmarking.stage_traces.worker",
        [
            "--artifact",
            case["artifact"],
            "--output",
            attempt,
            "--stage",
            case["stage"],
            "--warmup",
            options["warmup"],
            "--iterations",
            options["iterations"],
            "--cpu-threads",
            options["cpu_threads"],
            "--capture",
        ],
    )


def verify_capture(attempt, options, env):
    report = attempt / "nsys/trace.nsys-rep"
    result = attempt / "result.json"
    if not report.is_file() or report.stat().st_size == 0 or not result.exists():
        raise ValueError("worker exited without a nonempty report/result")
    database = attempt / "nsys/trace.sqlite"
    code, timed_out = launch(
        [
            options["nsys"],
            "export",
            "--type=sqlite",
            "--output=" + str(database),
            str(report),
        ],
        attempt / "logs/export.log",
        env=env,
        timeout=300,
        pid_path=attempt / "process.json",
    )
    if code or timed_out:
        raise ValueError("Nsight SQLite export failed; inspect export.log")
    device = json.loads(result.read_text())["device_identity"]
    return verify(
        database,
        options["iterations"],
        gpu_uuid=device["uuid"] if options["gpu_metrics"] else None,
    )


def trace_case(case, gpu, options):
    attempt = Path(case["attempt"])
    attempt.mkdir(parents=True, exist_ok=False)
    begun = time.monotonic()
    outcome = {"gpu": gpu, "case": case}
    try:
        for child in ("logs", "nsys"):
            (attempt / child).mkdir()
        env = worker_environment(gpu, options["cpu_threads"])
        command = capture_command(case, options)
        atomic_json(
            attempt / "command.json",
            {
                "command": command,
                "gpu": gpu,
                "CUDA_VISIBLE_DEVICES": env["CUDA_VISIBLE_DEVICES"],
                "gpu_metrics": options["gpu_metrics"],
                "gpu_metrics_frequency_hz": options["gpu_metrics_frequency"],
                "case": case,
            },
        )
        code, timed_out = launch(
            command,
            attempt / "logs/console.log",
            env=env,
            timeout=options["timeout"],
            pid_path=attempt / "process.json",
        )
        status = "timeout" if timed_out else "completed" if code == 0 else "failed"
        failure = attempt / "failure.json"
        if failure.exists():
            status = json.loads(failure.read_text())["status"]
        outcome.update(
            status=status,
            returncode=code,
            verification=verify_capture(attempt, options, env)
            if status == "completed"
            else None,
        )
    except Exception as error:
        outcome.update(status="failed", error=repr(error))
    outcome.update(elapsed_seconds=time.monotonic() - begun, ended=stamp())
    atomic_json(attempt / "outcome.json", outcome)
    return outcome
