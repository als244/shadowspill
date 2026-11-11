"""External sampled device metrics; no CUDA work or per-step decoding."""

import json
import os
import subprocess
import warnings
from contextlib import ExitStack
from pathlib import Path

FIELDS = (
    "timestamp,index,uuid,name,utilization.gpu,memory.used,power.draw,"
    "clocks.sm,clocks.mem,temperature.gpu"
)


class GpuMonitor:
    """Capture nvidia-smi output in a child, including startup and teardown.

    Metrics are optional observations: unavailability is recorded explicitly
    and does not change training. Analysis runs separately after GPUs release.
    ``devices=None`` uses the caller's visible mask, or all GPUs when unmasked.
    """

    def __init__(self, directory, *, devices=None, interval_ms=1000):
        if type(interval_ms) is not int or interval_ms <= 0:
            raise ValueError("interval_ms must be positive")
        self.directory = Path(directory)
        self.devices = (
            os.environ.get("CUDA_VISIBLE_DEVICES") if devices is None else devices
        )
        self.interval_ms = interval_ms
        self.process = None
        self.stack = ExitStack()
        self.status = {}

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        command = [
            "nvidia-smi",
            f"--query-gpu={FIELDS}",
            "--format=csv",
            "-lms",
            str(self.interval_ms),
        ]
        if self.devices:
            command += ["--id", self.devices]
        self.status = dict(
            command=command, interval_ms=self.interval_ms, timezone="UTC"
        )
        try:
            output = self.stack.enter_context(
                (self.directory / "device-metrics.csv").open("x")
            )
            errors = self.stack.enter_context(
                (self.directory / "device-metrics.stderr").open("x")
            )
            self.process = subprocess.Popen(
                command, stdout=output, stderr=errors, env={**os.environ, "TZ": "UTC"}
            )
        except OSError as error:
            self.status.update(status="unavailable", error=str(error))
            warnings.warn(f"GPU monitoring unavailable: {error}", stacklevel=2)
        return self

    def __exit__(self, *exc):
        try:
            if self.process is not None:
                if self.process.poll() is None:
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
                        self.process.wait()
                    self.status["status"] = "stopped"
                else:
                    self.status["status"] = "exited_early"
                    warnings.warn(
                        "GPU monitor exited early; inspect device-metrics.stderr",
                        stacklevel=2,
                    )
                self.status["returncode"] = self.process.returncode
        finally:
            self.stack.close()
            (self.directory / "monitor.json").write_text(
                json.dumps(self.status, indent=2) + "\n"
            )
