"""CPU-only Nsight launch settings for execution workers."""

import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from shutil import which
from typing import Self, cast


@dataclass(frozen=True)
class NsysCapture:
    """Collect a contiguous interval of one-based, complete training updates.

    ``rank_command`` runs inside the distributed launcher, before CUDA imports,
    to isolate one GPU and start one collector per rank. The worker uses the
    existing cudaProfilerApi boundaries and independent ``profiler_annotations``
    toggle. ``output_dir=None`` defaults to the benchmark run's profiling folder;
    custom launchers supply their common output directory to ``rank_command``.
    Hardware metrics require permission on the selected devices.
    """

    start_step: int = 2
    steps: int = 2
    gpu_metrics: bool = True
    gpu_metrics_frequency: int = 1000
    executable: str = "nsys"
    output_dir: str | Path | None = None
    # Buffer CUDA activity across short update windows. On CUDA 11+, the
    # collector's flush-on-fill mode can stall the worker inside a launch.
    cuda_flush_interval_ms: int = 60_000

    def __post_init__(self) -> None:
        for name in ("start_step", "steps"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.gpu_metrics) is not bool:
            raise TypeError("gpu_metrics must be a bool")
        if (
            type(self.cuda_flush_interval_ms) is not int
            or self.cuda_flush_interval_ms < 0
        ):
            raise ValueError("cuda_flush_interval_ms must be a nonnegative integer")
        if (
            type(self.gpu_metrics_frequency) is not int
            or not 10 <= self.gpu_metrics_frequency <= 200_000
        ):
            raise ValueError("gpu_metrics_frequency must be 10-200000 Hz")
        if not isinstance(self.executable, str) or not self.executable:
            raise ValueError("executable must be a nonempty Nsight executable name")
        if self.output_dir is not None:
            if not isinstance(self.output_dir, (str, Path)) or not str(self.output_dir):
                raise ValueError("output_dir must be a nonempty path or None")
            object.__setattr__(self, "output_dir", str(self.output_dir))

    @classmethod
    def from_settings(cls, settings: Mapping[str, object] | None) -> Self | None:
        """Resolve portable capture choices; executables and paths stay local."""
        if settings is None:
            return None
        if set(settings) - {
            "start_step",
            "steps",
            "gpu_metrics",
            "gpu_metrics_frequency",
            "cuda_flush_interval_ms",
        }:
            raise ValueError(
                "service Nsight settings cannot select paths or executables"
            )
        return cls(
            start_step=cast(int, settings.get("start_step", 2)),
            steps=cast(int, settings.get("steps", 2)),
            gpu_metrics=cast(bool, settings.get("gpu_metrics", True)),
            gpu_metrics_frequency=cast(
                int, settings.get("gpu_metrics_frequency", 1000)
            ),
            cuda_flush_interval_ms=cast(
                int, settings.get("cuda_flush_interval_ms", cls.cuda_flush_interval_ms)
            ),
        )

    @property
    def step_numbers(self) -> range:
        return range(self.start_step, self.start_step + self.steps)

    def command(
        self, worker_command: Iterable[object], *, output: str | Path
    ) -> list[str]:
        """Collector command for one already isolated rank; output is a stem.

        Prefer ``rank_command`` to mask devices and check the resulting report.
        Do not wrap torchrun itself: each rank needs its own collector.
        """
        executable = which(self.executable)
        if executable is None:
            raise FileNotFoundError(f"Nsight Systems executable: {self.executable}")
        command = [
            executable,
            "profile",
            "--trace=cuda,nvtx,osrt",
            f"--cuda-flush-interval={self.cuda_flush_interval_ms}",
            "--sample=none",
            "--cpuctxsw=none",
            "--osrt-threshold=10000",
            "--capture-range=cudaProfilerApi",
            # Keep the session open after our single start/stop pair. `stop`
            # imports the report while the worker still holds its pinned pool;
            # `repeat` ends collection but finalizes after the worker exits.
            # Do not use repeat:1: reaching its limit finalizes immediately.
            "--capture-range-end=repeat",
            "--wait=all",
            "--force-overwrite=false",
            f"--output={Path(output)}",
        ]
        if self.gpu_metrics:
            command += [
                "--gpu-metrics-devices=cuda-visible",
                f"--gpu-metrics-frequency={self.gpu_metrics_frequency}",
            ]
        return command + list(map(str, worker_command))

    def rank_command(
        self, worker_command: Sequence[object], *, output_dir: str | Path | None = None
    ) -> list[str]:
        """Wrap one worker command, suitable for torchrun's ``--no-python``.

        Each rank writes ``output_dir/rank-NNNNN/{trace.nsys-rep,capture.json}``.
        The directory may be shared across nodes or collected from each node
        afterward. Global rank names do not collide between nodes. Capture
        boundaries still belong to the worker's update-capture configuration.
        """
        directory = self.output_dir if output_dir is None else output_dir
        if directory is None:
            raise ValueError(
                "rank_command needs output_dir or NsysCapture(output_dir=...)"
            )
        return [
            sys.executable,
            *(["-S"] if sys.flags.no_site else []),
            "-u",
            "-m",
            "shadowspill.orchestration.capture.nsys_worker",
            "--config",
            json.dumps(asdict(self)),
            "--output-dir",
            str(directory),
            "--",
            *map(str, worker_command),
        ]
