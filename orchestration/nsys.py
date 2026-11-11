"""CPU-only Nsight launch settings and the shared Trainer capture adapter."""

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from shutil import which


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

    def __post_init__(self):
        for name in ("start_step", "steps"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.gpu_metrics) is not bool:
            raise TypeError("gpu_metrics must be a bool")
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

    @property
    def step_numbers(self):
        return range(self.start_step, self.start_step + self.steps)

    def command(self, worker_command, *, output):
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

    def rank_command(self, worker_command, *, output_dir=None):
        """Wrap one worker command, suitable for torchrun's ``--no-python``.

        Each rank writes ``output_dir/rank-NNNNN/{trace.nsys-rep,capture.json}``.
        The directory may be shared across nodes or collected from each node
        afterward. Global rank names do not collide between nodes. Capture
        boundaries still belong to the worker's Trainer configuration.
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
            "orchestration.nsys_worker",
            "--config",
            json.dumps(asdict(self)),
            "--output-dir",
            str(directory),
            "--",
            *map(str, worker_command),
        ]


def nsight_trainer(base):
    """Add bounded capture to a Trainer subclass, preserving its other hooks."""
    from ._nsys_trainer import NsightTrainerMixin

    if issubclass(base, NsightTrainerMixin):
        return base

    class NsightTrainer(NsightTrainerMixin, base):
        pass

    return NsightTrainer
