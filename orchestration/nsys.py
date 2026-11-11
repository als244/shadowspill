"""CPU-only Nsight launch settings and the shared Trainer capture adapter."""

from dataclasses import dataclass
from pathlib import Path
from shutil import which


@dataclass(frozen=True)
class NsysCapture:
    """Collect a contiguous interval of one-based, complete training updates.

    ``command`` wraps a worker launcher (including torchrun) in one Nsight
    process-tree capture. The worker uses the existing cudaProfilerApi capture
    boundaries and ShadowSpill's independent ``profiler_annotations`` toggle.
    Hardware metrics require permission on the selected devices.
    """

    start_step: int = 2
    steps: int = 2
    gpu_metrics: bool = True
    gpu_metrics_frequency: int = 1000
    executable: str = "nsys"

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

    @property
    def step_numbers(self):
        return range(self.start_step, self.start_step + self.steps)

    def command(self, worker_command, *, output):
        """Launch one collector around all workers; ``output`` is a report stem."""
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
            "--capture-range-end=stop",
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


def nsight_trainer(base):
    """Add bounded capture to a Trainer subclass, preserving its other hooks."""
    from ._nsys_trainer import NsightTrainerMixin

    if issubclass(base, NsightTrainerMixin):
        return base

    class NsightTrainer(NsightTrainerMixin, base):
        pass

    return NsightTrainer
