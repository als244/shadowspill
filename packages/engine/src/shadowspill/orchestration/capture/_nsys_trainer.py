"""Worker-only capture boundaries, shared by recipes and Plan benchmarks."""

from typing import Any, cast

import torch.distributed as dist

from ._nsys_runtime import UpdateCapture
from .nsys import NsysCapture


class NsightTrainerMixin:
    """Use under nsys --capture-range=cudaProfilerApi --capture-range-end=repeat.

    Every rank controls its own collector. Barriers bound the interval across
    workers; per-worker NVTX ranges identify each complete update.
    Trainer.fit evaluation/checkpoint work occurs after step returns. This mixin
    emits one start/stop pair; leaving the session open defers report conversion
    until process exit, after the worker releases its runtime and pinned pool.
    """

    step_count: int

    def __init__(
        self, *args: Any, nsys_start_step: int = 2, nsys_steps: int = 0, **kwargs: Any
    ) -> None:
        if type(nsys_start_step) is not int or nsys_start_step < 1:
            raise ValueError("nsys_start_step must be a positive integer")
        if type(nsys_steps) is not int or nsys_steps < 0:
            raise ValueError("nsys_steps must be a nonnegative integer")
        self.nsys_start_step = nsys_start_step
        self.nsys_steps = nsys_steps
        self._nsys_capture = (
            None
            if nsys_steps == 0
            else UpdateCapture(
                NsysCapture(start_step=nsys_start_step, steps=nsys_steps)
            )
        )
        cast(Any, super()).__init__(*args, **kwargs)

    def step(self, **inputs: Any) -> Any:
        window = self._nsys_capture
        if window is None:
            return cast(Any, super()).step(**inputs)
        number = self.step_count + 1
        captured = window.begin(number, can_start=True, request_id=f"step-{number}")
        rank = dist.get_rank() if dist.is_initialized() else 0
        label = f"train/step={number}/rank={rank}"
        if captured:
            inputs["profiler_annotations"] = True
        try:
            with window.annotation(label):
                result = cast(Any, super()).step(**inputs)
        except BaseException:
            window.failed("Training execution failed during capture.")
            raise
        # Trainer.step synchronizes execution before returning. Wait for every
        # worker before ending capture, including the final worker's GPU work.
        window.complete(number, request_id=f"step-{number}")
        return result
