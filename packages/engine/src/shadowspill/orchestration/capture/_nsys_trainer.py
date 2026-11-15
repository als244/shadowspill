"""Worker-only capture boundaries, shared by recipes and Plan benchmarks."""

from contextlib import nullcontext
from typing import Any, cast

import torch
import torch.distributed as dist

from ._nsys_runtime import capture_boundary


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
        cast(Any, super()).__init__(*args, **kwargs)

    def step(self, **inputs: Any) -> Any:
        number = self.step_count + 1
        last = self.nsys_start_step + self.nsys_steps - 1
        captured = self.nsys_steps > 0 and self.nsys_start_step <= number <= last
        if captured and number == self.nsys_start_step:
            capture_boundary(torch.cuda.profiler.start)
        rank = dist.get_rank() if dist.is_initialized() else 0
        label = f"train/step={number}/rank={rank}"
        if captured:
            inputs["profiler_annotations"] = True
        with torch.cuda.nvtx.range(label) if captured else nullcontext():
            result = cast(Any, super()).step(**inputs)
        # Trainer.step synchronizes execution before returning. Wait for every
        # worker before ending capture, including the final worker's GPU work.
        if captured and number == last:
            capture_boundary(torch.cuda.profiler.stop)
        return result
