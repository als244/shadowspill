"""Worker-only capture boundaries, shared by recipes and Plan benchmarks."""

from contextlib import nullcontext

import torch
import torch.distributed as dist


class NsightTrainerMixin:
    """Use under nsys --capture-range=cudaProfilerApi --capture-range-end=stop.

    Every rank controls its own collector. Barriers bound the interval across
    workers; per-worker NVTX ranges identify each complete update.
    Trainer.fit evaluation/checkpoint work occurs after step returns.
    """

    def __init__(self, *args, nsys_start_step=2, nsys_steps=0, **kwargs):
        if type(nsys_start_step) is not int or nsys_start_step < 1:
            raise ValueError("nsys_start_step must be a positive integer")
        if type(nsys_steps) is not int or nsys_steps < 0:
            raise ValueError("nsys_steps must be a nonnegative integer")
        self.nsys_start_step = nsys_start_step
        self.nsys_steps = nsys_steps
        super().__init__(*args, **kwargs)

    @staticmethod
    def _capture_boundary(action):
        distributed = dist.is_initialized()
        if distributed:
            dist.barrier()
        action()
        if distributed:
            dist.barrier()

    def step(self, **inputs):
        number = self.step_count + 1
        last = self.nsys_start_step + self.nsys_steps - 1
        captured = self.nsys_steps > 0 and self.nsys_start_step <= number <= last
        if captured and number == self.nsys_start_step:
            self._capture_boundary(torch.cuda.profiler.start)
        rank = dist.get_rank() if dist.is_initialized() else 0
        label = f"train/step={number}/rank={rank}"
        if captured:
            inputs["profiler_annotations"] = True
        with torch.cuda.nvtx.range(label) if captured else nullcontext():
            result = super().step(**inputs)
        # Trainer.step synchronizes execution before returning. Wait for every
        # worker before ending capture, including the final worker's GPU work.
        if captured and number == last:
            self._capture_boundary(torch.cuda.profiler.stop)
        return result
