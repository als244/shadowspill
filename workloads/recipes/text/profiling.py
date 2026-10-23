"""Bounded Nsight Systems capture around completed Trainer updates."""

from contextlib import nullcontext

import torch
import torch.distributed as dist

from shadowspill.training import Trainer


class NsightTrainer(Trainer):
    """Use with nsys --capture-range=cudaProfilerApi --capture-range-end=stop.

    Launch one nsys process around torchrun to collect all workers together.
    Rank zero controls the session; barriers bound the capture across workers.
    Step ranges exclude evaluation/checkpointing, which Trainer.fit performs
    after step returns. Set their cadences outside the captured step interval.
    """

    def __init__(self, *args, nsys_start_step=2, nsys_steps=0, **kwargs):
        if nsys_start_step < 1 or nsys_steps < 0:
            raise ValueError("nsys_start_step must be positive; nsys_steps nonnegative")
        self.nsys_start_step = nsys_start_step
        self.nsys_steps = nsys_steps
        super().__init__(*args, **kwargs)

    @staticmethod
    def _capture_boundary(action):
        distributed = dist.is_initialized()
        if distributed:
            dist.barrier()
        if not distributed or dist.get_rank() == 0:
            action()
        if distributed:
            dist.barrier()

    def step(self, data, *, hyperparams=None):
        number = self.step_count + 1
        last = self.nsys_start_step + self.nsys_steps - 1
        captured = self.nsys_steps > 0 and self.nsys_start_step <= number <= last
        if captured and number == self.nsys_start_step:
            self._capture_boundary(torch.cuda.profiler.start)
        rank = dist.get_rank() if dist.is_initialized() else 0
        label = f"train/step={number}/rank={rank}"
        with torch.cuda.nvtx.range(label) if captured else nullcontext():
            result = super().step(data, hyperparams=hyperparams)
        # Trainer.step synchronizes execution before returning. The collective
        # boundary waits for every worker before ending process-tree capture.
        if captured and number == last:
            self._capture_boundary(torch.cuda.profiler.stop)
        return result
