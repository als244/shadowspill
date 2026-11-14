"""Norms of pre-update compute weights and final optimizer gradients."""

import math
from typing import Any

import torch

from ..admission import BoundTask
from ..distributed.worker import DistributedWorker
from ..paths import ExecutionPath
from .bindings import norm_inputs


class ParameterNorms:
    def __init__(self, path: ExecutionPath, worker: DistributedWorker | None):
        from triton.runtime.jit import MockTensor  # type: ignore[import-untyped]

        from .kernels import BLOCK, PARTIALS, grid, norm_partials, norm_sum

        self.worker = worker
        self.names, self.reads = norm_inputs(
            path, 0 if worker is None else worker.groups.logical_rank
        )
        self.scratch = torch.empty(
            PARTIALS, device=path.storage.device, dtype=torch.float32
        )
        self.sums = torch.empty(
            2 * len(self.names), device=path.storage.device, dtype=torch.float32
        )
        self.bytes = (PARTIALS + 2 * len(self.names)) * 4
        self._parts, self._sum, self._grid = norm_partials, norm_sum, grid
        self._block, self._partials = BLOCK, PARTIALS
        for reads in self.reads.values():
            for read in reads:
                if not read.contribute or not read.elements:
                    continue
                norm_partials.warmup(
                    MockTensor(getattr(torch, read.dtype.removeprefix("torch."))),
                    self.scratch,
                    read.elements,
                    read.shape,
                    read.stride,
                    BLOCK,
                    grid=(grid(read.elements),),
                    enable_fp_fusion=False,
                )
        norm_sum.warmup(self.scratch, self.sums, 0, PARTIALS, PARTIALS, grid=(1,))

    def begin(self) -> dict[str, dict[str, torch.Tensor]]:
        self.sums.zero_()
        # Each invocation owns its CPU results; GPU buffers are reused only
        # after that invocation's completion handle has been resolved.
        return {
            name: {
                kind: torch.empty((), dtype=torch.float32)
                for kind in ("param_norm", "grad_norm")
            }
            for name in self.names
        }

    def observe(self, bound: BoundTask, arguments: list[Any]) -> None:
        for read in self.reads.get(bound.task.task_id, ()):
            if not read.contribute or not read.elements:
                continue
            tensor = arguments[read.argument]
            with torch.cuda.nvtx.range("parameter_norms"):
                count = self._grid(read.elements)
                self._parts[(count,)](
                    tensor,
                    self.scratch,
                    read.elements,
                    read.shape,
                    read.stride,
                    self._block,
                    enable_fp_fusion=False,
                )
                self._sum[(1,)](
                    self.scratch, self.sums, read.column, count, self._partials
                )

    def finish(self, output: dict[str, dict[str, torch.Tensor]]) -> None:
        # The invocation event already joins every task's reads. This is a
        # bounded scalar copy, never a download of model weights or gradients.
        local = self.sums.cpu().tolist()
        values = [local] if self.worker is None else self.worker.groups.gather(local)
        for index, name in enumerate(self.names):
            for offset, kind in enumerate(("param_norm", "grad_norm")):
                total = math.fsum(value[2 * index + offset] for value in values)
                output[name][kind].fill_(math.sqrt(total))

    def close(self) -> None:
        del self.scratch, self.sums
