"""Bounded collective buffers and completion-aware timings; no runtime pools."""

from time import perf_counter_ns

import torch
import torch.distributed as dist

from .operations import launch_collective
from .requests import CollectiveRequest


class CollectiveProbe:
    def __init__(
        self,
        request: CollectiveRequest,
        group: dist.ProcessGroup,
        device: torch.device,
    ) -> None:
        self.request, self.group, self.device = request, group, device
        self.value = request.ranks[dist.get_rank(group)] % 16 + 1
        count = len(request.ranks)
        elements = request.elements
        input_elements = (
            elements // count if request.operation == "all_gather" else elements
        )
        output_elements = (
            elements // count if request.operation == "reduce_scatter" else elements
        )
        dtype = getattr(torch, request.dtype.removeprefix("torch."))
        if request.operation == "all_gather" and request.inplace:
            self.output = torch.empty(output_elements, dtype=dtype, device=device)
            self.input = self.output.narrow(
                0, dist.get_rank(group) * input_elements, input_elements
            )
            self.input.fill_(self.value)
        else:
            self.input = torch.full(
                (input_elements,), self.value, dtype=dtype, device=device
            )
            if request.operation == "all_reduce":
                self.output = self.input
            elif request.operation == "reduce_scatter" and request.inplace:
                self.output = self.input.narrow(
                    0, dist.get_rank(group) * output_elements, output_elements
                )
            else:
                self.output = torch.empty(output_elements, dtype=dtype, device=device)
        self.stream = None
        if device.type == "cuda":
            self.stream = torch.cuda.Stream(device=device)  # type: ignore[no-untyped-call]
            self.stream.wait_stream(torch.cuda.current_stream(device))
            self.start = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
            self.end = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]

    def _reset(self) -> None:
        if self.request.operation == "all_reduce" or (
            self.request.operation == "reduce_scatter" and self.request.inplace
        ):
            # Repetition must not repeatedly multiply the preceding output.
            # This reset precedes the start event and needs no second buffer.
            self.input.fill_(self.value)

    def _launch(self) -> None:
        work = launch_collective(
            self.request.operation, self.output, self.input, self.group
        )
        # NCCL joins its communication stream to the current stream here. The
        # end event therefore includes completion, not just host submission.
        work.wait()

    def sample(self) -> int:
        if self.stream is None:
            # CPU/Gloo exercises the same launch/order path in tests. Public
            # hardware calibration resolves only CUDA devices.
            self._reset()
            start = perf_counter_ns()
            self._launch()
            return perf_counter_ns() - start
        with torch.cuda.stream(self.stream):
            self._reset()
            self.start.record()
            self._launch()
            self.end.record()
        self.end.synchronize()
        return round(float(self.start.elapsed_time(self.end)) * 1e6)

    def zero_seconds(self) -> float | None:
        """Price the zero performed by the native gradient-fetch callback."""
        if self.stream is None or not (
            self.request.operation == "reduce_scatter" and self.request.inplace
        ):
            return None
        from statistics import median

        samples = []
        for _ in range(5):
            with torch.cuda.stream(self.stream):
                self.start.record()
                self.input.zero_()
                self.end.record()
            self.end.synchronize()
            samples.append(float(self.start.elapsed_time(self.end)) / 1000)
        return median(samples)
