"""Independent pinned buffers and event timing on ordinary CUDA streams."""

import torch


class CopyProbe:
    def __init__(self, size: int, device: torch.device):
        self.host = [
            torch.empty(size, dtype=torch.uint8, pin_memory=True) for _ in range(2)
        ]
        self.gpu = [
            torch.empty(size, dtype=torch.uint8, device=device) for _ in range(2)
        ]
        for tensor in (*self.host, *self.gpu):
            tensor.zero_()
        self.streams = [torch.cuda.Stream(device=device) for _ in range(2)]  # type: ignore[no-untyped-call]
        self.starts = [torch.cuda.Event(enable_timing=True) for _ in range(2)]  # type: ignore[no-untyped-call]
        self.ends = [torch.cuda.Event(enable_timing=True) for _ in range(2)]  # type: ignore[no-untyped-call]
        self.ready = torch.cuda.Event()  # type: ignore[no-untyped-call]
        self.views: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {}
        torch.cuda.current_stream(device).synchronize()

    def sample(self, size: int, directions: tuple[int, ...]) -> tuple[int, ...]:
        if size not in self.views:
            self.views[size] = [
                (self.host[0][:size], self.gpu[0][:size]),
                (self.gpu[1][:size], self.host[1][:size]),
            ]
        # Both directions wait on the same event, with disjoint sources/destinations.
        self.ready.record()
        for index in directions:
            stream = self.streams[index]
            stream.wait_event(self.ready)
            with torch.cuda.stream(stream):
                self.starts[index].record()
                source, destination = self.views[size][index]
                destination.copy_(source, non_blocking=True)
                self.ends[index].record()
        for index in directions:
            self.ends[index].synchronize()
        return tuple(
            round(self.starts[index].elapsed_time(self.ends[index]) * 1e6)
            for index in directions
        )
