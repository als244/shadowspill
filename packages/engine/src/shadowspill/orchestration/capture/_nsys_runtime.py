"""Worker-only CUDA profiler boundaries shared by execution adapters."""

from collections.abc import Callable

import torch.distributed as dist


def capture_boundary(action: Callable[[], object]) -> None:
    distributed = dist.is_initialized()
    if distributed:
        dist.barrier()
    action()
    if distributed:
        dist.barrier()
