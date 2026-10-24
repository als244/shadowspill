"""Submit a packed collective; its caller owns completion and buffer lifetime."""

from typing import cast

import torch
import torch.distributed as dist


def launch_collective(
    operation: str,
    output: torch.Tensor,
    source: torch.Tensor,
    group: dist.ProcessGroup,
) -> dist.Work:
    if operation == "all_gather":
        work = dist.all_gather_single(output, source, group=group, async_op=True)
    elif operation == "reduce_scatter":
        work = dist.reduce_scatter_single(output, source, group=group, async_op=True)
    elif operation == "all_reduce":
        if output is not source:
            raise ValueError("all-reduce must update its input in place")
        work = dist.all_reduce(output, group=group, async_op=True)
    else:
        raise ValueError(f"unsupported framework collective: {operation}")
    assert work is not None
    return cast(dist.Work, work)
