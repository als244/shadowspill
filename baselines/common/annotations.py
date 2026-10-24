"""Optional NVTX launch scopes; they never synchronize the accelerator.

CUDA API correlation, rather than the duration of these host ranges, tells
an analyzer which GPU kernels belong to checkpoint recomputation.
"""

from contextlib import nullcontext


def phase(name: str, enabled: bool):
    if not enabled:
        return nullcontext()
    import torch

    return torch.cuda.nvtx.range(f"baseline.phase/{name}")


def checkpoint_contexts():
    """PyTorch non-reentrant checkpoint's first forward and its replay."""

    return phase("checkpoint_forward", True), phase("recompute", True)
