"""Small provider failures used at the public preparation boundaries."""

import torch
from torch import nn

profiling_failure = False


@torch.library.custom_op("shadowspill_phase_failure::opaque", mutates_args=())
def opaque(value: torch.Tensor) -> torch.Tensor:
    if profiling_failure:
        # Exercise OOM propagation without exhausting a shared physical device.
        raise torch.OutOfMemoryError("injected provider workspace allocation failure")
    return value.sin()


@opaque.register_fake
def opaque_fake(value):
    return torch.empty_like(value)


@torch.library.custom_op(
    "shadowspill_phase_failure::cpu_only", mutates_args=(), device_types="cpu"
)
def cpu_only(value: torch.Tensor) -> torch.Tensor:
    return value.sin()


@cpu_only.register_fake
def cpu_only_fake(value):
    return torch.empty_like(value)


class Opaque(nn.Module):
    def __init__(self, missing_backend=False):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))
        self.missing_backend = missing_backend

    def forward(self, value):
        operation = cpu_only if self.missing_backend else opaque
        return operation(value * self.scale)


class DataDependent(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, value):
        scaled = value * self.scale
        return scaled if scaled.sum().item() > 0 else -scaled
