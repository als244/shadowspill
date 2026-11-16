"""FP32 recurrent decay parameters, independent of projection precision."""

from collections.abc import Callable
from typing import Self

import torch
from torch import nn


class DecayParameters(nn.Module):
    """Own the decay state; child projections retain ordinary Module casting."""

    def __init__(self, heads: int) -> None:
        super().__init__()
        self.A_log = nn.Parameter(torch.empty(heads, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.empty(heads, dtype=torch.float32))

    def _apply(
        self, function: Callable[[torch.Tensor], torch.Tensor], recurse: bool = True
    ) -> Self:
        if recurse:
            for child in self.children():
                child._apply(function)

        def preserve_precision(value: torch.Tensor) -> torch.Tensor:
            converted = function(value)
            if converted.is_floating_point() and converted.dtype != torch.float32:
                # Move the original value: casting back from BF16 would have
                # already rounded both the parameter and any existing gradient.
                return value.to(device=converted.device, dtype=torch.float32)
            return converted

        return super()._apply(preserve_precision, recurse=False)
