"""Small modules whose parameters match the pure-PyTorch references."""

from __future__ import annotations

import mlops
import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """RMSNorm parameter container using the external optimized operation."""

    def __init__(self, width: int, *, epsilon: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.epsilon = float(epsilon)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return mlops.rms_norm(value, self.weight, self.epsilon)


class GatedRMSNorm(RMSNorm):
    """Gated RMSNorm parameter container using the external operation."""

    # A gate is a second input, so this deliberately does not accept what
    # RMSNorm accepts. The base class is here for the parameter it holds.
    def forward(  # type: ignore[override]
        self, value: torch.Tensor, gate: torch.Tensor
    ) -> torch.Tensor:
        return mlops.gated_rms_norm(value, gate, self.weight, self.epsilon)


__all__ = ["GatedRMSNorm", "RMSNorm"]
