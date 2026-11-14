"""Small modules whose parameters match the pure-PyTorch references."""

from __future__ import annotations

import mlops
import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """RMSNorm parameter container using the external optimized operation.

    A zero-centered norm keeps its weight as an offset from one, as Qwen 3.5
    does: the operation's zero-centered mode multiplies the normalized value
    by ``1 + weight`` in float32 before the cast, so the stored weights are
    small and the scale is kept at full precision."""

    def __init__(
        self, width: int, *, epsilon: float = 1e-5, zero_centered: bool = False
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(width))
        self.epsilon = float(epsilon)
        self.zero_centered = bool(zero_centered)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialise in place, so the values land wherever the
        parameter already lives."""

        with torch.no_grad():
            self.weight.fill_(0.0 if self.zero_centered else 1.0)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return mlops.rms_norm(
            value, self.weight, self.epsilon, zero_centered=self.zero_centered
        )


class GatedRMSNorm(RMSNorm):
    """Gated RMSNorm parameter container using the external operation."""

    # A gate is a second input, so this deliberately does not accept what
    # RMSNorm accepts. The base class is here for the parameter it holds.
    def forward(  # type: ignore[override]
        self, value: torch.Tensor, gate: torch.Tensor
    ) -> torch.Tensor:
        return mlops.gated_rms_norm(value, gate, self.weight, self.epsilon)


__all__ = ["GatedRMSNorm", "RMSNorm"]
