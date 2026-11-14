"""Ordinary model code; only the build phase imports this module."""

import torch
from torch import nn


class Regression(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(16, 32), nn.SiLU(), nn.Linear(32, 8))
        self.register_buffer("updates", torch.tensor(0, dtype=torch.int64))

    def forward(self, value):
        self.updates.add_(1)
        return self.network(value)


def objective(model, inputs, *, targets, ranks):
    # Each rank contributes to a global mean. Gradient reductions use SUM.
    return (model(inputs) - targets).square().sum() / (ranks * targets.numel())
