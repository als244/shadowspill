"""Small non-text experiment passed to the generic benchmark quickstart."""

from __future__ import annotations

from functools import partial

import torch
from torch import nn


def experiment(*, device, rows=64, width=16, outputs=8):
    if rows < 4 or rows % 4:
        raise ValueError("rows must be a positive multiple of four")
    generator = torch.Generator().manual_seed(7)
    features = torch.randn(rows, width, generator=generator)
    targets = torch.randn(rows, outputs, generator=generator)

    def model_factory():
        # Each budget gets exactly the same initialized values.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(11)
            return nn.Sequential(
                nn.Linear(width, 32), nn.SiLU(), nn.Linear(32, outputs)
            )

    def objective(model, features, targets):
        return (model(features) - targets).square().sum() / (rows * outputs)

    return {
        "model_factory": model_factory,
        "objective": objective,
        "optimizer": partial(torch.optim.AdamW, lr=3e-4),
        "hyperparams": {"lr": 3e-4},
        "candidates": {
            f"rows_{size}": tuple(
                (
                    features[start : start + size].clone(),
                    targets[start : start + size].clone(),
                )
                for start in range(0, rows, size)
            )
            for size in (rows // 4, rows // 2)
        },
        "units_per_step": rows,
        "unit_label": "samples",
    }
