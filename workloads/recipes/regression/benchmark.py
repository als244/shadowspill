"""Small non-text experiment passed to the generic benchmark quickstart."""

from __future__ import annotations

from functools import partial
from hashlib import sha256

import torch
from torch import nn


def experiment(*, device, rows=64, width=16, outputs=8, ranks=1, logical_rank=0):
    from shadowspill import Parallelism

    parallelism = Parallelism.dp(ranks_per_replica=ranks)
    if not 0 <= logical_rank < ranks:
        raise ValueError("logical_rank is outside the replica")
    if rows < 4 or rows % 4:
        raise ValueError("rows must be a positive multiple of four")
    generator = torch.Generator().manual_seed(7 + logical_rank)
    features = torch.randn(rows, width, generator=generator)
    targets = torch.randn(rows, outputs, generator=generator)

    def model_factory():
        # Build needs geometry only; execution fills the saved Plan's buffers.
        with torch.device("meta"):
            return nn.Sequential(
                nn.Linear(width, 32), nn.SiLU(), nn.Linear(32, outputs)
            )

    def objective(model, features, *, targets):
        return (model(features) - targets).square().sum() / (rows * outputs * ranks)

    def fill_tensor(request, chunk):
        # A deterministic small example, invariant to chunk size and budget.
        # Real recipes can read checkpoint slices into this same supplied buffer.
        if request.name.endswith("bias"):
            chunk.zero_()
            return
        seed = int.from_bytes(sha256(request.name.encode()).digest()[:4], "little")
        indices = torch.arange(request.offset, request.offset + chunk.numel())
        values = (indices * 19 + seed).remainder_(251)
        chunk.copy_(values).div_(125).sub_(1).mul_(request.shape[-1] ** -0.5)

    return {
        "model_factory": model_factory,
        "objective": objective,
        "optimizer": partial(torch.optim.AdamW, lr=torch.tensor(3e-4), foreach=False),
        "fill_tensor": fill_tensor,
        "build_options": {"parallelism": parallelism},
        "hyperparams": {"optimizer_group.0.lr": 3e-4},
        "candidates": {
            f"rows_{size}": {
                "args": [
                    (features[start : start + size].clone(),)
                    for start in range(0, rows, size)
                ],
                "kwargs": [
                    {"targets": targets[start : start + size].clone()}
                    for start in range(0, rows, size)
                ],
            }
            for size in (rows // 4, rows // 2)
        },
        "units_per_step": rows * ranks,
        "unit_label": "samples",
    }
