"""Importable non-text experiment; no workload or model-library dependency."""

import torch
from torch import nn


def experiment(*, device, fail_on_step=0):
    generator = torch.Generator().manual_seed(12)
    features = torch.randn(8, 4, generator=generator)
    targets = torch.randn(8, 2, generator=generator)

    def model_factory():
        with torch.device("meta"):
            return nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 2))

    def objective(model, features, targets):
        loss = (model(features) - targets).square().sum() / 16
        return loss, {"residual": loss.detach()}

    def fill_tensor(request, destination):
        destination.fill_(0 if request.name.endswith("bias") else 0.1)

    calls = 0

    def reduce(observed):
        nonlocal calls
        calls += 1
        if calls == fail_on_step:
            raise ValueError(f"intentional reducer failure at step {calls}")
        residual = sum(value["residual"] for value in observed.metrics)
        # CPU scalar tensors are valid reducer values. Displayed loss may
        # deliberately differ from the differentiated objective.
        return {"loss": residual / 2, "residual": residual}

    return {
        "model_factory": model_factory,
        "objective": objective,
        "optimizer": lambda params: torch.optim.SGD(params, lr=torch.tensor(0.02)),
        "fill_tensor": fill_tensor,
        "hyperparams": {"optimizer_group.0.lr": 0.02},
        "metric_reducer": reduce,
        "candidates": {
            name: {
                "args": [
                    (features[i : i + size].clone(), targets[i : i + size].clone())
                    for i in range(0, 8, size)
                ]
            }
            for name, size in [("two_images", 2), ("four_images", 4)]
        },
        "units_per_step": 8,
        "unit_label": "images",
    }
