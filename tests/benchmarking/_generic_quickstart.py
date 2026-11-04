"""Importable non-text experiment; no workload or model-library dependency."""

import torch
from torch import nn

from shadowspill.pytorch import (
    AxisChunks,
    GraphPairPolicy,
    RetentionPolicy,
    RetentionRule,
)
from shadowspill.task.inputs import TaskInputRole


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


class IndependentExamples:
    def __init__(self, chunk_size):
        self.chunk_size = chunk_size

    def configure_stage(self, stage):
        if stage.terminal:
            return None
        axes = {
            i: 0
            for i, (value, role) in enumerate(
                zip(stage.inputs, stage.input_roles, strict=True)
            )
            if value is not None
            and len(value.shape) == 2
            and role not in {TaskInputRole.PARAMETER, TaskInputRole.BUFFER}
        }
        return AxisChunks(self.chunk_size, axes) if axes else None


def chunked_experiment(*, device):
    setup = experiment(device=device)
    policies = {
        "inputs_only": GraphPairPolicy(),
        "selected_retained": GraphPairPolicy(
            recompute=RetentionPolicy(
                default="recompute",
                rules=(RetentionRule("aten.tanh.default", "retain"),),
            )
        ),
        "selected_only_save": GraphPairPolicy(
            save=RetentionPolicy(
                default="recompute",
                rules=(RetentionRule("aten.tanh.default", "retain"),),
            ),
            recompute=RetentionPolicy(default="recompute"),
        ),
    }
    setup["candidates"] = {
        f"{name}-{retention}-chunk{size}": {
            **inputs,
            "build_options": {
                "chunking": IndependentExamples(size),
                "graph_pair_policy": policy,
            },
        }
        for name, inputs in setup["candidates"].items()
        for size in (1, 3)
        for retention, policy in policies.items()
    }
    return setup
