"""The checkpoint format both backends write.

A checkpoint is ``torch.save`` of ``{"model", "optimizer", "step",
"model_from_optimizer"}``: the model's state dict, the optimizer's, the steps
taken, and which model entries were left out because an optimizer entry
reproduces them bit for bit by a cast -- a weight kept beside the
higher-precision master it is the rounding of is written once, as the master.
Which entries qualify is found from the values, not from names or dtypes.
ShadowSpill's ``PlannedTrainStep.save`` writes this format straight from its
pool; the PyTorch backend writes it with ``save`` here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def save(
    path: Path, module: nn.Module, optimizer: torch.optim.Optimizer, step: int
) -> None:
    """Write the module's and the optimizer's state, each weight at most once."""

    model = module.state_dict()
    state = optimizer.state_dict()
    names = dict(enumerate(name for name, _ in module.named_parameters()))
    derived = {}
    for index, entries in state["state"].items():
        weight = model[names[index]]
        for key, value in entries.items():
            if _reproduces(value, weight):
                derived[names[index]] = (index, key)
                break
    kept = {name: value for name, value in model.items() if name not in derived}
    torch.save(
        {
            "model": kept,
            "optimizer": state,
            "step": step,
            "model_from_optimizer": derived,
        },
        path,
    )


def load(
    state: dict[str, Any], module: nn.Module, optimizer: torch.optim.Optimizer
) -> int:
    """Restore a checkpoint into the module and optimizer; return its step."""

    model = dict(state["model"])
    dtypes = {name: value.dtype for name, value in module.state_dict().items()}
    for name, (index, key) in state.get("model_from_optimizer", {}).items():
        model[name] = state["optimizer"]["state"][index][key].to(dtypes[name])
    module.load_state_dict(model)
    optimizer.load_state_dict(state["optimizer"])
    return int(state["step"])


def _reproduces(value: object, weight: torch.Tensor) -> bool:
    return (
        torch.is_tensor(value)
        and value.dtype != weight.dtype
        and value.shape == weight.shape
        and torch.equal(_bits(value.to(weight.dtype)), _bits(weight))
    )


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.contiguous().reshape(-1).view(torch.uint8)
