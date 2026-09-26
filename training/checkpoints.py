"""The checkpoint format both backends write.

A checkpoint is ``torch.save`` of ``{"model", "optimizer", "step"}``: the
model's state dict, the optimizer's, and the steps taken. A weight trained over
a master copy at another precision is written as its master, under the weight's
name, so the checkpoint holds the training state at full precision once and the
weight follows from it -- and it loads into a plain model of either precision
as it stands. ShadowSpill's ``PlannedTrainStep.save`` writes this format
straight from its pool; the PyTorch backend writes it with ``save`` here.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def save(
    path: Path,
    module: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    masters: Mapping[str, torch.Tensor] | None = None,
) -> None:
    """Write the module's and the optimizer's state, each master in place of
    its weight."""

    model = module.state_dict()
    for name, master in (masters or {}).items():
        for alias in _names(module)[name]:
            model[alias] = master.detach()
    torch.save(
        {"model": model, "optimizer": optimizer.state_dict(), "step": step}, path
    )


def load(
    state: dict[str, Any],
    module: nn.Module,
    optimizer: torch.optim.Optimizer,
    masters: Mapping[str, torch.Tensor] | None = None,
) -> int:
    """Restore a checkpoint into the module, its masters and the optimizer;
    return its step. A master takes the value written for its weight, and the
    weight that value's cast."""

    model = dict(state["model"])
    dtypes = {name: value.dtype for name, value in module.state_dict().items()}
    with torch.no_grad():
        for name, master in (masters or {}).items():
            master.copy_(model[name])
            for alias in _names(module)[name]:
                model[alias] = model[alias].to(dtypes[alias])
    module.load_state_dict(model)
    optimizer.load_state_dict(state["optimizer"])
    return int(state["step"])


def _names(module: nn.Module) -> dict[str, tuple[str, ...]]:
    """Every name a weight goes by, under the first of them."""

    names: dict[int, list[str]] = {}
    for name, parameter in module.named_parameters(remove_duplicate=False):
        names.setdefault(id(parameter), []).append(name)
    return {every[0]: tuple(every) for every in names.values()}
