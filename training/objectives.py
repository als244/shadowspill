"""What a training step minimizes, and the module that computes it.

An objective is a function of the model and one microbatch, as the data source
makes it -- ``objective(model, tokens, targets, seq_lens, **options)`` -- that
returns the loss the step differentiates. By this package's convention the
loss sums over the positions the targets train and divides by all the
microbatch's positions, padding included; then a step's gradient weighs every
trained position alike across microbatches, and the trainer reports the loss
per trained position.

``Objective`` wraps the model with its objective and options, so that one
module both trains and evaluates: ShadowSpill plans evaluation
(``plan_forward``) over the same imported state as training (``plan_step``),
and imported state belongs to the module it was imported with.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch
import torch.nn as nn


def model_loss(
    model: nn.Module,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    seq_lens: torch.Tensor,
    **options: Any,
) -> torch.Tensor:
    """The model's own ``loss(tokens, targets, seq_lens=..., **options)``, for a
    model that computes its training loss itself."""

    return model.loss(tokens, targets, seq_lens=seq_lens, **options)


class Objective(nn.Module):
    """A model whose forward pass is a training objective."""

    def __init__(
        self,
        model: nn.Module,
        objective: Callable[..., torch.Tensor],
        options: Mapping[str, Any],
    ) -> None:
        super().__init__()
        self.model = model
        self.objective = objective
        self.options = dict(options)

    def forward(
        self, tokens: torch.Tensor, targets: torch.Tensor, seq_lens: torch.Tensor
    ) -> torch.Tensor:
        return self.objective(self.model, tokens, targets, seq_lens, **self.options)


def planned_objective(
    module: Objective,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    seq_lens: torch.Tensor,
) -> torch.Tensor:
    """What ShadowSpill plans: the module's forward pass."""

    return module(tokens, targets, seq_lens)
