"""What a training step minimizes, and the module that computes it.

An objective is a function of the model and one microbatch, as the data source
makes it -- ``objective(model, tokens, targets, seq_lens, **options)`` -- that
returns the model's loss summed over the positions the targets train. Padding
and a target of ``-100`` count nowhere.

``Objective`` wraps the model with its objective and options and divides that
sum by ``trained_total``, a buffer holding the trained positions of the whole
step (or evaluation set) the microbatch belongs to, which the trainer sets
before each step as a hyperparameter, the way it sets the learning rate. So a
microbatch's value is its share of the step's mean loss over trained tokens:
the step's loss is the sum of its microbatches' shares, the mean over the
step's trained tokens whatever the microbatch geometry, and its gradient
weighs every trained token by that one total. One module both trains and
evaluates: ShadowSpill plans evaluation (``plan_forward``) over the same
imported state as training (``plan_step``), and imported state belongs to the
module it was imported with.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from itertools import chain
from typing import Any

import torch
import torch.nn as nn
from torch.utils._pytree import tree_map

from shadowspill.pytorch.contracts import ObjectiveResult


def model_loss(
    model: nn.Module,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    seq_lens: torch.Tensor,
    **options: Any,
) -> torch.Tensor:
    """The model's own ``loss(tokens, targets, seq_lens=..., reduction="sum",
    **options)``: its loss summed over the positions the targets train, for a
    model that computes its training loss itself."""

    return model.loss(tokens, targets, seq_lens=seq_lens, reduction="sum", **options)


def model_loss_with_metrics(
    model, tokens, targets, seq_lens, **options
) -> ObjectiveResult:
    """A model's summed loss and compact metric pytree, returned separately."""

    loss, metrics = model.loss(
        tokens,
        targets,
        seq_lens=seq_lens,
        reduction="sum",
        return_metrics=True,
        **options,
    )
    return ObjectiveResult(loss, metrics)


def unpack_objective(value):
    """Extract the loss and observations from the wrapper's exportable result."""

    if isinstance(value, dict):
        return value["loss"], value["metrics"]
    return value, None


class Objective(nn.Module):
    """A model whose forward pass is a microbatch's share of the training
    objective: the objective's sum over trained positions divided by the
    ``trained_total`` buffer, the trained positions of the whole step."""

    TRAINED_TOTAL = "trained_total"  # the buffer's name, as a hyperparameter
    trained_total: torch.Tensor

    def __init__(
        self,
        model: nn.Module,
        objective: Callable[..., torch.Tensor | ObjectiveResult],
        options: Mapping[str, Any],
    ) -> None:
        super().__init__()
        self.model = model
        self.objective = objective
        self.options = dict(options)
        state = next(chain(model.parameters(), model.buffers()), None)
        device = None if state is None else state.device
        self.register_buffer(
            self.TRAINED_TOTAL, torch.ones((), device=device, dtype=torch.float32)
        )

    def reset_parameters(self) -> None:
        """The total's value before a step sets it: one, so the objective is the
        plain sum until then (initialization visits every module's own storage,
        and this buffer is this module's)."""

        with torch.no_grad():
            self.trained_total.fill_(1.0)

    def forward(
        self, tokens: torch.Tensor, targets: torch.Tensor, seq_lens: torch.Tensor
    ) -> torch.Tensor | dict[str, Any]:
        summed = self.objective(self.model, tokens, targets, seq_lens, **self.options)
        if isinstance(summed, ObjectiveResult):
            metrics = tree_map(
                lambda x: x.detach() if isinstance(x, torch.Tensor) else x,
                summed.metrics,
            )
            return {"loss": summed.loss / self.trained_total, "metrics": metrics}
        return summed / self.trained_total


def planned_objective(
    module: Objective,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    seq_lens: torch.Tensor,
) -> torch.Tensor | ObjectiveResult:
    """What ShadowSpill plans: the module's forward pass."""

    loss, metrics = unpack_objective(module(tokens, targets, seq_lens))
    return loss if metrics is None else ObjectiveResult(loss, metrics)
