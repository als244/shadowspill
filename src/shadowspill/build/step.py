"""Build a complete training update; scheduling remains in planning."""

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
from torch import nn

from shadowspill.artifacts.program import Program
from shadowspill.pytorch.capture.retention import RetentionPolicy
from shadowspill.pytorch.partition import PartitionSpec

from .training import _build_training


def build_step(
    model: nn.Module,
    *,
    objective: Callable[..., Any],
    optimizer: torch.optim.Optimizer,
    example_args: Sequence[Any] = (),
    example_kwargs: Mapping[str, Any] | None = None,
    num_examples: int = 1,
    partition: PartitionSpec = "auto",
    operator_recompute: RetentionPolicy | None = None,
    grad_dtype: torch.dtype | None = None,
    target: str | torch.device = "cuda",
) -> Program:
    """Capture gradients and local updates without initializing persistent state.

    The supplied optimizer must refer to the model's parameters. Neither is
    changed. Saved defaults describe lazily created optimizer state; actual
    tensor contents are supplied at execution. CPU planning chooses when each
    independent update runs after its required gradient writes and weight reads.
    """
    return _build_training(
        model,
        objective=objective,
        optimizer=optimizer,
        example_args=example_args,
        example_kwargs=example_kwargs,
        num_examples=num_examples,
        partition=partition,
        operator_recompute=operator_recompute,
        grad_dtype=grad_dtype,
        target=target,
    )
