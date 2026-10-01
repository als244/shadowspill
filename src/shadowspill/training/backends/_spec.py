"""Private execution contracts: compiled work, never source/loop policy."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from shadowspill.pytorch.distributed import Distributed

from .._types import ParameterObserver


@dataclass(frozen=True)
class StepSpec:
    objective: Any
    optimizer: Callable[
        [nn.Module], Callable[[Iterable[torch.Tensor]], torch.optim.Optimizer]
    ]
    hyperparams: tuple[str, ...] = ()
    master_dtype: torch.dtype | None = None
    grad_dtype: torch.dtype | None = None
    parameter_metrics: ParameterObserver | None = None
    distributed: Distributed | None = None
    shard_optimizer: bool = True
