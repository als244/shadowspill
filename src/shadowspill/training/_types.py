"""Internal structural types; callers supply ordinary functions and iterables."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

import torch
from torch import nn

from shadowspill.pytorch.distributed import Distributed

if TYPE_CHECKING:
    from .backends._spec import StepSpec

type Objective = Callable[[nn.Module, Any], Any]
type Microbatches = Callable[[Any], Iterable[tuple[Any, float | torch.Tensor]]]
type ParameterObserver = Callable[[torch.Tensor, torch.Tensor], Any]
type Initializer = Callable[[nn.Module], None]
type ParameterGroups = Callable[[nn.Module], Iterable[Mapping[str, Any]]]
type OptimizerConstructor = Callable[..., torch.optim.Optimizer]


class StepExecution(Protocol):
    model: nn.Module
    plan: Any
    planning: Any

    def run(
        self, microbatches: Sequence[Sequence[Any]], hyperparams: Mapping[str, Any]
    ) -> tuple[Any, Any, Any]: ...
    def synchronize(self) -> None: ...
    def save_plan(self, path: Path) -> None: ...
    def save(
        self, path: Path, *, weights: Literal["master", "compute"] = "master"
    ) -> None: ...
    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...
    def close(self) -> None: ...


class ForwardExecution(Protocol):
    model: nn.Module
    plan: Any

    def run(self, data: Any) -> Any: ...
    def synchronize(self) -> None: ...
    def close(self) -> None: ...


class Backend(Protocol):
    device: torch.device

    def prepare_step(
        self,
        model: nn.Module,
        spec: StepSpec,
        examples: Mapping[str, Sequence[Sequence[Any]]],
    ) -> tuple[StepExecution, str]: ...
    def prepare_forward(
        self,
        model: nn.Module,
        forward_fn: Callable[..., Any],
        example: Any,
        *,
        training: bool = False,
        distributed: Distributed | None = None,
    ) -> ForwardExecution: ...
