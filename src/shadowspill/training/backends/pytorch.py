"""Ordinary eager or compiled PyTorch execution of generic objectives."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, Self

import torch
from torch import nn
from torch.utils._pytree import tree_map

from shadowspill.pytorch.accelerator import resolve_device
from shadowspill.pytorch.distributed import Distributed

from .. import checkpoints
from .._inputs import on_device
from ._spec import StepSpec
from ._weights import Weights


class PyTorch:
    """Compile the objective and its autograd graph, or execute them eagerly."""

    def __init__(
        self, *, compile: bool = True, device: str | int | torch.device | None = "auto"
    ) -> None:
        self.compile = compile
        self.device = resolve_device(device, allow_cpu=True)
        self._sessions: list[_Step | _Forward] = []

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exception: object) -> None:
        self.close()

    def prepare_step(
        self,
        model: nn.Module,
        spec: StepSpec,
        examples: Mapping[str, Sequence[Sequence[Any]]],
    ) -> tuple[_Step, str]:
        if spec.distributed is not None:
            raise NotImplementedError(
                "Distributed ownership currently requires the ShadowSpill backend"
            )
        if len(examples) != 1:
            raise ValueError(
                "PyTorch needs one microbatch candidate; select a function"
            )
        model.to(self.device)
        selected = next(iter(examples))
        session = _Step(self, model, spec)
        self._sessions.append(session)
        return session, selected

    def prepare_forward(
        self,
        model: nn.Module,
        forward_fn: Callable[..., Any],
        example: Any,
        *,
        training: bool = False,
        distributed: Distributed | None = None,
    ) -> _Forward:
        if distributed is not None:
            raise NotImplementedError(
                "Distributed ownership currently requires the ShadowSpill backend"
            )
        model.to(self.device)
        session = _Forward(self, model, forward_fn)
        self._sessions.append(session)
        return session

    def synchronize(self) -> None:
        if self.device.type != "cpu":
            torch.accelerator.synchronize(self.device)

    def close(self) -> None:
        for session in reversed(self._sessions):
            session.close()
        self._sessions.clear()


class _Step:
    plan = None
    planning = None

    def __init__(self, backend: PyTorch, model: nn.Module, spec: StepSpec) -> None:
        self.backend, self.model = backend, model
        model.zero_grad(set_to_none=True)
        self.weights = Weights(model, spec.master_dtype, spec.grad_dtype)
        self.optimizer = spec.optimizer(model)(self.weights.parameters())
        self.objective = spec.objective
        self.observer = spec.parameter_metrics
        if backend.compile:
            self.objective = torch.compile(
                self.objective, fullgraph=True, dynamic=False
            )
            if self.observer is not None:
                self.observer = torch.compile(
                    self.observer, fullgraph=True, dynamic=True
                )
        self.steps = 0
        self.closed = False

    def run(
        self,
        microbatches: Sequence[Sequence[Any]],
        hyperparams: Mapping[str, Any],
    ) -> tuple[Any, Any, Any]:
        if self.closed:
            raise RuntimeError("training execution is closed")
        _set_hyperparams(self.model, self.optimizer, hyperparams)
        losses, metrics = [], []
        for data, scale in microbatches:
            loss, values = self.objective(
                self.model, *on_device((data, scale), self.backend.device)
            )
            loss.backward()
            self.weights.accumulate()
            losses.append(loss.detach())
            metrics.append(tree_map(_detach, values))
        observations = self.weights.observe(self.observer)
        self.weights.step(self.optimizer)
        self.steps += 1
        return losses, metrics, observations

    def synchronize(self) -> None:
        self.backend.synchronize()

    def save_plan(self, path: Path) -> None:
        """Ordinary PyTorch execution has no admitted memory plan to export."""
        del path

    def save(
        self, path: Path, *, weights: Literal["master", "compute"] = "master"
    ) -> None:
        checkpoints.save(
            path,
            self.model,
            self.optimizer,
            self.steps,
            self.weights.masters,
            weights=weights,
        )

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.steps = checkpoints.load(
            dict(state), self.model, self.optimizer, self.weights.masters
        )

    def close(self) -> None:
        self.closed = True


class _Forward:
    plan = None

    def __init__(
        self, backend: PyTorch, model: nn.Module, forward_fn: Callable[..., Any]
    ) -> None:
        self.backend, self.model = backend, model
        self.call = (
            torch.compile(forward_fn, fullgraph=True, dynamic=False)
            if backend.compile
            else forward_fn
        )
        self.closed = False

    def run(self, data: Any) -> Any:
        if self.closed:
            raise RuntimeError("forward execution is closed")
        with torch.no_grad():
            return self.call(self.model, on_device(data, self.backend.device))

    def synchronize(self) -> None:
        self.backend.synchronize()

    def close(self) -> None:
        self.closed = True


def _detach(value: Any) -> Any:
    return value.detach() if isinstance(value, torch.Tensor) else value


def _set_hyperparams(
    model: nn.Module, optimizer: torch.optim.Optimizer, values: Mapping[str, Any]
) -> None:
    buffers = dict(model.named_buffers())
    for name, value in values.items():
        groups = [group for group in optimizer.param_groups if name in group]
        buffer = buffers.get(name)
        if groups and buffer is not None:
            raise ValueError(
                f"{name!r} names both an optimizer value and a model buffer"
            )
        if buffer is not None:
            with torch.no_grad():
                buffer.fill_(value)
        elif groups:
            for group in groups:
                held = group[name]
                if isinstance(held, torch.Tensor):
                    held.fill_(value)
                elif isinstance(held, (tuple, list)):
                    items = (
                        [value] * len(held)
                        if isinstance(value, (int, float))
                        else list(value)
                    )
                    if len(items) != len(held):
                        raise ValueError(f"{name!r} needs {len(held)} values per group")
                    group[name] = type(held)(items)
                else:
                    group[name] = value
        else:
            raise KeyError(f"no optimizer value or model buffer named {name!r}")
