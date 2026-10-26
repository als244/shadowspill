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
from .._hyperparams import set_hyperparams
from .._inputs import on_device
from .._weights import Weights
from ._spec import StepSpec


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
        self._frozen_versions = {
            name: parameter._version
            for name, parameter in model.named_parameters(remove_duplicate=False)
            if not parameter.requires_grad
        }
        self._restored_frozen: set[str] = set()
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
        set_hyperparams(self.model, self.optimizer, hyperparams)
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
        self,
        path: Path,
        *,
        weights: Literal["master", "compute"] = "master",
        include_optimizer: bool = True,
        frozen_state_id: str | None = None,
    ) -> None:
        checkpoints.save(
            path,
            self.model,
            self.optimizer,
            self.steps,
            self.weights.masters,
            weights=weights,
            include_optimizer=include_optimizer,
            frozen_state_id=frozen_state_id,
            mutable_frozen=self._restored_frozen
            | {
                name
                for name, parameter in self.model.named_parameters(
                    remove_duplicate=False
                )
                if name in self._frozen_versions
                and parameter._version != self._frozen_versions[name]
            },
        )

    def load_state_dict(
        self, state: Mapping[str, Any], *, frozen_state_id: str | None = None
    ) -> None:
        if "optimizer" not in state and self.steps:
            raise ValueError(
                "a checkpoint without optimizer state requires a fresh Trainer; "
                "use prepare(checkpoint=...) to resume with a fresh optimizer"
            )
        self.steps = checkpoints.load(
            dict(state),
            self.model,
            self.optimizer,
            self.weights.masters,
            frozen_state_id=frozen_state_id,
        )
        if frozen_state_id is not None:
            self._restored_frozen.update(
                set(state["model"]) & set(self._frozen_versions)
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
