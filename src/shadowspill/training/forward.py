"""A generic callable forward runner over the selected execution backend."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any, Self

from torch import nn

from shadowspill.pytorch.distributed import Distributed

from ._model import initialize_model, model_mode, validate_initialization
from ._types import Backend, ForwardExecution, Initializer


def model_forward(model: nn.Module, data: Any) -> Any:
    return model(data)


class Forward:
    """Prepare once, call or stream any supported input/output pytree."""

    def __init__(
        self,
        model: nn.Module,
        *,
        forward_fn: Callable[..., Any] | None = None,
        backend: Backend | None = None,
        training: bool = False,
        distributed: Distributed | None = None,
    ) -> None:
        if backend is None:
            from .backends import PyTorch

            backend = PyTorch()
        self.model = model
        self.forward_fn = forward_fn or model_forward
        self.backend = backend
        self.training = training
        self.distributed = distributed
        self._execution: ForwardExecution | None = None
        self._closed = False

    @property
    def plan(self) -> Any:
        return None if self._execution is None else self._execution.plan

    def prepare(
        self,
        example_data: Any,
        *,
        initialize: Initializer | None = None,
        checkpoint: str | Path | None = None,
    ) -> Self:
        if self._closed or self._execution is not None:
            raise RuntimeError("Forward must be open and unprepared")
        state = None
        if checkpoint is not None:
            import torch

            path = Path(checkpoint)
            path = path / "state.pt" if path.is_dir() else path
            state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            state = state.get("model", state)

        validate_initialization(self.model, initialize=initialize, state=state)

        def fill(model: nn.Module) -> None:
            initialize_model(model, initialize=initialize, state=state)

        prepare_model = getattr(self.backend, "initialize_model", None)
        if prepare_model is None:
            fill(self.model)
        else:
            self.model = prepare_model(self.model, fill)
        modes = {name: m.training for name, m in self.model.named_modules()}
        with model_mode(self.model, self.training):
            self._execution = self.backend.prepare_forward(
                self.model,
                self.forward_fn,
                example_data,
                training=self.training,
                distributed=self.distributed,
            )
        self.model = self._execution.model
        for name, module in self.model.named_modules():
            module.training = modes[name]
        return self

    def __call__(self, data: Any) -> Any:
        if self._execution is None or self._closed:
            raise RuntimeError("call prepare before using an open Forward")
        with model_mode(self.model, self.training):
            return self._execution.run(data)

    def synchronize(self) -> None:
        if self._execution is None or self._closed:
            raise RuntimeError("call prepare before using an open Forward")
        self._execution.synchronize()

    def map(self, source: Iterable[Any]) -> Iterator[Any]:
        for data in source:
            yield self(data)

    def close(self) -> None:
        if self._execution is not None and not self._closed:
            self._execution.close()
        self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exception: object) -> None:
        self.close()
