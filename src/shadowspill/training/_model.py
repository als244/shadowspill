"""Initialization and optimizer construction independent of model families."""

from __future__ import annotations

from collections.abc import Collection, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Self, cast

import torch
from torch import nn

from ._types import Initializer, OptimizerConstructor, ParameterGroups


def initialize_model(
    model: nn.Module,
    *,
    initialize: Initializer | None = None,
    state: Mapping[str, Any] | None = None,
    missing_parameters: Collection[str] = (),
) -> nn.Module:
    values = (*model.parameters(), *model.buffers())
    meta = [value.is_meta for value in values]
    if any(meta):
        if not all(meta):
            raise ValueError("model mixes meta and initialized state")
        if initialize is None and state is None:
            raise ValueError("a meta model requires initialize= or checkpoint=")
        if state is not None and initialize is None:
            missing_buffers = set(dict(model.named_buffers())) - set(model.state_dict())
            if missing_buffers:
                raise ValueError(
                    "checkpoint omits nonpersistent meta buffers: "
                    + ", ".join(sorted(missing_buffers))
                    + "; supply initialize= for values omitted by the checkpoint"
                )
        # Keep tied parameter identities. No reset runs unless explicitly supplied.
        converted: dict[int, torch.Tensor] = {}
        storages: dict[int, torch.Tensor] = {}
        for module in model.modules():
            for registry in (module._parameters, module._buffers):
                for name, value in registry.items():
                    if value is None:
                        continue
                    if id(value) not in converted:
                        source = value.untyped_storage()
                        if source._cdata not in storages:
                            storages[source._cdata] = torch.empty(
                                source.nbytes(), dtype=torch.uint8, device="cpu"
                            )
                        tensor = torch.empty(0, dtype=value.dtype, device="cpu").set_(
                            storages[source._cdata].untyped_storage(),
                            value.storage_offset(),
                            value.shape,
                            value.stride(),
                        )
                        tensor.requires_grad_(value.requires_grad)
                        converted[id(value)] = (
                            nn.Parameter(tensor, requires_grad=value.requires_grad)
                            if isinstance(value, nn.Parameter)
                            else tensor
                        )
                    # Parameter values retain their subclass above.
                    cast(dict[str, torch.Tensor | None], registry)[name] = converted[
                        id(value)
                    ]
    if initialize is not None:
        initialize(model)
    if state is not None:
        if missing_parameters:
            omitted = set(missing_parameters)
            if not omitted <= set(dict(model.named_parameters(remove_duplicate=False))):
                raise ValueError("omitted checkpoint state must name model parameters")
            keys = model.load_state_dict(state, strict=False)
            if set(keys.missing_keys) != omitted or keys.unexpected_keys:
                raise ValueError(
                    "checkpoint entries differ from declared master-backed weights"
                )
        else:
            model.load_state_dict(state)
    return model


def reset_parameters(model: nn.Module) -> None:
    """Explicit initializer for modules implementing reset_parameters."""
    for module in model.modules():
        reset = getattr(module, "reset_parameters", None)
        if callable(reset):
            reset()


@contextmanager
def model_mode(model: nn.Module, training: bool) -> Iterator[None]:
    modes = [(module, module.training) for module in model.modules()]
    model.train(training)
    try:
        yield
    finally:
        for module, previous in modes:
            module.training = previous


@dataclass(frozen=True)
class OptimizerFactory:
    constructor: OptimizerConstructor
    arguments: Mapping[str, Any]
    parameter_names: tuple[str, ...]
    groups: tuple[Mapping[str, Any], ...] | None

    @classmethod
    def bind(
        cls,
        model: nn.Module,
        constructor: OptimizerConstructor,
        arguments: Mapping[str, Any],
        parameter_groups: ParameterGroups | None = None,
    ) -> Self:
        if isinstance(constructor, torch.optim.Optimizer):
            raise TypeError(
                "pass an optimizer constructor; live optimizers use plan_step"
            )
        named = dict(model.named_parameters())
        names = {id(value): name for name, value in named.items()}
        groups = None
        if parameter_groups is not None:
            resolved = []
            for original in parameter_groups(model):
                group = dict(original)
                try:
                    group["params"] = tuple(names[id(p)] for p in group["params"])
                except KeyError as error:
                    raise ValueError(
                        "parameter_groups must select registered model parameters"
                    ) from error
                resolved.append(group)
            groups = tuple(resolved)
        return cls(constructor, dict(arguments), tuple(named), groups)

    def __call__(self, parameters: Iterable[torch.Tensor]) -> torch.optim.Optimizer:
        named = dict(zip(self.parameter_names, parameters, strict=True))
        values: list[Any]
        if self.groups is None:
            values = list(named.values())
        else:
            values = [
                {**group, "params": [named[name] for name in group["params"]]}
                for group in self.groups
            ]
        optimizer = self.constructor(values, **self.arguments)
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("optimizer constructor must return torch.optim.Optimizer")
        return optimizer
