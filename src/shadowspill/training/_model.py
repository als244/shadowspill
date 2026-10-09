"""Initialization and optimizer construction independent of model families."""

from __future__ import annotations

from collections.abc import Collection, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from itertools import groupby
from typing import Any, Self

import torch
from torch import nn

from shadowspill.pytorch.representations import materialize_meta_state
from shadowspill.pytorch.state.serialization import decode_tensor_state

from ._types import Initializer, OptimizerConstructor, ParameterGroups


def validate_initialization(
    model: nn.Module,
    *,
    initialize: Initializer | None,
    state: Mapping[str, Any] | None,
) -> bool:
    """Check initialization inputs before allocating any parameter payload."""
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
    return any(meta)


def initialize_model(
    model: nn.Module,
    *,
    initialize: Initializer | None = None,
    state: Mapping[str, Any] | None = None,
    missing_parameters: Collection[str] = (),
) -> nn.Module:
    if validate_initialization(model, initialize=initialize, state=state):
        # Keep tied parameter identities. No reset runs unless explicitly supplied.
        materialize_meta_state(model)
    if initialize is not None:
        initialize(model)
    if state is not None:
        state = decode_tensor_state(state, model.state_dict())
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
        optimizer = build_optimizer(self.constructor, values, self.arguments)
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("optimizer constructor must return torch.optim.Optimizer")
        return optimizer


def build_optimizer(
    constructor: OptimizerConstructor,
    parameters: Iterable[Any],
    arguments: Mapping[str, Any],
) -> torch.optim.Optimizer:
    """Apply training defaults when constructing MLOps AdamW.

    Explicit constructor and group rounding choices win. Other optimizers and
    caller-owned optimizer instances retain their own semantics. Mixed moment
    dtypes under ``opt_state_dtype="parameter"`` get consecutive dtype groups
    without changing parameter order or stochastic rounding salts.
    """
    target = constructor
    supplied = dict(arguments)
    while isinstance(target, partial):
        supplied = {**target.keywords, **supplied}
        target = target.func
    if (
        getattr(target, "implementation_id", None) != "builtin.adamw.triton"
        or "opt_state_rounding" in supplied
    ):
        return constructor(parameters, **arguments)

    values = list(parameters)
    if not values:
        return constructor(values, **arguments)
    groups = values if values and isinstance(values[0], dict) else [{"params": values}]
    configured = []
    for original in groups:
        group = dict(original)
        group["params"] = list(group["params"])
        dtype = group.get(
            "opt_state_dtype", supplied.get("opt_state_dtype", torch.bfloat16)
        )
        if "opt_state_rounding" not in group:
            if dtype == torch.bfloat16:
                group["opt_state_rounding"] = "stochastic"
            elif dtype == "parameter" and group["params"]:
                for bf16, entries in groupby(
                    enumerate(group["params"]),
                    key=lambda entry: entry[1].dtype == torch.bfloat16,
                ):
                    indices, members = zip(*entries, strict=True)
                    part = {
                        **group,
                        "params": list(members),
                        "opt_state_rounding": "stochastic" if bf16 else "nearest",
                    }
                    if "param_names" in group:
                        part["param_names"] = [group["param_names"][i] for i in indices]
                    configured.append(part)
                continue
        configured.append(group)
    return constructor(configured, **arguments)
