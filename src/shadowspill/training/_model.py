"""Initialize models and restore their training/evaluation modes."""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from torch import nn

from shadowspill.checkpoint.representations import decode_tensor_state
from shadowspill.pytorch.representations import materialize_meta_state

type Initializer = Callable[[nn.Module], None]


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
