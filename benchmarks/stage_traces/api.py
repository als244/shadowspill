"""Small model adapter contract; tracing and sweep code are model-independent."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any

import torch


def resolve(name):
    module, attribute = name.split(":", 1)
    return getattr(import_module(module), attribute)


@dataclass
class TraceWorkload:
    """A meta model, scalar objective, real small inputs and state fill recipes.

    State names use model.named_parameters/buffers names, expanded to physical
    tensor components for wrappers. Each recipe names a callable taking the
    destination tensor plus keyword arguments. Values are generated only for
    the isolated stage. Unknown state or integral inputs are errors.
    """

    model: torch.nn.Module
    objective: Callable[..., torch.Tensor]
    inputs: tuple[Any, ...]
    state_initializers: dict[str, dict]
    metadata: dict = field(default_factory=dict)
    context: Callable[[], AbstractContextManager] = nullcontext
