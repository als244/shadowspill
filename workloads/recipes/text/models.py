"""Building a model as structure only, and giving it its first values.

A run can build its model on meta, declaring shapes and dtypes without values.
The generic trainer's fresh initializer currently materializes ordinary CPU
storage before the backend imports it. ShadowSpill then copies state into its
pinned pool; direct pool or streaming initialization is not implied by this helper.
Callers with tight host-memory limits can initialize once and supply file-backed
values instead.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

import torch
import torch.nn as nn


def build_on_meta(
    model: Callable[..., nn.Module], dtype: str | None = None, **arguments: Any
) -> nn.Module:
    """``model(**arguments)`` built on ``meta``, declaring its parameters in
    ``dtype`` -- a ``torch`` dtype name such as ``"bfloat16"`` -- or in the
    current default dtype."""

    previous_dtype = torch.get_default_dtype()
    previous_device = torch.get_default_device()
    if dtype is not None:
        torch.set_default_dtype(getattr(torch, dtype))
    torch.set_default_device("meta")
    try:
        return model(**arguments)
    finally:
        torch.set_default_device(previous_device)
        torch.set_default_dtype(previous_dtype)


def config_preset(preset: Callable[[], Any], **changes: Any) -> Any:
    """The config ``preset()`` returns -- a dataclass -- with ``changes`` in place
    of its own values: a model's preset with another ``max_seq_len``, say."""

    return replace(preset(), **changes)


def initialize(module: nn.Module) -> None:
    """Let every module fill the storage it owns, in module order -- the order
    ShadowSpill's import initializes in, so both backends draw the same values."""

    for submodule in module.modules():
        reset = getattr(submodule, "reset_parameters", None)
        if callable(reset):
            reset()
