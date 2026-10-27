"""Building a model as structure only, and giving it its first values.

A run can build its model on meta, declaring shapes and dtypes without values.
Ordinary reference engines materialize it and call ``initialize`` below.
ShadowSpill builds from its metadata; the execution worker later fills its
declared tensor shards through the bounded Executor initialization interface.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn


def build_on_meta(
    model: Callable[..., nn.Module],
    dtype: str | None = None,
    lora: dict[str, Any] | None = None,
    device: str = "meta",
    **arguments: Any,
) -> nn.Module:
    """``model(**arguments)`` built on ``device`` (``meta`` unless another is
    given, so a run can declare shapes without values), declaring its
    parameters in ``dtype`` -- a ``torch`` dtype name such as
    ``"bfloat16"`` -- or in the current default dtype."""

    # The device is a context, pushed and popped as it was: asking PyTorch
    # for the default device allocates a probe tensor, which a construction
    # context such as DeepSpeed's zero.Init places on the accelerator, and
    # restoring that would leave every later default allocation there.
    previous_dtype = torch.get_default_dtype()
    if dtype is not None:
        torch.set_default_dtype(getattr(torch, dtype))
    try:
        with torch.device(device):
            result = model(**arguments)
            if lora is not None:
                from workloads.lora import configure_lora  # a caller-side recipe

                result = configure_lora(result, **lora)
        return result
    finally:
        torch.set_default_dtype(previous_dtype)


def initialize(module: nn.Module) -> None:
    """Reset an ordinary materialized model in module order."""

    for submodule in module.modules():
        reset = getattr(submodule, "reset_parameters", None)
        if callable(reset):
            reset()
