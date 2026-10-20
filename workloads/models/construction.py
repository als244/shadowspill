"""Building a model as structure only, and giving it its first values.

A run can build its model on meta, declaring shapes and dtypes without values.
ShadowSpill imports that structure into its pool before running the initializer.
When a value is not directly addressable, including sharded CPU state, setup
stages only the storage roots used by the current operation and writes mutations
back to their owned pool ranges. Initializers should visit individual tensors
instead of passing the whole model to a bulk operation.
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
    """Let every module fill the storage it owns, in module order -- the order
    ShadowSpill's import initializes in, so both backends draw the same values."""

    for submodule in module.modules():
        reset = getattr(submodule, "reset_parameters", None)
        if callable(reset):
            reset()
