"""Apply declared storage dtypes once, before initialization and capture."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from shadowspill.parallelism.selectors import select, validate_selectors
from shadowspill_client.config.precision import parameter_dtypes

if TYPE_CHECKING:
    from torch import nn


def apply_parameter_dtypes(
    model: nn.Module,
    *,
    defaults: Mapping[str, str] | None = None,
    overrides: Mapping[str, str] | None = None,
) -> None:
    """Select floating parameter/buffer storage without changing aliases or flags.

    User rules win over defaults for each tensor. Within either layer, exact
    names win over globs; conflicting matches fail instead of depending on dict
    order. Defaults may describe optional layers; every user pattern must match
    a floating tensor. Integer metadata is never cast.

    This is a setup operation, not a permanent override of Module.to(). It does
    not select activation, accumulation or optimizer-state precision. Apply it
    before loading values; conversions cannot restore already rounded weights.
    """
    import torch

    defaults = parameter_dtypes({} if defaults is None else defaults)
    overrides = parameter_dtypes({} if overrides is None else overrides)
    tensors = {
        name: tensor
        for name, tensor in (
            *model.named_parameters(remove_duplicate=False),
            *model.named_buffers(remove_duplicate=False),
        )
        if tensor.is_floating_point()
    }
    validate_selectors(tuple(tensors), overrides)
    aliases: dict[int, list[str]] = {}
    for name, tensor in tensors.items():
        aliases.setdefault(id(tensor), []).append(name)

    # Validate every selector before mutation. Then convert one tensor at a
    # time from its original value, never through an intermediate BF16 copy.
    choices = []
    for names in aliases.values():
        dtype_name = select(names, overrides, None)
        if dtype_name is None:
            dtype_name = select(names, defaults, None)
        if dtype_name is None:
            continue
        tensor = tensors[names[0]]
        dtype = getattr(torch, dtype_name)
        if tensor.dtype == dtype:
            continue
        if type(tensor) not in (torch.Tensor, torch.nn.Parameter):
            raise TypeError(
                f"parameter_dtypes cannot convert tensor subclass at {names[0]!r}; "
                "configure its storage through the installed provider"
            )
        choices.append((tensor, dtype))
    with torch.no_grad():
        for tensor, dtype in choices:
            converted = tensor.detach().to(dtype=dtype)
            gradient = tensor.grad if tensor.is_leaf else None
            converted_gradient = (
                None if gradient is None else gradient.detach().to(dtype=dtype)
            )
            tensor.data = converted
            if gradient is not None:
                assert converted_gradient is not None
                gradient.data = converted_gradient
