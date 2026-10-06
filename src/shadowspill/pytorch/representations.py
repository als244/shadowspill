"""Logical tensor wrappers and their ordinary, storage-owning components.

Use PyTorch's tensor flatten/unflatten protocol. Physical leaves are the only
values that may be registered as byte storage; differentiation still operates
on the logical tensor. This module does not interpret a representation's math.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterator
from typing import Any, cast

import torch
from torch import nn
from torch.utils._python_dispatch import is_traceable_wrapper_subclass

is_wrapper = is_traceable_wrapper_subclass
type ComponentKey = tuple[int, tuple[str, ...]]
type RootInputKey = int | ComponentKey


def tensor_components(
    tensor: torch.Tensor, path: tuple[str, ...] = ()
) -> Iterator[tuple[tuple[str, ...], torch.Tensor]]:
    """Visit physical leaves in the order AOTAutograd flattens them."""
    if not is_wrapper(tensor):
        yield path, tensor
        return
    names, _metadata = tensor.__tensor_flatten__()
    for name in names:
        component = getattr(tensor, name)
        if not isinstance(component, torch.Tensor):
            raise TypeError(
                f"tensor component {'.'.join((*path, name))!r} is not a tensor"
            )
        yield from tensor_components(component, (*path, name))


def component_at(tensor: torch.Tensor, path: tuple[str, ...]) -> torch.Tensor:
    for name in path:
        tensor = getattr(tensor, name)
    return tensor


def detached_representation(tensor: torch.Tensor) -> torch.Tensor:
    """Keep independent physical views when a wrapper's detach reuses its leaves.

    The views still share storage and do not copy data, but rebinding the live
    model's tensor objects cannot move these references onto another device.
    """
    return map_tensor(tensor, lambda value: value.detach()).detach()


def map_tensor(
    tensor: torch.Tensor,
    leaf: Callable[[torch.Tensor], torch.Tensor],
    *,
    memo: dict[int, Any] | None = None,
) -> torch.Tensor:
    """Rebuild wrappers around mapped leaves, preserving ties and metadata."""
    if memo is None:
        memo = {}
    if id(tensor) in memo:
        return cast(torch.Tensor, memo[id(tensor)])
    if not is_wrapper(tensor):
        result = leaf(tensor)
    else:
        names, metadata = tensor.__tensor_flatten__()
        components = {
            name: map_tensor(getattr(tensor, name), leaf, memo=memo) for name in names
        }
        result = type(tensor).__tensor_unflatten__(
            components,
            copy.deepcopy(metadata),
            tuple(tensor.shape),
            tuple(tensor.stride()),
        )
        result.requires_grad_(tensor.requires_grad)
        if isinstance(tensor, nn.Parameter):
            # Parameter(wrapper) dispatches detach and may discard Python metadata.
            # PyTorch uses this marker for wrapper parameters as well.
            result._is_param = True
    memo[id(tensor)] = result
    return result


def empty_representation(
    value: torch.Tensor,
    like: torch.Tensor,
    *,
    owners: dict[int, torch.Tensor] | None = None,
    memo: dict[int, Any] | None = None,
) -> torch.Tensor:
    """Allocate component storage beside a tensor, preserving views and ties."""
    if owners is None:
        owners = {}

    def allocate(component: torch.Tensor) -> torch.Tensor:
        storage = component.untyped_storage()
        if storage._cdata not in owners:
            owners[storage._cdata] = like.new_empty(
                (storage.nbytes(),), dtype=torch.uint8
            )
        bytes_for_dtype = (
            storage.nbytes() // component.element_size() * component.element_size()
        )
        result = (
            owners[storage._cdata][:bytes_for_dtype]
            .view(component.dtype)
            .as_strided(component.shape, component.stride(), component.storage_offset())
        )
        result.requires_grad_(component.requires_grad)
        return (
            nn.Parameter(result, requires_grad=component.requires_grad)
            if isinstance(component, nn.Parameter)
            else result
        )

    return map_tensor(value, allocate, memo=memo)


def materialize_meta_state(model: nn.Module) -> None:
    """Allocate empty CPU state while retaining parameter ties and physical views."""
    owners: dict[int, torch.Tensor] = {}
    memo: dict[int, Any] = {}
    like = torch.empty(0, device="cpu")
    for module in model.modules():
        for registry in (module._parameters, module._buffers):
            for name, value in registry.items():
                if value is not None and value.is_meta:
                    cast(dict[str, torch.Tensor | None], registry)[name] = (
                        empty_representation(value, like, owners=owners, memo=memo)
                    )


__all__ = [
    "component_at",
    "empty_representation",
    "is_wrapper",
    "map_tensor",
    "tensor_components",
]
