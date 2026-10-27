"""Bounded tensor defaults; conversions write directly into the supplied lease."""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import torch

from .storage import dtype_of

if TYPE_CHECKING:
    from .storage import Storage


def fill_default(
    state: "Storage",
    object_id: str,
    destination: torch.Tensor,
    start: int,
    initializer: Mapping[str, Any],
    named: Mapping[str, str],
) -> None:
    kind = initializer["kind"]
    if kind == "constant":
        destination.fill_(initializer["value"])
    elif kind == "value":
        value = torch.as_tensor(initializer["value"], dtype=destination.dtype)
        if value.numel() == 1:
            destination.fill_(value.item())
        else:
            if not value.is_contiguous() or tuple(value.shape) != tuple(
                state.geometry[object_id]["shape"]
            ):
                raise ValueError("initializer shape differs from the built tensor")
            destination.copy_(value.reshape(-1)[start : start + destination.numel()])
    elif kind == "parameter":
        _copy_parameter(
            state, object_id, destination, start, named[initializer["name"]]
        )
    else:
        raise ValueError(f"unknown tensor initializer: {kind}")


def _copy_parameter(
    state: "Storage", target: str, destination: torch.Tensor, start: int, source: str
) -> None:
    target_region, source_region = state.regions.get(target), state.regions.get(source)
    geometry = (
        state.geometry[target] if target_region is None else target_region.geometry
    )
    source_geometry = (
        state.geometry[source] if source_region is None else source_region.geometry
    )
    if any(geometry[key] != source_geometry[key] for key in ("shape", "stride")):
        raise ValueError("parameter initializer requires matching storage layouts")
    dtype = dtype_of(source_geometry)
    alias = state.aliases[source]
    global_start = start + (0 if target_region is None else target_region.offset)
    source_start = global_start - (0 if source_region is None else source_region.offset)
    source_length = (
        state.spill.sizes[alias] // dtype.itemsize
        if source_region is None
        else source_region.length
    )
    if source_start < 0 or source_start + destination.numel() > source_length:
        raise ValueError("parameter initializer requires the matching local shard")
    with state.spill.buffer(
        alias, source_start * dtype.itemsize, destination.numel() * dtype.itemsize
    ) as values:
        destination.copy_(values.view(dtype))
