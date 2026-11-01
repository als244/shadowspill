"""ShadowSpill's initial values, for every engine that builds the catalog model.

The catalog initializer ShadowSpill fills its pools from derives each value
from the seed, the tensor's name and its block offset, never from a chunk's
size or a rank's shard. An engine that asks it for the same names at the same
seed therefore starts from the same model: each value drawn at the dtype
ShadowSpill stores that parameter at, so a master weight widened from it holds
what ShadowSpill's master holds.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from workloads.models.shared.routed_lora import FACTORS
from workloads.recipes.text.initialization import Initialize


@dataclass(frozen=True)
class Chunk:
    """The elements of the tensor ``name`` the initializer fills, from
    ``offset`` within its ``storage_elements``."""

    name: str
    shape: tuple[int, ...]
    offset: int
    storage_elements: int
    storage_offset: int = 0


def fresh(name: str) -> bool:
    """A parameter no checkpoint holds: a LoRA factor the run creates."""
    return name.rsplit(".", 1)[-1] in {"lora_a", "lora_b", *FACTORS}


def shard_rows(rows: int, world: int, rank: int) -> tuple[int, int]:
    """The first row and row count of ``rank``'s piece of dimension zero, as
    ``torch.chunk`` splits it and FSDP2 shards it: the last ranks may be
    empty."""

    per = -(-rows // world)
    first = min(rank * per, rows)
    return first, min(per, rows - first)


class SharedInitialization:
    """The catalog initializer ShadowSpill's engine builds for ``request`` at
    ``seed``, and the dtype ShadowSpill stores each parameter at: the dtype
    it has in the model built at ``model_dtype``."""

    def __init__(self, request: Any, seed: int, *, model_dtype: str) -> None:
        self.generate = Initialize(
            request.entry.family,
            request.entry.config(**request.config),
            seed=seed,
        )
        self.stored = {
            name: parameter.dtype
            for name, parameter in request.build(dtype=model_dtype).named_parameters()
        }

    def values(
        self, name: str, shape: tuple[int, ...], start: int = 0, stop: int | None = None
    ) -> torch.Tensor:
        """Elements ``[start, stop)`` of the parameter ``name``, flat, at the
        dtype ShadowSpill stores it at."""

        total = math.prod(shape)
        stop = total if stop is None else stop
        destination = torch.empty(stop - start, dtype=self.stored[name])
        self.generate(Chunk(name, tuple(shape), start, total), destination)
        return destination


@torch.no_grad()
def fill_parameters(
    module: nn.Module,
    shared: SharedInitialization,
    *,
    only: Callable[[str], bool] | None = None,
    rename: Callable[[str], str] = lambda name: name,
    copy: Callable[[torch.Tensor, torch.Tensor | None], None] | None = None,
    needs_values: Callable[[torch.Tensor], bool] = lambda parameter: True,
) -> list[str]:
    """Write ShadowSpill's initial values into every parameter of ``module``
    that ``only`` selects (all when it is None), under its name as ``rename``
    gives it; returns the names written.

    A dimension-zero sharded parameter (FSDP2's) gets this rank's rows, drawn
    alone. Any other goes through ``copy(parameter, value)`` with the whole
    value, or ``None`` where ``needs_values`` says the rank does not write it
    (a partitioned parameter written by its first rank); by default the value
    is copied in place.
    """

    written = []
    for raw, parameter in module.named_parameters():
        name = rename(raw)
        if only is not None and not only(name):
            continue
        shape = tuple(getattr(parameter, "ds_shape", parameter.shape))
        if hasattr(parameter, "to_local") and hasattr(parameter, "device_mesh"):
            placements = parameter.placements
            if (
                len(placements) != 1
                or not placements[0].is_shard()
                or placements[0].dim != 0
            ):
                raise ValueError(f"{name}: only dimension-zero shards are filled")
            local = parameter.to_local()
            mesh = parameter.device_mesh
            first, count = shard_rows(shape[0], mesh.size(), mesh.get_local_rank())
            if count != local.shape[0]:
                raise ValueError(
                    f"{name}: this rank's shard holds {local.shape[0]} rows,"
                    f" not {count}"
                )
            row = math.prod(shape[1:])
            values = shared.values(name, shape, first * row, (first + count) * row)
            local.copy_(values.view(local.shape))
        else:
            value = (
                shared.values(name, shape).view(shape)
                if needs_values(parameter)
                else None
            )
            if copy is None:
                parameter.copy_(value)
            else:
                copy(parameter, value)
        written.append(name)
    return written


__all__ = [
    "Chunk",
    "SharedInitialization",
    "fill_parameters",
    "fresh",
    "shard_rows",
]
