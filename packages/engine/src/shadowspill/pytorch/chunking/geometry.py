"""Track a logical chunk factor through shape and layout changes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from shadowspill.errors import CaptureError
from shadowspill.task.storage import TaskStorageContract

if TYPE_CHECKING:
    from shadowspill.pytorch.capture.artifacts import GraphArtifact


@dataclass(frozen=True, slots=True)
class ChunkAxis:
    """One dimension contains ``before * extent * after`` ordered elements."""

    dim: int
    before: int = 1
    after: int = 1

    def __post_init__(self) -> None:
        if any(
            isinstance(x, bool) or not isinstance(x, int)
            for x in (self.dim, self.before, self.after)
        ):
            raise TypeError("chunk axis dimensions and factors must be integers")
        if self.before < 1 or self.after < 1:
            raise ValueError("chunk axis factors must be positive")

    def shape(self, shape: tuple[int, ...], extent: int) -> tuple[int, ...]:
        result = list(shape)
        result[self.dim] = self.before * extent * self.after
        return tuple(result)

    def view(self, value: torch.Tensor, extent: int) -> torch.Tensor:
        shape = list(value.shape)
        shape[self.dim : self.dim + 1] = [self.before, extent, self.after]
        strides = list(value.stride())
        stride = strides[self.dim]
        strides[self.dim : self.dim + 1] = [
            stride * extent * self.after,
            stride * self.after,
            stride,
        ]
        return value.as_strided(shape, strides)

    def slice(
        self, value: torch.Tensor, extent: int, start: int, count: int
    ) -> torch.Tensor:
        return (
            self.view(value, extent)
            .narrow(self.dim + 1, start, count)
            .reshape(self.shape(tuple(value.shape), count))
        )


@dataclass(frozen=True, slots=True)
class LocalLayout:
    """Chunk-owned coordinates that must not escape into a full boundary."""

    axis: ChunkAxis | None


REDUCED = "sum"
Layout = ChunkAxis | LocalLayout | str | None


def reshape_axis(
    axis: ChunkAxis, old: tuple[int, ...], new: tuple[int, ...], rows: int
) -> ChunkAxis:
    before = math.prod(old[: axis.dim]) * axis.before
    after = math.prod(old[axis.dim + 1 :]) * axis.after
    candidates = []
    for dim, extent in enumerate(new):
        prefix, suffix = math.prod(new[:dim]), math.prod(new[dim + 1 :])
        if not prefix or not suffix or before % prefix or after % suffix:
            continue
        left, right = before // prefix, after // suffix
        if extent == left * rows * right:
            candidate = ChunkAxis(dim, left, right)
            if rows != 1:
                return candidate
            candidates.append(candidate)
    if candidates:
        # A singleton can also be represented inside either neighboring axis.
        # Prefer its explicit dimension when available: packing it into heads
        # or features changes its apparent position after a later transpose.
        return min(candidates, key=lambda item: item.before * item.after)
    raise CaptureError(
        f"reshape {old} -> {new} does not preserve the logical chunk factor"
    )


@dataclass(frozen=True, slots=True)
class ChunkTask:
    """Build-time composition; runtime payloads contain only plain metadata/code."""

    rows: int
    size: int
    inputs: tuple[Layout, ...]
    outputs: tuple[Layout, ...]
    children: tuple[tuple[int, GraphArtifact], ...]
    accumulating_children: tuple[tuple[int, GraphArtifact], ...]
    identity: dict[str, Any]
    storage_contract: TaskStorageContract
