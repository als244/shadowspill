"""Saved target defaults for conversion, without opening fresh model weights."""

from collections.abc import Sequence
from math import prod
from typing import Any

import torch

from shadowspill.artifacts.tensor_spans import spans
from shadowspill.artifacts.tensors import TensorChunk

from .state import PlanState
from .tensors import byte_view, geometry, region_bytes


def _shape(value: Any) -> tuple[int, ...]:
    if isinstance(value, (tuple, list)):
        if not value:
            return (0,)
        inner = _shape(value[0])
        if any(_shape(item) != inner for item in value):
            raise ValueError("target temporary initializer is ragged")
        return (len(value), *inner)
    if type(value) not in (bool, int, float):
        raise ValueError("target temporary initializer requires portable numbers")
    return ()


def _at(value: Any, shape: tuple[int, ...], index: int) -> Any:
    for axis, size in enumerate(shape):
        inner = prod(shape[axis + 1 :])
        value = value[(index // inner) % size]
    return value


class TemporaryDefaults:
    """Preflight saved declarations and fill only the borrowed destination.

    Decoded values belong to the owner's admitted Plan metadata. No tensor or
    flattened copy of a complete constant is created. Non-scalar values retain
    their declared strided geometry, including owned subregions and zero padding.
    Missing/external defaults fail before conversion reads any source tensors.
    """

    def __init__(self, states: Sequence[PlanState]) -> None:
        self._values: dict[
            tuple[int, str], tuple[TensorChunk, Any, tuple[int, ...]]
        ] = {}
        for state in states:
            for chunk, initial in state.temporary_initializers():
                if not region_bytes(chunk):
                    continue
                if initial is None or initial.get("kind") not in {"constant", "value"}:
                    raise ValueError(
                        f"target temporary has no portable default: {chunk.name}"
                    )
                value = initial["value"]
                shape = _shape(value)
                if prod(shape) == 1:
                    value, shape = _at(value, shape, 0), ()
                elif initial["kind"] == "constant" or shape != chunk.shape:
                    raise ValueError("target temporary initializer shape differs")
                if shape:
                    extent = 1
                    for stride, size in sorted(zip(chunk.stride, shape, strict=True)):
                        if size > 1:
                            if stride < extent:
                                raise ValueError("target temporary view overlaps")
                            extent += (size - 1) * stride
                key = state.logical_rank, chunk.name
                if key in self._values:
                    raise ValueError("duplicate target temporary initializer")
                self._values[key] = chunk, value, shape

    def __call__(
        self, rank: int, request: TensorChunk, destination: torch.Tensor
    ) -> None:
        entry = self._values.get((rank, request.name))
        if entry is None:
            raise ValueError("checkpoint state cannot use fresh target defaults")
        chunk, value, shape = entry
        byte_view(destination)
        start, stop = request.offset, request.offset + destination.numel()
        if (
            geometry(request) != geometry(chunk)
            or destination.dtype != chunk.dtype
            or start < chunk.region_offset
            or stop > chunk.region_offset + region_bytes(chunk) // chunk.dtype.itemsize
        ):
            raise ValueError("target temporary request differs from its owned region")
        if not shape:
            destination.fill_(value)
            return
        destination.zero_()
        for logical, physical, count in spans(
            shape, chunk.stride, chunk.storage_offset
        ):
            begin, end = max(start, physical), min(stop, physical + count)
            for index in range(begin, end):
                destination[index - start] = _at(
                    value, shape, logical + index - physical
                )
