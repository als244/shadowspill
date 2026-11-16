"""Logical worker views over an already owned physical input buffer."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from numpy.typing import NDArray

from shadowspill.parallelism import InputPlacement
from shadowspill_client.inputs.validation import Path

from .materialize import PackedBuffer


@dataclass(frozen=True)
class RoutedBuffer:
    owner: PackedBuffer
    logical_rank: int
    contributes_loss: bool
    fields: Mapping[Path, NDArray[Any]]


def route_buffer(
    buffer: PackedBuffer, placement: InputPlacement, logical_rank: int
) -> RoutedBuffer:
    """Select declared aligned slices/copies, never infer ownership from DP/EP.

    Views borrow the physical buffer until worker staging has finished. The
    installed model/objective adapter must honor loss ownership and execute its
    required collectives even with an empty local slice or no local contribution.
    This function supplies data; it does not add distributed model operators.
    """
    if buffer.closed:
        raise RuntimeError("packed buffer is closed")
    if placement.partition_index(logical_rank) != buffer.location[0]:
        raise ValueError("logical worker does not own this input partition")
    fields = {}
    for path, array in buffer.fields.items():
        # Require a declaration even for a scalar/shared metadata leaf.
        region = placement.region(path, array.shape, logical_rank)
        view = array[region] if array.ndim else array.view()
        view = view.view()
        view.setflags(write=False)
        fields[path] = view
    return RoutedBuffer(
        buffer,
        logical_rank,
        placement.contributes_loss(logical_rank),
        MappingProxyType(fields),
    )
