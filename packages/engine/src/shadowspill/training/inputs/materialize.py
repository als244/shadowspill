"""Fill bounded physical buffers and retain exact caller-order field regions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Self

import numpy as np
from numpy.typing import NDArray

from shadowspill_client.buffers import BufferBudget, BufferLease
from shadowspill_client.protocol.inputs import InputSchema
from shadowspill_client.protocol.tensors import storage_dtype, tensor_bytes

from .packing import PackedAssignment
from .shapes import BufferShape
from .validation import Path
from .values import VALIDATION_SCRATCH_BYTES, ValidatedTensor


@dataclass(frozen=True)
class DatumRegion:
    datum: int
    partition: int
    microbatch: int
    slot: int
    fields: Mapping[Path, tuple[int | slice, ...]]


@dataclass(frozen=True)
class BufferSize:
    retained: int
    staging: int


def buffer_size(physical: BufferShape, schema: InputSchema, records: int) -> BufferSize:
    """Use the same physical and region accounting in preparation and copying."""
    if type(records) is not int or not 0 <= records <= physical.max_datums:
        raise ValueError("record count is outside the physical buffer capacity")
    definitions = {tuple(field.path): field for field in schema.fields}
    physical.validate(definitions)
    size = sum(
        tensor_bytes(definitions[path].dtype, tuple(shape))
        for path, shape in physical.fields.items()
    )
    retained = size + 256 * (records + 1) * (
        1 + sum(1 + len(shape) for shape in physical.fields.values())
    )
    return BufferSize(retained, retained + VALIDATION_SCRATCH_BYTES)


class PackedBuffer:
    def __init__(
        self,
        fields: Mapping[Path, NDArray[Any]],
        regions: tuple[DatumRegion, ...],
        groups: tuple[tuple[int, ...], ...],
        location: tuple[int, int],
        lease: BufferLease,
    ) -> None:
        self.fields = MappingProxyType(dict(fields))
        self.regions = regions
        self.groups = groups
        self.location = location
        self._lease = lease
        self.closed = False

    def close(self) -> None:
        self.fields = MappingProxyType({})
        self.closed = True
        self._lease.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def field_regions(
    assignment: PackedAssignment, schema: InputSchema, buffer: int
) -> tuple[DatumRegion, ...]:
    """Inverse maps retain Datum identity through sorting, grouping and padding."""
    if type(buffer) is not int or not 0 <= buffer < len(assignment.schedule.buffers):
        raise ValueError("buffer index is outside the prepared schedule")
    definitions = {tuple(field.path): field for field in schema.fields}
    physical = assignment.schedule.buffers[buffer]
    partition, microbatch = assignment.schedule.locations[buffer]
    offsets = {path: 0 for path in physical.fields}
    regions = []
    for slot, index in enumerate(assignment.records[buffer]):
        fields: dict[Path, tuple[int | slice, ...]] = {}
        for path in physical.fields:
            definition = definitions[path]
            value = assignment.batch.fields[index][path]
            axes: list[int | slice] = [slice(0, size) for size in value.shape]
            if definition.pack_axis is not None:
                axis = definition.pack_axis
                start, size = offsets[path], value.shape[axis]
                axes[axis] = slice(start, start + size)
                offsets[path] += size
            elif definition.batching == "stack":
                axes.insert(0, slot)
            fields[path] = tuple(axes)
        regions.append(
            DatumRegion(index, partition, microbatch, slot, MappingProxyType(fields))
        )
    return tuple(regions)


def materialize_buffer(
    assignment: PackedAssignment,
    schema: InputSchema,
    index: int,
    budget: BufferBudget,
    *,
    timeout: float | None = None,
) -> PackedBuffer:
    """Own one fixed physical buffer; release after its worker staging completes.

    All logical validation and packing are already complete. The model's input
    adapter uses regions/groups to construct admitted sequence/media metadata.
    The engine does not invent attention or encoder boundaries from capacity.
    """
    if assignment.batch.owner.closed:
        raise RuntimeError("input snapshot is closed")
    definitions = {tuple(field.path): field for field in schema.fields}
    physical = assignment.schedule.buffers[index]
    # Fixed conversion scratch and bounded region/container bookkeeping are
    # separate from persistent spill and from the original owned input payload.
    size = buffer_size(physical, schema, len(assignment.records[index]))
    lease = budget.reserve(size.staging, timeout=timeout)
    try:
        regions = field_regions(assignment, schema, index)
        fields = {}
        for path, shape in physical.fields.items():
            definition = definitions[path]
            destination = np.empty(shape, dtype=storage_dtype(definition.dtype))
            assert isinstance(definition.padding, (bool, int, float))
            ValidatedTensor(
                definition.dtype, tuple(shape), default=definition.padding
            ).copy_into(destination)
            fields[path] = destination
        copied_shared = set()
        for region in regions:
            for path, axes in region.fields.items():
                if definitions[path].batching == "shared":
                    if path in copied_shared:
                        continue
                    copied_shared.add(path)
                # Scalar stack indexing produces a NumPy scalar. A one-element
                # slice followed by reshape keeps a writable zero-dimensional view.
                destination = fields[path]
                if axes and isinstance(axes[0], int):
                    destination = destination[
                        slice(axes[0], axes[0] + 1), *axes[1:]
                    ].squeeze(axis=0)
                elif axes:
                    destination = destination[axes]
                assignment.batch.fields[region.datum][path].copy_into(destination)
        result = PackedBuffer(
            fields,
            regions,
            assignment.groups[index],
            assignment.schedule.locations[index],
            lease,
        )
        lease.shrink(size.retained)
        return result
    except BaseException:
        lease.close()
        raise
