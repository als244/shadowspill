"""Conservative integer envelopes implied by the complete validation contract."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import floor
from types import MappingProxyType

from shadowspill_client import BatchConfig
from shadowspill_client._records import number
from shadowspill_client.protocol.inputs import InputSchema, TensorField

from .shapes import BufferShape
from .validation import Path


class CoverageError(ValueError):
    """Requested shapes do not have a supported proof of advertised coverage."""


@dataclass(frozen=True)
class InputBounds:
    records: int
    individual: Mapping[Path, tuple[int, ...]]
    total: Mapping[Path, tuple[int, ...]]


def input_bounds(
    schema: InputSchema, config: BatchConfig, fields: Mapping[Path, object]
) -> InputBounds:
    if config.limits.keys() != {cap.name for cap in schema.capacities}:
        raise ValueError("batch limits must name exactly the schema capacities")
    definitions = {tuple(field.path): field for field in schema.fields}
    if fields.keys() - definitions.keys():
        raise ValueError("physical shape names an unknown field")
    limits = {name: floor(number(v, name)) for name, v in config.limits.items()}
    counts = [limits[cap.name] for cap in schema.capacities if cap.reduction == "count"]
    # A fixed positive dimension with a sum bound also bounds record count.
    # Variable axes can be zero: their token sum alone never bounds empty Datums.
    for cap in schema.capacities:
        if cap.reduction == "sum" and tuple(cap.path) in fields:
            assert cap.axis is not None
            dimension = definitions[tuple(cap.path)].shape[cap.axis]
            if isinstance(dimension, int) and dimension > 0:
                counts.append(limits[cap.name] // dimension)
    if not counts:
        raise CoverageError(
            "advertised Datum count is unbounded; declare a count limit"
        )
    records = min(counts)
    maxima: dict[object, int] = {}
    totals: dict[object, int] = {}

    def key(path: Path, axis: int) -> object:
        dimension = definitions[path].shape[axis]
        return dimension if isinstance(dimension, str) else (path, axis)

    for cap in schema.capacities:
        path = tuple(cap.path)
        # A capacity on an omitted field cannot bound a selected aligned field.
        if cap.reduction == "count" or path not in fields:
            continue
        assert cap.axis is not None
        identity, limit = key(path, cap.axis), limits[cap.name]
        maxima[identity] = min(maxima.get(identity, limit), limit)
        if cap.reduction == "sum":
            totals[identity] = min(totals.get(identity, limit), limit)
    individual, aggregate = {}, {}
    for path in fields:
        local, summed = [], []
        for axis, dimension in enumerate(definitions[path].shape):
            identity = key(path, axis)
            if records == 0:
                maximum = 0
            elif isinstance(dimension, int):
                maximum = min(dimension, maxima.get(identity, dimension))
            elif identity in maxima:
                maximum = maxima[identity]
            else:
                raise CoverageError(f"unbounded dimension {dimension!r} at {path}")
            local.append(maximum)
            summed.append(
                min(records * maximum, totals.get(identity, records * maximum))
            )
        individual[path], aggregate[path] = tuple(local), tuple(summed)
    return InputBounds(
        records, MappingProxyType(individual), MappingProxyType(aggregate)
    )


def shared_constant(field: TensorField) -> bool:
    # Without finite validation NaNs could evade both numeric inequalities.
    return field.finite and field.minimum is not None and field.minimum == field.maximum


def shape_covers(
    buffer: BufferShape, definitions: Mapping[Path, TensorField], bounds: InputBounds
) -> bool:
    for path, physical in buffer.fields.items():
        field = definitions[path]
        for axis, maximum in enumerate(bounds.individual[path]):
            if axis == field.pack_axis:
                continue
            if field.batching == "shared":
                dimension = field.shape[axis]
                # Shared fields cannot be independently padded for each record.
                if (isinstance(dimension, str) and maximum != 0) or maximum != physical[
                    axis
                ]:
                    return False
            elif maximum > physical[axis + (field.batching == "stack")]:
                return False
    return True


def guaranteed_slots(
    buffer: BufferShape, definitions: Mapping[Path, TensorField], bounds: InputBounds
) -> int:
    if not shape_covers(buffer, definitions, bounds):
        return 0
    slots = min(buffer.max_datums, bounds.records)
    for path, field in definitions.items():
        if path not in buffer.fields:
            continue
        if field.pack_axis is not None:
            maximum = bounds.individual[path][field.pack_axis]
            if maximum:
                slots = min(slots, buffer.fields[path][field.pack_axis] // maximum)
        elif field.batching == "shared" and not shared_constant(field):
            slots = min(slots, 1)
    return slots


def covers_aggregate(
    buffer: BufferShape, definitions: Mapping[Path, TensorField], bounds: InputBounds
) -> bool:
    if buffer.max_datums < bounds.records or not shape_covers(
        buffer, definitions, bounds
    ):
        return False
    for path, field in definitions.items():
        if path not in buffer.fields:
            continue
        if field.pack_axis is not None:
            if (
                bounds.total[path][field.pack_axis]
                > buffer.fields[path][field.pack_axis]
            ):
                return False
        elif (
            field.batching == "shared"
            and bounds.records > 1
            and not shared_constant(field)
        ):
            return False
    return True
