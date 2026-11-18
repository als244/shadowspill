"""Materialize declared timing samples directly into bounded physical buffers."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from hashlib import sha256
from types import MappingProxyType

import numpy as np

from shadowspill.training.definitions.samples import (
    RepresentativePattern,
    UniformIntegers,
)
from shadowspill.training.inputs.adapters import AdaptedInputs, InputBinding
from shadowspill.training.inputs.materialize import (
    BufferContents,
    DatumRegion,
    buffer_size,
    materialize_contents,
)
from shadowspill.training.inputs.routing import route_buffer
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.inputs.packing import atomic_groups
from shadowspill_client.inputs.validation import Path
from shadowspill_client.inputs.values import ValidatedTensor, validate_values
from shadowspill_client.protocol.tensors import TensorPayload, storage_dtype


def _drawn(
    path: Path, slot: int, described: ValidatedTensor, spec: UniformIntegers
) -> ValidatedTensor:
    """One Datum's values for a drawn field.

    The field and the Datum's place in the pattern fix the draw, so every rank
    and every preparation of a pattern sees the same values.
    """
    seed = sha256(f"{'/'.join(path)}:{slot}".encode()).digest()[:8]
    values = np.random.default_rng(int.from_bytes(seed, "little")).integers(
        spec.low, spec.high, size=described.shape, dtype=np.int64, endpoint=True
    )
    return ValidatedTensor(
        described.dtype,
        described.shape,
        source=TensorPayload(
            described.dtype,
            described.shape,
            values.astype(storage_dtype(described.dtype)).tobytes(),
        ),
    )


def _contents(binding: InputBinding, pattern: RepresentativePattern) -> BufferContents:
    schema, physical = binding.layout.schema, binding.layout.buffer
    described = pattern.describe(schema, physical)
    definitions = {tuple(f.path): f for f in schema.fields}
    # Validate source and rounded scalar representations before filling a whole
    # physical field. This uses admitted scratch, including BF16 conversion.
    for sample_fields in described:
        for path, value in sample_fields.items():
            scalar = np.asarray(value.default)
            validate_values(
                TensorPayload(scalar.dtype.name, (), scalar.tobytes()),
                definitions[path],
                f"representative {path}",
            )
            target = np.empty((), dtype=storage_dtype(value.dtype))
            ValidatedTensor(value.dtype, (), default=value.default).copy_into(target)
            validate_values(
                TensorPayload(value.dtype, (), target.tobytes()),
                definitions[path],
                f"converted representative {path}",
            )
    offsets = dict.fromkeys(physical.fields, 0)
    values: list[Mapping[Path, ValidatedTensor]] = []
    regions = []
    partition = binding.layout.placement.partition_index(binding.layout.logical_rank)
    for sample, fields in zip(pattern.samples, described, strict=True):
        drawn = {
            path: spec
            for path, spec in sample.values.items()
            if isinstance(spec, UniformIntegers)
        }
        for _ in range(sample.count):
            slot = len(values)
            axes: dict[Path, tuple[int | slice, ...]] = {}
            for path, value in fields.items():
                definition = definitions[path]
                region: list[int | slice] = [slice(0, size) for size in value.shape]
                if definition.pack_axis is not None:
                    axis = definition.pack_axis
                    start = offsets[path]
                    offsets[path] += value.shape[axis]
                    region[axis] = slice(start, offsets[path])
                elif definition.batching == "stack":
                    region.insert(0, slot)
                axes[path] = tuple(region)
            values.append(
                MappingProxyType(
                    {
                        **fields,
                        **{
                            path: _drawn(path, slot, fields[path], spec)
                            for path, spec in drawn.items()
                        },
                    }
                )
                if drawn
                else fields
            )
            regions.append(
                DatumRegion(slot, partition, 0, slot, MappingProxyType(axes))
            )
    return BufferContents(
        tuple(regions), atomic_groups(pattern.count, pattern.groups), tuple(values)
    )


@contextmanager
def representative_inputs(
    binding: InputBinding, pattern: RepresentativePattern, *, max_bytes: int
) -> Iterator[AdaptedInputs]:
    """Reserve before region/array creation and release every owner on failure."""
    layout = binding.layout
    physical = buffer_size(layout.buffer, layout.schema, pattern.count)
    peak = max(physical.staging, physical.retained + binding.staging_bytes)
    if peak > max_bytes:
        raise ValueError(
            f"representative staging needs {peak} bytes; limit is {max_bytes}"
        )
    budget = BufferBudget(max_bytes, max_requests=2)
    try:
        partition = layout.placement.partition_index(layout.logical_rank)
        with (
            materialize_contents(
                layout.buffer,
                layout.schema,
                pattern.count,
                (partition, 0),
                lambda: _contents(binding, pattern),
                budget,
                timeout=0,
            ) as physical_buffer,
            binding.adapt(
                route_buffer(physical_buffer, layout.placement, layout.logical_rank),
                budget,
                timeout=0,
            ) as adapted,
        ):
            yield adapted
    finally:
        budget.close()
        assert budget.used == (0, 0)
