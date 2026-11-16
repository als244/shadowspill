"""Compact representative input descriptions; no arrays or framework objects."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any

from shadowspill_client._records import integer, number, text
from shadowspill_client.inputs.packing import atomic_groups
from shadowspill_client.inputs.shapes import BufferShape
from shadowspill_client.inputs.validation import Path
from shadowspill_client.inputs.values import ValidatedTensor
from shadowspill_client.protocol.controls import Scalar
from shadowspill_client.protocol.inputs import InputSchema


@dataclass(frozen=True)
class RepresentativeSample:
    """Repeat one aligned logical Datum with explicit scalar-valued tensor fills."""

    values: Mapping[Path, Scalar]
    dimensions: Mapping[str, int] = field(default_factory=dict)
    count: int = 1

    def __post_init__(self) -> None:
        integer(self.count, "representative count", minimum=1)
        values = dict(self.values)
        for path, value in values.items():
            if (
                not isinstance(path, tuple)
                or len(path) < 2
                or path[0] not in ("inputs", "loss_inputs")
                or any(not isinstance(p, str) or not p for p in path)
            ):
                raise ValueError("representative values require tensor field paths")
            if type(value) is not bool:
                number(value, "representative scalar")
        dimensions = dict(self.dimensions)
        for name, size in dimensions.items():
            text(name, "representative dimension")
            integer(size, "representative dimension")
        object.__setattr__(self, "values", MappingProxyType(values))
        object.__setattr__(self, "dimensions", MappingProxyType(dimensions))

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "dimensions": dict(self.dimensions),
            "values": [
                {"path": list(path), "value": value}
                for path, value in sorted(self.values.items())
            ],
        }


@dataclass(frozen=True)
class RepresentativePattern:
    """A timing scenario for one physical buffer, independent of admission bounds."""

    label: str
    samples: Sequence[RepresentativeSample]
    groups: Sequence[Sequence[int]] | None = None

    def __post_init__(self) -> None:
        text(self.label, "representative label")
        samples = tuple(self.samples)
        if not samples or any(not isinstance(s, RepresentativeSample) for s in samples):
            raise ValueError("representative pattern requires sample descriptions")
        if sum(s.count for s in samples) > 65536:
            raise ValueError("representative pattern exceeds 65536 records")
        groups = (
            None
            if self.groups is None
            else atomic_groups(sum(s.count for s in samples), self.groups)
        )
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "groups", groups)

    @property
    def count(self) -> int:
        return sum(sample.count for sample in self.samples)

    def describe(
        self, schema: InputSchema, physical: BufferShape
    ) -> tuple[Mapping[Path, ValidatedTensor], ...]:
        """Validate shape/alignment fit once per compressed sample, without arrays."""
        definitions = {tuple(f.path): f for f in schema.fields}
        physical.validate(definitions)
        if self.count > physical.max_datums:
            raise ValueError("representative records exceed physical capacity")
        used = dict.fromkeys(physical.fields, 0)
        shared: dict[Path, tuple[tuple[int, ...], Scalar]] = {}
        result = []
        for sample in self.samples:
            if sample.values.keys() != physical.fields.keys():
                raise ValueError("representative values differ from required fields")
            dimensions = {
                size
                for path in physical.fields
                for size in definitions[path].shape
                if isinstance(size, str)
            }
            if dimensions != sample.dimensions.keys():
                raise ValueError(
                    "representative symbolic dimensions differ from schema"
                )
            values = {}
            for path, value in sample.values.items():
                definition = definitions[path]
                # Reuse portable scalar dtype/range validation without allocating
                # tensors. Actual floating conversion is checked in bounded
                # scratch before the materializer allocates the physical arrays.
                replace(definition, required=False, default=value)
                if (definition.minimum is not None and value < definition.minimum) or (
                    definition.maximum is not None and value > definition.maximum
                ):
                    raise ValueError(
                        f"representative value is outside field bounds: {path}"
                    )
                shape = tuple(
                    sample.dimensions[n] if isinstance(n, str) else n
                    for n in definition.shape
                )
                for axis, actual in enumerate(shape):
                    limit = physical.fields[path][
                        axis + (definition.batching == "stack")
                    ]
                    if definition.pack_axis == axis:
                        used[path] += actual * sample.count
                        actual = used[path]
                    if actual > limit or (
                        definition.batching == "shared" and actual != limit
                    ):
                        raise ValueError(
                            f"representative shape exceeds physical field {path}"
                        )
                if definition.batching == "shared":
                    signature = (shape, value)
                    if shared.setdefault(path, signature) != signature:
                        raise ValueError(
                            "representative shared fields must be identical"
                        )
                values[path] = ValidatedTensor(definition.dtype, shape, default=value)
            result.append(MappingProxyType(values))
        return tuple(result)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "samples": [sample.to_dict() for sample in self.samples],
            "groups": None if self.groups is None else [list(g) for g in self.groups],
        }

    def capacities(
        self, schema: InputSchema, physical: BufferShape
    ) -> Mapping[str, int]:
        described = self.describe(schema, physical)
        totals = {}
        for capacity in schema.capacities:
            if capacity.reduction == "count":
                totals[capacity.name] = self.count
                continue
            assert capacity.axis is not None
            path = tuple(capacity.path)
            sizes = [
                values[path].shape[capacity.axis]
                * (sample.count if capacity.reduction == "sum" else 1)
                for sample, values in zip(self.samples, described, strict=True)
                if path in values
            ]
            totals[capacity.name] = (
                sum(sizes) if capacity.reduction == "sum" else max(sizes, default=0)
            )
        return MappingProxyType(totals)
