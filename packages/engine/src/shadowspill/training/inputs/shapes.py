"""Fixed physical input shapes and scheduled capacities of prepared families."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from shadowspill.parallelism import InputPlacement
from shadowspill_client import BatchConfig
from shadowspill_client._records import number
from shadowspill_client.protocol.inputs import InputSchema, TensorField

from .validation import Path


@dataclass(frozen=True)
class BufferShape:
    """Physical tensor shapes include the leading record axis of stacked fields."""

    max_datums: int
    fields: Mapping[Path, Sequence[int]]

    def __post_init__(self) -> None:
        if type(self.max_datums) is not int or self.max_datums < 1:
            raise ValueError("max_datums must be a positive integer")
        shapes = {}
        for path, shape in self.fields.items():
            if (
                not isinstance(path, tuple)
                or not path
                or any(not isinstance(p, str) or not p for p in path)
            ):
                raise ValueError(
                    "physical field paths must be nonempty tuples of names"
                )
            if any(type(n) is not int or n < 0 for n in shape):
                raise ValueError("physical dimensions must be nonnegative integers")
            shapes[path] = tuple(shape)
        if not shapes:
            raise ValueError("a physical buffer must declare its tensor fields")
        object.__setattr__(self, "fields", MappingProxyType(shapes))

    def validate(self, definitions: Mapping[Path, TensorField]) -> None:
        for path, shape in self.fields.items():
            if path not in definitions:
                raise ValueError(f"physical shape names unknown field {path}")
            field = definitions[path]
            stacked = field.batching == "stack"
            if len(shape) != len(field.shape) + stacked or (
                stacked and shape[0] != self.max_datums
            ):
                raise ValueError(
                    f"physical shape/record axis differs from schema: {path}"
                )
            for axis, declared in enumerate(field.shape):
                actual = shape[axis + stacked]
                if (
                    type(declared) is int
                    and field.pack_axis != axis
                    and actual != declared
                ):
                    raise ValueError(
                        f"physical fixed dimension differs from schema: {path}"
                    )

    def resources(self, definitions: Mapping[Path, TensorField]) -> tuple[int, ...]:
        capacities = [self.max_datums]
        for path in packed_paths(definitions, self.fields):
            axis = definitions[path].pack_axis
            assert axis is not None
            capacities.append(self.fields[path][axis])
        return tuple(capacities)

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_datums": self.max_datums,
            "fields": [
                {"path": list(path), "shape": list(shape)}
                for path, shape in sorted(self.fields.items())
            ],
        }


def packed_paths(
    definitions: Mapping[Path, TensorField], paths: Mapping[Path, Any]
) -> tuple[Path, ...]:
    return tuple(path for path in sorted(paths) if definitions[path].batching == "pack")


@dataclass(frozen=True)
class PackingSchedule:
    """One prepared complete distributed schedule, indexed by input partition.

    Cooperating physical copies are absent here. Empty model buffers still
    appear and count at full capacity; preparation owns collective compatibility.
    """

    name: str
    partitions: Sequence[Sequence[BufferShape]]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("packing schedule requires a name")
        partitions = tuple(tuple(part) for part in self.partitions)
        if (
            not partitions
            or not any(partitions)
            or any(not isinstance(b, BufferShape) for part in partitions for b in part)
        ):
            raise ValueError("packing schedule requires declared physical buffers")
        fields = next(buffer.fields.keys() for part in partitions for buffer in part)
        if any(
            buffer.fields.keys() != fields for part in partitions for buffer in part
        ):
            raise ValueError(
                "every scheduled buffer must describe the same field closure"
            )
        object.__setattr__(self, "partitions", partitions)

    @property
    def buffers(self) -> tuple[BufferShape, ...]:
        return tuple(buffer for part in self.partitions for buffer in part)

    @property
    def locations(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (part, index)
            for part, buffers in enumerate(self.partitions)
            for index in range(len(buffers))
        )

    def capacities(self, schema: InputSchema) -> Mapping[str, int]:
        definitions = {tuple(field.path): field for field in schema.fields}
        for buffer in self.buffers:
            buffer.validate(definitions)
        totals = {}
        for capacity in schema.capacities:
            if not capacity.scheduled:
                continue
            if capacity.reduction == "count":
                total = sum(buffer.max_datums for buffer in self.buffers)
            else:
                path = tuple(capacity.path)
                if path not in self.buffers[0].fields:
                    raise ValueError(f"scheduled capacity field is absent: {path}")
                field = definitions[path]
                assert capacity.axis is not None
                axis = capacity.axis + (field.batching == "stack")
                if (
                    capacity.reduction == "sum"
                    and field.batching != "stack"
                    and field.pack_axis != capacity.axis
                ):
                    raise ValueError(
                        "scheduled sums require a stacked field or its packed axis"
                    )
                sizes = [
                    buffer.fields[path][axis]
                    * (
                        buffer.max_datums
                        if field.batching == "stack" and capacity.reduction == "sum"
                        else 1
                    )
                    for buffer in self.buffers
                ]
                total = (
                    sum(sizes) if capacity.reduction == "sum" else max(sizes, default=0)
                )
            totals[capacity.name] = total
        return MappingProxyType(totals)

    def validate(
        self, schema: InputSchema, batch: BatchConfig, placement: InputPlacement
    ) -> None:
        if len(self.partitions) != placement.partition_count:
            raise ValueError("schedule must describe every logical input partition")
        for name, total in self.capacities(schema).items():
            if name not in batch.limits:
                raise ValueError(f"missing scheduled capacity limit: {name}")
            if total > number(batch.limits[name], name):
                raise ValueError(
                    f"schedule {self.name}: scheduled {name}={total} "
                    f"exceeds cap {batch.limits[name]}, including padding"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "name": self.name,
            "partitions": [
                [buffer.to_dict() for buffer in part] for part in self.partitions
            ],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PackingSchedule:
        if (
            type(value.get("version")) is not int
            or value["version"] != 1
            or set(value) != {"version", "name", "partitions"}
        ):
            raise ValueError("unsupported packing schedule record")
        partitions = []
        for part in value["partitions"]:
            buffers = []
            for buffer in part:
                if set(buffer) != {"max_datums", "fields"}:
                    raise ValueError("invalid physical buffer record")
                if any(
                    not isinstance(item, Mapping)
                    or set(item) != {"path", "shape"}
                    or not isinstance(item["path"], list)
                    for item in buffer["fields"]
                ):
                    raise ValueError("invalid physical tensor shape record")
                paths = [tuple(item["path"]) for item in buffer["fields"]]
                if len(paths) != len(set(paths)):
                    raise ValueError("duplicate physical buffer field")
                buffers.append(
                    BufferShape(
                        buffer["max_datums"],
                        {
                            path: item["shape"]
                            for path, item in zip(paths, buffer["fields"], strict=True)
                        },
                    )
                )
            partitions.append(buffers)
        return cls(value["name"], partitions)
