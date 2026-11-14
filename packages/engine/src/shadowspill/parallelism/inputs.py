"""Logical input partitions and field delivery, independent of parameter shards."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

from .groups import GroupRef


@dataclass(frozen=True)
class InputFieldPlacement:
    path: tuple[str, ...]
    mode: Literal["shared", "replicate", "slice"] = "replicate"
    axis: int | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.path, str)
            or not self.path
            or any(not isinstance(part, str) or not part for part in self.path)
        ):
            raise ValueError("input field path must contain nonempty names")
        object.__setattr__(self, "path", tuple(self.path))
        if self.mode not in ("shared", "replicate", "slice"):
            raise ValueError("unknown input field delivery mode")
        if self.mode == "slice":
            if type(self.axis) is not int or self.axis < 0:
                raise ValueError("sliced fields require a nonnegative axis")
        elif self.axis is not None:
            raise ValueError("only sliced fields have an axis")

    def region(
        self, shape: Sequence[int], *, index: int, workers: int
    ) -> tuple[slice, ...]:
        """Balanced contiguous slices; short dimensions can give empty regions."""
        if (
            type(workers) is not int
            or workers < 1
            or type(index) is not int
            or not 0 <= index < workers
        ):
            raise ValueError("invalid input partition worker index")
        if any(type(size) is not int or size < 0 for size in shape):
            raise ValueError("input shape must contain nonnegative integers")
        regions = [slice(0, size) for size in shape]
        if self.axis is not None:
            if self.axis >= len(shape):
                raise ValueError("input slice axis exceeds the field rank")
            size = shape[self.axis]
            regions[self.axis] = slice(
                size * index // workers, size * (index + 1) // workers
            )
        return tuple(regions)

    def to_dict(self) -> dict[str, Any]:
        return {"path": list(self.path), "mode": self.mode, "axis": self.axis}


@dataclass(frozen=True)
class OutputPlacement:
    mode: Literal["leader", "sum", "concat"] = "leader"
    axis: int | None = None

    def __post_init__(self) -> None:
        if self.mode not in ("leader", "sum", "concat"):
            raise ValueError("unknown output placement mode")
        if self.mode == "concat":
            if type(self.axis) is not int or self.axis < 0:
                raise ValueError("concatenated outputs require a nonnegative axis")
        elif self.axis is not None:
            raise ValueError("only concatenated outputs have an axis")

    def to_dict(self) -> dict[str, Any]:
        return {"mode": self.mode, "axis": self.axis}


@dataclass(frozen=True)
class InputPlacement:
    """Each group is one logical input partition, in declared logical-rank order.

    ``leader`` loss ownership counts a replicated loss on the first group member.
    ``partitioned`` counts disjoint local contributions from every member. Neither
    changes gradient reductions or adds a denominator; compatible model/objective
    implementations must honor these declarations, including empty local slices.
    Output rules describe assembly within each cooperating group. Packing retains
    the separate inverse map from logical partition positions to caller records.
    """

    partitions: GroupRef
    fields: Sequence[InputFieldPlacement] = ()
    loss: Literal["leader", "partitioned"] = "leader"
    outputs: Mapping[str, OutputPlacement] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.partitions, GroupRef):
            raise TypeError("input partitions must be a GroupRef")
        fields = tuple(self.fields)
        if any(not isinstance(item, InputFieldPlacement) for item in fields):
            raise TypeError("input fields require InputFieldPlacement records")
        if len({item.path for item in fields}) != len(fields):
            raise ValueError("input field placement paths must be unique")
        if self.loss not in ("leader", "partitioned"):
            raise ValueError("unknown loss contribution ownership")
        outputs = dict(self.outputs)
        if any(not isinstance(name, str) or not name for name in outputs):
            raise ValueError("output names must be nonempty strings")
        if any(not isinstance(item, OutputPlacement) for item in outputs.values()):
            raise TypeError("output fields require OutputPlacement records")
        object.__setattr__(self, "fields", fields)
        object.__setattr__(self, "outputs", MappingProxyType(outputs))

    @classmethod
    def independent(cls, world_size: int, **options: Any) -> InputPlacement:
        if type(world_size) is not int or world_size < 1:
            raise ValueError("world_size must be a positive integer")
        return cls(
            GroupRef("inputs", tuple((rank,) for rank in range(world_size))), **options
        )

    def validate(self, world_size: int) -> None:
        if type(world_size) is not int or world_size < 1:
            raise ValueError("world_size must be a positive integer")
        if sorted(rank for group in self.partitions.groups for rank in group) != list(
            range(world_size)
        ):
            raise ValueError(
                "input partitions must cover every logical rank exactly once"
            )

    @property
    def partition_count(self) -> int:
        return len(self.partitions.groups)

    def partition_index(self, logical_rank: int) -> int:
        members = self.partitions.members(logical_rank)
        return self.partitions.groups.index(members)

    def contributes_loss(self, logical_rank: int) -> bool:
        index = self.partitions.index(logical_rank)
        return self.loss == "partitioned" or index == 0

    def region(
        self, path: tuple[str, ...], shape: Sequence[int], logical_rank: int
    ) -> tuple[slice, ...]:
        index = self.partitions.index(logical_rank)
        for item in self.fields:
            if item.path == tuple(path):
                return item.region(shape, index=index, workers=self.partitions.size)
        raise ValueError(f"input field has no delivery declaration: {path}")

    def owns_output(self, name: str, logical_rank: int) -> bool:
        index = self.partitions.index(logical_rank)
        return self.outputs[name].mode != "leader" or index == 0

    def scheduled_capacity(self, capacities: Sequence[Sequence[int]]) -> int:
        """Sum actual padded buffer capacities once per logical input partition.

        Counts can differ only when a prepared distributed schedule supports it.
        A masked model buffer contributes its full capacity; a cooperating copy
        does not contribute it again. This helper does not admit a schedule.
        """
        if len(capacities) != self.partition_count:
            raise ValueError("scheduled buffers must describe every input partition")
        if any(
            type(size) is not int or size < 0 for group in capacities for size in group
        ):
            raise ValueError("scheduled buffer capacities must be nonnegative integers")
        return sum(size for group in capacities for size in group)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "partitions": {
                "name": self.partitions.name,
                "groups": [list(group) for group in self.partitions.groups],
            },
            "fields": [item.to_dict() for item in self.fields],
            "loss": self.loss,
            "outputs": {name: item.to_dict() for name, item in self.outputs.items()},
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> InputPlacement:
        data = dict(value)
        if type(data.get("version")) is not int or data.pop("version") != 1:
            raise ValueError("unsupported input placement version")
        data["partitions"] = GroupRef(**data["partitions"])
        data["fields"] = tuple(InputFieldPlacement(**item) for item in data["fields"])
        data["outputs"] = {
            name: OutputPlacement(**item) for name, item in data["outputs"].items()
        }
        return cls(**data)
