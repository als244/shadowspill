"""One complete-model replica, repeated without changing its sharding factor."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from .groups import GroupRef
from .placements import (
    Placement,
    Replicated,
    Sharded,
    placement_from_record,
    placement_record,
)
from .storage import ParameterStorage, StorageMode


@dataclass(frozen=True)
class Parallelism:
    """Logical ownership only; compatible model layers implement communication.

    Named groups partition the replica's ranks, in caller-specified order.
    All replicas use the same local model shapes. Distinct graphs per rank and
    noncontiguous placement maps are outside the initial implementation.
    """

    ranks_per_replica: int = 1
    num_replicas: int = 1
    groups: Mapping[str, Sequence[Sequence[int]]] = field(default_factory=dict)
    default_parameter_placement: Placement = field(
        default_factory=lambda: Replicated("all_ranks")
    )
    parameter_placements: Mapping[str, Placement] = field(default_factory=dict)
    default_parameter_storage: ParameterStorage = field(
        default_factory=lambda: ParameterStorage(
            group="all_ranks",
            weights="sharded",
            gradients="sharded",
            optimizer="sharded",
        )
    )
    parameter_storage: Mapping[str, ParameterStorage] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("ranks_per_replica", "num_replicas"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        groups = self.groups or {"all_ranks": (tuple(range(self.ranks_per_replica)),)}
        frozen = {}
        for name, members in groups.items():
            reference = GroupRef(name, tuple(tuple(group) for group in members))
            if sorted(rank for group in reference.groups for rank in group) != list(
                range(self.ranks_per_replica)
            ):
                raise ValueError(f"{name}: groups must cover exactly one replica")
            frozen[name] = reference.groups
        object.__setattr__(self, "groups", MappingProxyType(frozen))
        for name, expected in (
            ("parameter_placements", (Replicated, Sharded)),
            ("parameter_storage", (ParameterStorage,)),
        ):
            values = dict(getattr(self, name))
            if any(not isinstance(key, str) or not key for key in values):
                raise ValueError(f"{name}: selectors must be nonempty parameter names")
            if any(not isinstance(value, expected) for value in values.values()):
                raise TypeError(f"{name}: invalid declaration")
            object.__setattr__(self, name, MappingProxyType(values))
        if not isinstance(self.default_parameter_placement, (Replicated, Sharded)):
            raise TypeError("default_parameter_placement must describe a placement")
        if not isinstance(self.default_parameter_storage, ParameterStorage):
            raise TypeError("default_parameter_storage must be ParameterStorage")
        for placement in (
            self.default_parameter_placement,
            *self.parameter_placements.values(),
        ):
            self.group(placement.group)
        for storage in (ParameterStorage(), *self.parameter_storage.values()):
            storage = storage.inherit(self.default_parameter_storage)
            for component in ("weights", "gradients", "optimizer"):
                self.group(storage.group_for(component))

    @classmethod
    def dp(
        cls,
        *,
        ranks_per_replica: int = 1,
        num_replicas: int = 1,
        weights: StorageMode = "sharded",
        gradients: StorageMode = "sharded",
        optimizer: StorageMode = "sharded",
    ) -> Parallelism:
        return cls(
            ranks_per_replica=ranks_per_replica,
            num_replicas=num_replicas,
            default_parameter_storage=ParameterStorage(
                group="all_ranks",
                weights=weights,
                gradients=gradients,
                optimizer=optimizer,
            ),
        )

    @property
    def world_size(self) -> int:
        return self.ranks_per_replica * self.num_replicas

    def group(self, name: str) -> GroupRef:
        if name not in self.groups:
            raise ValueError(f"unknown logical group: {name}")
        return GroupRef(
            name,
            tuple(
                tuple(rank + replica * self.ranks_per_replica for rank in group)
                for replica in range(self.num_replicas)
                for group in self.groups[name]
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "ranks_per_replica": self.ranks_per_replica,
            "num_replicas": self.num_replicas,
            "groups": {
                name: [list(group) for group in groups]
                for name, groups in self.groups.items()
            },
            "default_parameter_placement": placement_record(
                self.default_parameter_placement
            ),
            "parameter_placements": {
                name: placement_record(value)
                for name, value in self.parameter_placements.items()
            },
            "default_parameter_storage": self.default_parameter_storage.to_dict(),
            "parameter_storage": {
                name: value.to_dict() for name, value in self.parameter_storage.items()
            },
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Parallelism:
        data = dict(value)
        data["default_parameter_placement"] = placement_from_record(
            data["default_parameter_placement"]
        )
        data["parameter_placements"] = {
            name: placement_from_record(item)
            for name, item in data["parameter_placements"].items()
        }
        data["default_parameter_storage"] = ParameterStorage(
            **data["default_parameter_storage"]
        )
        data["parameter_storage"] = {
            name: ParameterStorage(**item)
            for name, item in data["parameter_storage"].items()
        }
        return cls(**data)
