"""One complete-model replica, repeated without changing its sharding factor."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from types import MappingProxyType
from typing import Any

from shadowspill_client.parallelism import MIN_SHARD_BYTES
from shadowspill_client.placement.groups import GroupRef
from shadowspill_client.placement.inputs import InputPlacement

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
    """How a model's parameters are owned across ranks.

    The builders take it as ``parallelism``. It declares ownership only: it
    creates no process groups and divides no model code, and the model's
    layers implement any communication. A replica is ``ranks_per_replica``
    ranks that together hold the model state; ``num_replicas`` copies of that
    layout run at successive rank offsets without changing how finely state
    is sharded. Every replica has the same local model shapes. Each rank
    builds its own Program; the ranks need only agree on what they exchange.
    Parameter selectors are module-qualified names or wildcard patterns, such
    as ``"layers.*.experts.*"``; an exact name wins over a pattern.

    Parameters
    ----------
    ranks_per_replica : int, default 1
        Ranks in one replica. Positive.
    num_replicas : int, default 1
        Copies of the replica layout. Positive.
    groups : mapping of str to sequence of sequence of int, default {}
        Named collections of rank groups within one replica, each collection
        covering the replica's ranks exactly once, in the caller's order. By
        default one collection, ``"all_ranks"``, of one group of every rank.
    default_parameter_placement : Replicated or Sharded, optional
        How a parameter no selector matches is computed with. By default
        ``Replicated()``.
    parameter_placements : mapping of str to Replicated or Sharded, default {}
        Placements by parameter selector.
    default_parameter_storage : ParameterStorage, optional
        How a parameter no selector matches is stored; it sets every
        component's mode and group. By default weights, gradients and
        optimizer state sharded over ``"all_ranks"``.
    parameter_storage : mapping of str to ParameterStorage, default {}
        Storage by parameter selector; unset fields inherit the default, as
        `ParameterStorage` describes.
    inputs : InputPlacement, optional
        How each batch's inputs, loss and outputs divide across ranks. By
        default the caller supplies each rank's inputs.

    Raises
    ------
    ValueError
        If a count is not a positive integer, a group collection does not
        cover a replica exactly once, a selector is empty, a placement or
        storage names an unknown group, or the default storage leaves a
        component unset.
    TypeError
        If a placement, storage or ``inputs`` has another type.
    """

    ranks_per_replica: int = 1
    num_replicas: int = 1
    groups: Mapping[str, Sequence[Sequence[int]]] = field(default_factory=dict)
    default_parameter_placement: Placement = field(default_factory=Replicated)
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
    inputs: InputPlacement | None = None

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
            if isinstance(placement, Sharded):
                self.group(placement.group)
        for storage in (ParameterStorage(), *self.parameter_storage.values()):
            storage = storage.inherit(self.default_parameter_storage)
            for component in ("weights", "gradients", "optimizer"):
                self.group(storage.group_for(component))
        if self.inputs is not None:
            if not isinstance(self.inputs, InputPlacement):
                raise TypeError("inputs must be InputPlacement")
            self.inputs.validate(self.world_size)

    @classmethod
    def dp(
        cls,
        *,
        ranks_per_replica: int = 1,
        num_replicas: int = 1,
        weights: StorageMode = "sharded",
        gradients: StorageMode = "sharded",
        optimizer: StorageMode = "sharded",
        min_shard_bytes: int = MIN_SHARD_BYTES,
        inputs: InputPlacement | None = None,
    ) -> Parallelism:
        """Declare data parallelism: every rank computes with whole parameters.

        Parameters
        ----------
        ranks_per_replica : int, default 1
            Ranks that together store one copy of the model state.
        num_replicas : int, default 1
            Copies of that layout.
        weights : str, default "sharded"
            How weights are stored over a replica's ranks: ``"sharded"`` or
            ``"replicated"``.
        gradients : str, default "sharded"
            How gradients are stored, likewise.
        optimizer : str, default "sharded"
            How optimizer state is stored, likewise.
        min_shard_bytes : int, default MIN_SHARD_BYTES
            Weight storage size, in bytes, below which a parameter keeps every
            component replicated; 2 MiB by default. Zero shards every
            parameter.
        inputs : InputPlacement, optional
            How each batch's inputs divide across ranks.

        Returns
        -------
        Parallelism
            The declaration, with one group collection, ``"all_ranks"``, of
            each replica's ranks.
        """
        return cls(
            ranks_per_replica=ranks_per_replica,
            num_replicas=num_replicas,
            inputs=inputs,
            default_parameter_storage=ParameterStorage(
                group="all_ranks",
                weights=weights,
                gradients=gradients,
                optimizer=optimizer,
                min_shard_bytes=min_shard_bytes,
            ),
        )

    @property
    def world_size(self) -> int:
        """Total ranks: ``ranks_per_replica * num_replicas``."""
        return self.ranks_per_replica * self.num_replicas

    def group(self, name: str) -> GroupRef:
        """Return a group collection across every replica.

        Parameters
        ----------
        name : str
            A key of ``groups``.

        Returns
        -------
        GroupRef
            The collection's groups, repeated for each replica at successive
            rank offsets.

        Raises
        ------
        ValueError
            If no collection has that name.
        """
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
        """Return the declaration as plain data.

        Returns
        -------
        dict
            Every field as plain records, with placements tagged by ``kind``;
            ``inputs`` only when set.
        """
        return {
            **({"inputs": self.inputs.to_dict()} if self.inputs is not None else {}),
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
        """Rebuild a declaration from a `to_dict` record.

        Parameters
        ----------
        value : mapping
            The declaration, in `to_dict` form.

        Returns
        -------
        Parallelism
            The declaration.

        Raises
        ------
        ValueError
            If a required key is missing or a key is unknown, naming the key;
            if a placement kind is unknown; or if the declaration is invalid.
        """
        data = dict(value)
        for key in _REQUIRED_KEYS:
            if key not in data:
                raise ValueError(f"parallelism: missing key {key!r}")
        for key in data:
            if key not in _REQUIRED_KEYS + _OPTIONAL_KEYS:
                raise ValueError(f"parallelism: unknown key {key!r}")
        if "inputs" in data:
            data["inputs"] = InputPlacement.from_dict(data["inputs"])
        data["default_parameter_placement"] = placement_from_record(
            data["default_parameter_placement"], "default_parameter_placement"
        )
        data["parameter_placements"] = {
            name: placement_from_record(item, f"parameter_placements[{name!r}]")
            for name, item in data["parameter_placements"].items()
        }
        data["default_parameter_storage"] = _storage_from_record(
            data["default_parameter_storage"], "default_parameter_storage"
        )
        data["parameter_storage"] = {
            name: _storage_from_record(item, f"parameter_storage[{name!r}]")
            for name, item in data["parameter_storage"].items()
        }
        return cls(**data)


_REQUIRED_KEYS = (
    "default_parameter_placement",
    "parameter_placements",
    "default_parameter_storage",
    "parameter_storage",
)
_OPTIONAL_KEYS = ("ranks_per_replica", "num_replicas", "groups", "inputs")


def _storage_from_record(value: Mapping[str, Any], where: str) -> ParameterStorage:
    keys = [item.name for item in fields(ParameterStorage)]
    for key in value:
        if key not in keys:
            raise ValueError(f"{where}: unknown key {key!r}")
    return ParameterStorage(**value)
