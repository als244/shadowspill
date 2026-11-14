"""Logical persistent state identities, independent of Program-local object IDs.

The same inventory describes operation sharing and checkpoint conversion. Saved
tensor geometry and generic parameter ownership remain authoritative; describing
them here neither constructs tensors nor chooses a new distribution strategy.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from ._values import digest, freeze, nonnegative_int, plain

type StateRole = Literal[
    "parameter", "buffer", "master", "gradient", "optimizer", "rng", "registered"
]
_ROLES = {"parameter", "buffer", "master", "gradient", "optimizer", "rng", "registered"}


def _name(value: object, name: str) -> None:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError(f"{name} must be a nonempty name")


@dataclass(frozen=True, order=True)
class StateId:
    """A role, registered name and representation component, without rank/shape.

    Components are structural, not dot-joined: a name containing a dot cannot
    collide with a different name's representation component. Optimizer names
    remain opaque; their parameter owner is explicit in the tensor descriptor.
    """

    role: StateRole
    name: str
    component_path: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.role not in _ROLES:
            raise ValueError(f"unknown state role: {self.role}")
        _name(self.name, "state name")
        if not isinstance(self.component_path, (list, tuple)):
            raise TypeError("state component path must be a sequence")
        for part in self.component_path:
            _name(part, "state component")
        object.__setattr__(self, "component_path", tuple(self.component_path))

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "name": self.name,
            "component_path": list(self.component_path),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> StateId:
        return cls(**value)

    @property
    def key(self) -> str:
        return "state-v1-" + digest(self.to_dict())


@dataclass(frozen=True)
class StateTensor:
    identity: StateId
    geometry: Mapping[str, Any]
    parameter_name: str | None = None
    ownership: Mapping[str, Any] = field(default_factory=dict)
    checkpoint: bool = True
    optimizer_type: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.identity, StateId):
            raise TypeError("state identity must be StateId")
        geometry = dict(self.geometry)
        if not {"dtype", "shape", "stride"}.issubset(geometry):
            raise ValueError("state tensor requires dtype, shape and stride")
        _name(geometry["dtype"], "state dtype")
        for key in ("shape", "stride"):
            if not isinstance(geometry[key], (list, tuple)):
                raise TypeError(f"state {key} must be a sequence")
            for value in geometry[key]:
                nonnegative_int(value, key)
        if len(geometry["shape"]) != len(geometry["stride"]):
            raise ValueError("state tensor rank and stride differ")
        geometry.setdefault("storage_offset", 0)
        nonnegative_int(geometry["storage_offset"], "storage_offset")
        if self.parameter_name is not None:
            _name(self.parameter_name, "parameter owner")
        if self.optimizer_type is not None:
            _name(self.optimizer_type, "optimizer type")
        if type(self.checkpoint) is not bool:
            raise TypeError("checkpoint must be boolean")
        object.__setattr__(self, "geometry", freeze(geometry))
        object.__setattr__(self, "ownership", freeze(dict(self.ownership)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity": self.identity.to_dict(),
            "geometry": plain(self.geometry),
            "parameter_name": self.parameter_name,
            "ownership": plain(self.ownership),
            "checkpoint": self.checkpoint,
            "optimizer_type": self.optimizer_type,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> StateTensor:
        data = dict(value)
        data["identity"] = StateId.from_dict(data["identity"])
        return cls(**data)


@dataclass(frozen=True)
class StateBinding:
    identity: StateId
    object_id: str | None = None
    alias_group_id: str | None = None
    host_binding: str | None = None
    storage: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.identity, StateId):
            raise TypeError("state identity must be StateId")
        if self.host_binding is not None:
            _name(self.host_binding, "host binding")
            if self.object_id is not None or self.alias_group_id is not None:
                raise ValueError("host state cannot also name a physical object")
            if self.storage:
                raise ValueError("host state does not describe a pool allocation")
        else:
            _name(self.object_id, "state object ID")
            _name(self.alias_group_id, "state alias ID")
            keys = {
                "device_id",
                "object_offset_bytes",
                "object_size_bytes",
                "size_bytes",
                "spill_size_bytes",
                "transfer",
            }
            if set(self.storage) != keys:
                raise ValueError("physical state requires its complete storage layout")
            _name(self.storage["device_id"], "state device")
            for key in keys - {"device_id", "transfer"}:
                nonnegative_int(self.storage[key], key)
            if (
                self.storage["object_offset_bytes"] + self.storage["object_size_bytes"]
                > self.storage["size_bytes"]
            ):
                raise ValueError("state view exceeds its storage")
        object.__setattr__(self, "storage", freeze(dict(self.storage)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity": self.identity.to_dict(),
            "object_id": self.object_id,
            "alias_group_id": self.alias_group_id,
            "host_binding": self.host_binding,
            "storage": plain(self.storage),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> StateBinding:
        data = dict(value)
        data["identity"] = StateId.from_dict(data["identity"])
        return cls(**data)


@dataclass(frozen=True)
class StateInventory:
    logical_rank: int
    tensors: tuple[StateTensor, ...]
    bindings: tuple[StateBinding, ...]

    def __post_init__(self) -> None:
        nonnegative_int(self.logical_rank, "logical rank")
        object.__setattr__(self, "tensors", tuple(self.tensors))
        object.__setattr__(self, "bindings", tuple(self.bindings))
        if any(not isinstance(t, StateTensor) for t in self.tensors) or any(
            not isinstance(b, StateBinding) for b in self.bindings
        ):
            raise TypeError("state inventory requires tensor and binding records")
        tensors = {t.identity for t in self.tensors}
        bindings = {b.identity for b in self.bindings}
        if len(tensors) != len(self.tensors) or len(bindings) != len(self.bindings):
            raise ValueError("duplicate state identity")
        if tensors != bindings:
            raise ValueError("every logical state tensor requires exactly one binding")

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "logical_rank": self.logical_rank,
            "tensors": [t.to_dict() for t in self.tensors],
            "bindings": [b.to_dict() for b in self.bindings],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> StateInventory:
        if (
            set(value) != {"version", "logical_rank", "tensors", "bindings"}
            or type(value["version"]) is not int
            or value["version"] != 1
        ):
            raise ValueError("unsupported logical state inventory schema")
        return cls(
            value["logical_rank"],
            tuple(StateTensor.from_dict(t) for t in value["tensors"]),
            tuple(StateBinding.from_dict(b) for b in value["bindings"]),
        )

    def shared_aliases(self, other: StateInventory) -> dict[str, str]:
        """Match common state on one worker; never alias physical replicas.

        Only states present in both inventories are mapped. Joint admission owns
        additional state storage; checkpoint conversion may change layouts but
        direct operation sharing must preserve geometry, ownership and ties.
        """
        if self.logical_rank != other.logical_rank:
            raise ValueError("different logical workers cannot share physical state")
        tensors = {t.identity: t for t in self.tensors}
        bindings = {b.identity: b for b in self.bindings}
        other_bindings = {b.identity: b for b in other.bindings}
        forward: dict[str, str] = {}
        reverse: dict[str, str] = {}
        for tensor in other.tensors:
            if tensor.identity not in tensors:
                continue
            if tensor != tensors[tensor.identity]:
                raise ValueError(
                    f"shared state layout or ownership differs: {tensor.identity}"
                )
            left, right = bindings[tensor.identity], other_bindings[tensor.identity]
            if (left.host_binding is None) != (right.host_binding is None):
                raise ValueError("state changes between host and physical storage")
            if left.storage != right.storage:
                raise ValueError(
                    f"shared state physical storage differs: {tensor.identity}"
                )
            if left.alias_group_id is None or right.alias_group_id is None:
                continue  # Host states are bound by logical identity, not pool aliases.
            src, dst = left.alias_group_id, right.alias_group_id
            if (
                forward.setdefault(dst, src) != src
                or reverse.setdefault(src, dst) != dst
            ):
                raise ValueError(f"shared state aliasing differs: {tensor.identity}")
        return forward
