"""Independent storage choices for weights, gradients and optimizer tensors."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Literal

type StorageMode = Literal["sharded", "replicated"]
type Component = Literal["weights", "gradients", "optimizer"]

_COMPONENT_GROUPS: dict[Component, str] = {
    "weights": "weight_group",
    "gradients": "gradient_group",
    "optimizer": "optimizer_group",
}


@dataclass(frozen=True)
class ParameterStorage:
    """Where a parameter's weights, gradients and optimizer state are stored.

    Each component is ``"sharded"``, each rank of a group storing an equal
    part, or ``"replicated"``, each storing all of it. A sharded component's
    storage group holds only ranks that compute with the same slice of the
    parameter. Unset fields come from the default storage of the
    `Parallelism`, never from another matching selector: a mode or
    ``min_shard_bytes`` left None takes the default's value, and a component
    is stored over its own group (such as ``weight_group``), else ``group``,
    else the default's group for that component, else the default's
    ``group``. Master weights follow the optimizer state; scalar optimizer
    controls are never sharded. Buffers are stored whole on every rank: they
    have no gradient or optimizer state, and a Program that changes one
    changes each rank's own copy.

    Parameters
    ----------
    group : str, optional
        The group collection the components are stored over.
    weights : str, optional
        ``"sharded"`` or ``"replicated"``.
    gradients : str, optional
        ``"sharded"`` or ``"replicated"``.
    optimizer : str, optional
        ``"sharded"`` or ``"replicated"``, for optimizer state and masters.
    weight_group : str, optional
        Group collection for the weights, instead of ``group``.
    gradient_group : str, optional
        Group collection for the gradients, instead of ``group``.
    optimizer_group : str, optional
        Group collection for the optimizer state, instead of ``group``.
    min_shard_bytes : int, optional
        Weight storage size, in bytes, below which a parameter keeps every
        component replicated: its shard would save little memory and cost a
        collective at every fetch. Zero, or unset everywhere, shards every
        parameter.

    Raises
    ------
    ValueError
        If a mode is neither ``"sharded"`` nor ``"replicated"``, a group name
        is empty, or ``min_shard_bytes`` is not a nonnegative integer.
    """

    group: str | None = None
    weights: StorageMode | None = None
    gradients: StorageMode | None = None
    optimizer: StorageMode | None = None
    weight_group: str | None = None
    gradient_group: str | None = None
    optimizer_group: str | None = None
    min_shard_bytes: int | None = None

    def __post_init__(self) -> None:
        if self.min_shard_bytes is not None and (
            type(self.min_shard_bytes) is not int or self.min_shard_bytes < 0
        ):
            raise ValueError("min_shard_bytes must be a nonnegative integer")
        for name in ("weights", "gradients", "optimizer"):
            if getattr(self, name) not in {None, "sharded", "replicated"}:
                raise ValueError(f"{name} must be sharded or replicated")
        for name in ("group", "weight_group", "gradient_group", "optimizer_group"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{name} must be a nonempty group name")

    def inherit(self, default: ParameterStorage) -> ParameterStorage:
        """Return this storage with each unset field taken from ``default``.

        A component group stays unset when this storage sets ``group``, so
        the component is stored over that ``group``.

        Parameters
        ----------
        default : ParameterStorage
            The storage to inherit from.

        Returns
        -------
        ParameterStorage
            The combined storage.
        """
        values = {
            item.name: getattr(self, item.name)
            if getattr(self, item.name) is not None
            else getattr(default, item.name)
            for item in fields(self)
        }
        if self.group is not None:
            for name in _COMPONENT_GROUPS.values():
                values[name] = getattr(self, name)
        return ParameterStorage(**values)

    def group_for(self, component: Component) -> str:
        """Return the group collection a component is stored over.

        Parameters
        ----------
        component : str
            ``"weights"``, ``"gradients"`` or ``"optimizer"``.

        Returns
        -------
        str
            The component's group, or else ``group``.

        Raises
        ------
        ValueError
            If the component's mode or group is unset.
        """
        group = getattr(self, _COMPONENT_GROUPS[component]) or self.group
        if group is None or getattr(self, component) is None:
            raise ValueError("default parameter storage must specify every component")
        return str(group)

    def to_dict(self) -> dict[str, str | int | None]:
        """Return the storage as plain data.

        Returns
        -------
        dict
            Every field, by name.
        """
        return asdict(self)
