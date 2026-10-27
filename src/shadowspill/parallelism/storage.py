"""Independent storage choices for weights, gradients and optimizer tensors."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Literal

type StorageMode = Literal["sharded", "replicated"]
type Component = Literal["weights", "gradients", "optimizer"]


@dataclass(frozen=True)
class ParameterStorage:
    """Omitted fields inherit from the default, never another matching pattern.

    ``group`` is the common storage group. Component-specific groups override
    it. Master weights follow optimizer ownership; scalar optimizer controls do
    not acquire a parameter-shaped shard just because moments do.

    ``min_shard_bytes`` is opt-in (zero by default). Parameters whose unpadded
    physical weight storage is smaller keep all components replicated. Frozen
    tensor-wrapper payloads/scales are also checked separately during lowering.
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
        return ParameterStorage(
            **{
                item.name: getattr(self, item.name)
                if getattr(self, item.name) is not None
                else getattr(default, item.name)
                for item in fields(self)
            }
        )

    def group_for(self, component: Component) -> str:
        override = {
            "weights": "weight_group",
            "gradients": "gradient_group",
            "optimizer": "optimizer_group",
        }[component]
        group = getattr(self, override) or self.group
        if group is None or getattr(self, component) is None:
            raise ValueError("default parameter storage must specify every component")
        return str(group)

    def to_dict(self) -> dict[str, str | int | None]:
        return asdict(self)
