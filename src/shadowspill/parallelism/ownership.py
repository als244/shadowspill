"""Resolved logical parameter copies and flat element ownership."""

from dataclasses import asdict, dataclass
from math import prod
from typing import Any

from .groups import GroupRef
from .storage import Component


@dataclass(frozen=True)
class StorageOwnership:
    """Parameter-element ranges; dtype-specific byte counts are derived later."""

    elements: int
    groups: tuple[tuple[int, ...], ...]

    @property
    def capacity(self) -> int:
        return (self.elements + len(self.groups[0]) - 1) // len(self.groups[0])

    def region(self, logical_rank: int) -> tuple[int, int]:
        index = GroupRef("storage", self.groups).index(logical_rank)
        start = self.capacity * index
        return min(start, self.elements), max(
            0, min(self.capacity, self.elements - start)
        )

    def matching_owners(self, logical_rank: int) -> tuple[int, ...]:
        index = GroupRef("storage", self.groups).index(logical_rank)
        return tuple(group[index] for group in self.groups)


@dataclass(frozen=True)
class ParameterOwnership:
    name: str
    names: tuple[str, ...]
    local_shape: tuple[int, ...]
    global_shape: tuple[int, ...]
    trainable: bool
    copy_groups: tuple[tuple[int, ...], ...]
    weights: StorageOwnership
    gradients: StorageOwnership
    optimizer: StorageOwnership

    def copies(self, logical_rank: int) -> tuple[int, ...]:
        return GroupRef(self.name, self.copy_groups).members(logical_rank)

    def storage(self, component: Component, logical_rank: int) -> StorageOwnership:
        """Restrict matching stored fragments to copies of this logical shard."""
        owned = getattr(self, component)
        copies = self.copies(logical_rank)
        return StorageOwnership(
            prod(self.local_shape),
            tuple(group for group in owned.groups if group[0] in copies),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
