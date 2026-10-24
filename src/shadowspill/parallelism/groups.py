"""Ordered logical groups; no devices, communicators or framework imports."""

from dataclasses import dataclass


@dataclass(frozen=True)
class GroupRef:
    """A model-facing reference obtained from ``parallelism.group(name)``.

    Membership is expressed in logical ranks. Providers can inspect ``size``
    during construction; execution binds the reference to a live communicator.
    This object never creates one.
    """

    name: str
    groups: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        groups = tuple(tuple(group) for group in self.groups)
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("group name must be nonempty")
        if not groups or any(not group for group in groups):
            raise ValueError(f"{self.name}: groups must be nonempty")
        if len({len(group) for group in groups}) != 1:
            raise ValueError(f"{self.name}: symmetric groups must have equal size")
        ranks = [rank for group in groups for rank in group]
        if any(type(rank) is not int or rank < 0 for rank in ranks):
            raise ValueError(f"{self.name}: logical ranks must be nonnegative integers")
        if len(set(ranks)) != len(ranks):
            raise ValueError(f"{self.name}: a rank belongs to more than one group")
        object.__setattr__(self, "groups", groups)

    @property
    def size(self) -> int:
        return len(self.groups[0])

    def members(self, logical_rank: int) -> tuple[int, ...]:
        for group in self.groups:
            if logical_rank in group:
                return group
        raise ValueError(f"logical rank {logical_rank} is outside {self.name}")

    def index(self, logical_rank: int) -> int:
        return self.members(logical_rank).index(logical_rank)
