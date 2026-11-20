"""Ordered logical groups; no devices, communicators or framework imports."""

from dataclasses import dataclass


@dataclass(frozen=True)
class GroupRef:
    """Named groups of logical ranks, such as `Parallelism.group` returns.

    Every group has the same number of ranks, and no rank is in two. It names
    ranks only and creates no communicator: a worker binds it to its own
    group's communicator when it runs, so code that builds a model can read
    `size` beforehand.

    Parameters
    ----------
    name : str
        The groups' name. Nonempty.
    groups : sequence of sequence of int
        The groups, each of nonnegative logical ranks in order: a rank's
        position is its rank in its group's communicator. Stored as tuples.

    Raises
    ------
    ValueError
        If ``name`` or a group is empty, there are no groups, groups differ
        in size, a rank is not a nonnegative integer, or a rank is in two
        groups.
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
        """Ranks in each group."""
        return len(self.groups[0])

    def members(self, logical_rank: int) -> tuple[int, ...]:
        """Return the group that holds a rank.

        Parameters
        ----------
        logical_rank : int
            The rank.

        Returns
        -------
        tuple of int
            The group's ranks, in order.

        Raises
        ------
        ValueError
            If no group holds the rank.
        """
        for group in self.groups:
            if logical_rank in group:
                return group
        raise ValueError(f"logical rank {logical_rank} is outside {self.name}")

    def index(self, logical_rank: int) -> int:
        """Return a rank's position in its group.

        Parameters
        ----------
        logical_rank : int
            The rank.

        Returns
        -------
        int
            The position, from zero.

        Raises
        ------
        ValueError
            If no group holds the rank.
        """
        return self.members(logical_rank).index(logical_rank)
