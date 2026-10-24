"""Translate complete worker assignments without changing logical member order."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class RankAssignment:
    # Position is launcher rank; value is the saved Plan's logical rank.
    logical_by_worker: tuple[int, ...]

    @classmethod
    def resolve(
        cls, world_size: int, rank_map: Mapping[int, int] | None
    ) -> "RankAssignment":
        if type(world_size) is not int or world_size < 1:
            raise ValueError("world_size must be a positive integer")
        if rank_map is None and world_size > 1:
            raise ValueError("supply the complete rank_map to every distributed worker")
        values = dict(enumerate(range(world_size))) if rank_map is None else rank_map
        ranks = list(range(world_size))
        if (
            any(type(rank) is not int for rank in (*values.keys(), *values.values()))
            or sorted(values) != ranks
            or sorted(values.values()) != ranks
        ):
            raise ValueError(
                "rank_map must map every launcher rank to exactly one logical rank"
            )
        return cls(tuple(values[worker] for worker in ranks))

    def workers(self, logical_members: Sequence[int]) -> tuple[int, ...]:
        inverse = {
            logical: worker for worker, logical in enumerate(self.logical_by_worker)
        }
        if not logical_members or len(set(logical_members)) != len(logical_members):
            raise ValueError("logical group members must be nonempty and unique")
        if any(
            type(rank) is not int or rank not in inverse for rank in logical_members
        ):
            raise ValueError("logical group names a rank outside this assignment")
        return tuple(inverse[rank] for rank in logical_members)


def validate_topology(
    topology: Mapping[str, Sequence[int]],
    assignment: RankAssignment,
    hostnames: Sequence[str],
) -> None:
    """Compare node membership, not user-chosen node labels, with physical hosts."""
    if not topology:
        return
    world = len(assignment.logical_by_worker)
    ranks = [rank for members in topology.values() for rank in members]
    if sorted(ranks) != list(range(world)) or any(
        not group for group in topology.values()
    ):
        raise ValueError("topology must contain exactly the declared logical ranks")
    if len(hostnames) != world:
        raise ValueError("startup must report a hostname for every worker")
    nodes = []
    for members in topology.values():
        hosts = {hostnames[worker] for worker in assignment.workers(members)}
        if len(hosts) != 1:
            raise ValueError("rank_map splits a planned node across physical hosts")
        nodes.append(next(iter(hosts)))
    if len(set(nodes)) != len(nodes):
        raise ValueError("rank_map places distinct planned nodes on the same host")
