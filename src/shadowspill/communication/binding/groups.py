"""Coordinated startup and ordered process groups for saved logical membership."""

import socket
from collections.abc import Mapping, Sequence
from datetime import timedelta
from typing import Any

import torch.distributed as dist

from shadowspill.parallelism import GroupRef

from .ranks import RankAssignment, validate_topology


class GroupBindings:
    """Own only groups created here; the caller owns distributed rendezvous."""

    def __init__(
        self,
        *,
        identity: str,
        groups: Sequence[Sequence[int]],
        rank_map: Mapping[int, int] | None,
        topology: Mapping[str, Sequence[int]],
        backend: str,
        timeout: timedelta = timedelta(seconds=60),
        startup_error: str | None = None,
    ) -> None:
        if not dist.is_initialized():
            raise RuntimeError("initialize worker rendezvous before binding groups")
        self._owned: list[dist.ProcessGroup] = []
        self._groups: dict[tuple[int, ...], dist.ProcessGroup] = {}
        try:
            # CPU validation precedes backend group creation. An invalid local
            # assignment must be reported to peers rather than strand them.
            control = dist.new_group(backend="gloo", timeout=timeout)
            self._control = control
            self._owned.append(control)
            world, worker = dist.get_world_size(), dist.get_rank()
            assignment, error = None, startup_error
            members: tuple[tuple[int, ...], ...] = ()
            try:
                assignment = RankAssignment.resolve(world, rank_map)
                members = tuple(sorted(set(tuple(group) for group in groups)))
                for group in members:
                    assignment.workers(group)
            except (TypeError, ValueError) as failure:
                error = str(failure)
            description = {
                "identity": identity,
                "assignment": None
                if assignment is None
                else assignment.logical_by_worker,
                "groups": members,
                "topology": dict(topology),
                "backend": backend,
            }
            observed: list[Any] = [None] * world
            dist.all_gather_object(
                observed, (description, error, socket.gethostname()), group=control
            )
            failures = [
                f"worker {i}: {item[1]}" for i, item in enumerate(observed) if item[1]
            ]
            if failures:
                raise ValueError("invalid distributed startup: " + "; ".join(failures))
            if any(item[0] != description for item in observed):
                raise ValueError(
                    "workers disagree on what they exchange, the rank map or the groups"
                )
            assert assignment is not None
            validate_topology(topology, assignment, [item[2] for item in observed])
            self.assignment = assignment
            self.logical_rank = assignment.logical_by_worker[worker]
            hosts = [item[2] for item in observed]
            self.topology = dict(topology) or {
                f"node{index}": tuple(
                    sorted(
                        assignment.logical_by_worker[rank]
                        for rank, hostname in enumerate(hosts)
                        if hostname == host
                    )
                )
                for index, host in enumerate(dict.fromkeys(hosts))
            }
            for group in members:
                # PyTorch sorts ranks by default. Shard order must instead match
                # the logical order saved in the Plan, even after reassignment.
                process_group = dist.new_group(
                    ranks=list(assignment.workers(group)),
                    backend=backend,
                    timeout=timeout,
                    sort_ranks=False,
                )
                if self.logical_rank in group:
                    self._owned.append(process_group)
                    self._groups[group] = process_group
                    if dist.get_rank(process_group) != group.index(self.logical_rank):
                        raise RuntimeError(
                            "backend did not preserve logical group order"
                        )
        except BaseException:
            self.close()
            raise

    def resolve(self, reference: GroupRef) -> dist.ProcessGroup:
        return self.for_members(reference.members(self.logical_rank))

    def for_members(self, members: Sequence[int]) -> dist.ProcessGroup:
        try:
            return self._groups[tuple(members)]
        except KeyError as error:
            raise ValueError(
                "this logical group was not bound for the worker"
            ) from error

    def gather(self, value: Any) -> list[Any]:
        """Exchange small control records without involving backend GPU streams."""
        observed: list[Any] = [None] * len(self.assignment.logical_by_worker)
        dist.all_gather_object(observed, value, group=self._control)
        return observed

    def check(self, error: str | None) -> None:
        failures = [
            f"worker {worker}: {item}"
            for worker, item in enumerate(self.gather(error))
            if item is not None
        ]
        if failures:
            raise RuntimeError("distributed operation failed: " + "; ".join(failures))

    def continue_sampling(self, needed: bool, allowed: bool) -> bool:
        votes = self.gather((needed, allowed))
        return any(vote[0] for vote in votes) and all(vote[1] for vote in votes)

    def close(self) -> None:
        for group in reversed(self._owned):
            dist.destroy_process_group(group)
        self._owned.clear()
        self._groups.clear()
