"""Bind saved weight/gradient shards to the existing compound transfer path."""

from typing import TYPE_CHECKING

import torch

from shadowspill.parallelism import GroupRef

if TYPE_CHECKING:
    from shadowspill.artifacts.program import Program
    from shadowspill.communication.binding.groups import GroupBindings


class ShardedTransfers:
    """Own descriptor references; registered runtime objects retain their own."""

    def __init__(self, program: "Program", groups: "GroupBindings", device: int):
        self.descriptors: dict[str, int] = {}
        cache: dict[tuple[str, int, int, torch.dtype | None, str | None], int] = {}
        try:
            shards = {
                **program.bindings.get("weight_shards", {}),
                **program.bindings.get("gradient_shards", {}),
            }
            for alias, description in shards.items():
                group = groups.resolve(GroupRef(alias, description["groups"]))
                gradient = alias in program.bindings.get("gradient_shards", {})
                dtype = (
                    getattr(torch, description["dtype"].removeprefix("torch."))
                    if gradient
                    else None
                )
                replicas = description.get("replica_groups", ())
                replica_name = (
                    groups.resolve(GroupRef(alias + ".replicas", replicas)).group_name
                    if replicas
                    else None
                )
                key = (
                    group.group_name,
                    description["padded_bytes"],
                    description["logical_bytes"],
                    dtype,
                    replica_name,
                )
                if key not in cache:
                    cache[key] = int(
                        torch.ops.shadowspill._create_sharded_transfer(
                            group.group_name,
                            device,
                            description["padded_bytes"],
                            dtype,
                            replica_name,
                            description["logical_bytes"],
                        )
                    )
                self.descriptors[alias] = cache[key]
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        for descriptor in set(self.descriptors.values()):
            torch.ops.shadowspill._close_sharded_transfer(descriptor)
        self.descriptors.clear()
