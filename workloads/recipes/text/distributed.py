"""Caller-owned groups for the text preset's data-parallel setup."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import timedelta

import torch.distributed as dist

from shadowspill.pytorch import Distributed, TransferSharding


def data_parallel_groups(
    *, device, shard_size: int | None, shard_gradients: bool, timeout: float
):
    world = dist.get_world_size()
    if shard_size is not None and world % shard_size:
        raise ValueError("--transfer-shard-size must divide the data-parallel world")
    owned = []

    def create(ranks, backend):
        group = dist.new_group(
            ranks=list(ranks),
            backend=backend,
            timeout=timedelta(seconds=timeout),
            **({"device_id": device} if backend == "nccl" else {}),
        )
        if group != dist.GroupMember.NON_GROUP_MEMBER:
            owned.append(group)
        return group

    try:
        participants = create(range(world), "nccl")
        sharding = None
        if shard_size is not None:
            if shard_size == world:
                shards, host, replicas = participants, dist.group.WORLD, None
            else:
                rank = dist.get_rank()
                shards = host = replicas = None
                for first in range(0, world, shard_size):
                    members = range(first, first + shard_size)
                    candidate = create(members, "nccl")
                    cpu = create(members, "gloo")
                    if rank in members:
                        shards, host = candidate, cpu
                for position in range(shard_size):
                    members = range(position, world, shard_size)
                    candidate = create(members, "nccl")
                    if rank in members:
                        replicas = candidate
            sharding = TransferSharding(shards, replicas, host)
        specification = Distributed(
            participants,
            transfer_sharding=sharding,
            shard_gradients=shard_gradients,
            timeout=timeout,
        )
    except BaseException:
        for group in reversed(owned):
            dist.destroy_process_group(group)
        raise

    @contextmanager
    def resources():
        try:
            yield
        finally:
            for group in reversed(owned):
                dist.destroy_process_group(group)

    return specification, resources
