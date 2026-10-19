"""torchrun CPU canary: shard-only setup preserves each group's leader values."""

from __future__ import annotations

import json
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist

from shadowspill.ir import TransferPhase, TransferSpec
from shadowspill.pytorch.distributed._backing import TransferSharding
from shadowspill.pytorch.distributed._backing_storage import ShardedStorage
from shadowspill.pytorch.state import initialization


def verify_lazy_publication(rank, root, storage, value, expected, group):
    declaration = torch.empty_like(value)
    item = SimpleNamespace(
        unbacked=False, size_bytes=value.numel(), sharding=storage
    )
    registry = SimpleNamespace(
        storage={declaration.untyped_storage()._cdata: item}.get
    )
    received, calls = [], []

    def publish(runtime, record, full, *, synchronize):
        assert record is item and synchronize
        received.append(storage.synchronize_owned(full))

    def fill(scratch):
        assert rank == root
        calls.append(True)
        scratch.copy_(value)

    with (
        patch.object(initialization, "registry_for", return_value=registry),
        patch.object(initialization, "_write_storage_bytes", publish),
        initialization.pool_values(object()),
    ):
        initialization.initialize_storage_(declaration, fill)
        if value.numel() == 17:
            def broken(scratch):
                raise OSError("unreadable checkpoint")

            try:
                initialization.initialize_storage_(declaration, broken)
            except RuntimeError as error:
                assert "unreadable checkpoint" in str(error)
            else:
                raise AssertionError("failed source read was not propagated")
    assert calls == ([True] if rank == root else [])
    assert len(received) == 1
    actual = torch.empty_like(expected)
    dist.all_gather_into_tensor(actual, received[0], group=group)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def main():
    dist.init_process_group("gloo", timeout=timedelta(seconds=90))
    try:
        rank, world = dist.get_rank(), dist.get_world_size()
        sizes = sorted({world, max(1, world // 2), 1}, reverse=True)
        for size in sizes:
            if world % size:
                continue
            groups = [
                dist.new_group(list(range(start, start + size)), backend="gloo")
                for start in range(0, world, size)
            ]
            group = groups[rank // size]
            root = rank // size * size
            for length in (1, 17, 65539):
                shard_bytes = ((length + size * 16 - 1) // (size * 16)) * 16
                storage = ShardedStorage(
                    TransferSharding(group, host_group=group),
                    group,
                    "test",
                    shard_bytes * size,
                    shard_bytes,
                    rank % size * shard_bytes,
                    TransferSpec(
                        shard_bytes,
                        (TransferPhase("copy", bytes=shard_bytes),),
                        (TransferPhase("copy", bytes=shard_bytes),),
                    ),
                )
                # Nonleaders intentionally disagree. Publication must retain
                # only the leader's bytes, including a deterministic padded tail.
                value = ((torch.arange(length) + rank * 17) % 251).to(torch.uint8)
                shard = storage.synchronize_owned(value)
                actual = torch.empty(shard_bytes * size, dtype=torch.uint8)
                dist.all_gather_into_tensor(actual, shard, group=group)
                expected = torch.zeros_like(actual)
                expected[:length] = ((torch.arange(length) + root * 17) % 251).to(
                    torch.uint8
                )
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                verify_lazy_publication(rank, root, storage, value, expected, group)
            print(json.dumps({"rank": rank, "shard_group_size": size, "passed": True}))
            dist.destroy_process_group(group)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
