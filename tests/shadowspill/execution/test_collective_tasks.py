"""Packed task bindings preserve logical order and join the caller stream."""

import time
from datetime import timedelta

import pytest


def _worker(worker, rendezvous, cuda, world_size):
    import torch
    import torch.distributed as dist

    from shadowspill.communication.binding.groups import GroupBindings
    from shadowspill.execution.distributed.collectives import CollectiveTask

    device = torch.device("cuda", worker) if cuda else torch.device("cpu")
    if cuda:
        torch.cuda.set_device(device)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=worker,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    bindings = None
    try:
        mapping = {0: 1, 1: 0} if world_size == 2 else {0: 2, 1: 0, 2: 3, 3: 1}
        groups = ((0, 1),) if world_size == 2 else ((3, 0), (2, 1))
        bindings = GroupBindings(
            identity="collective-tasks",
            groups=groups,
            rank_map=mapping,
            topology={},
            backend="nccl" if cuda else "gloo",
        )
        logical = mapping[worker]
        members = next(group for group in groups if logical in group)
        for operation, inplace in (
            ("all_gather", False),
            ("all_gather", True),
            ("reduce_scatter", False),
            ("all_reduce", True),
        ):
            task = CollectiveTask(
                dict(
                    operation=operation,
                    inplace=inplace,
                    groups=groups,
                    tensor_bytes=64,
                    dtype="torch.float32",
                ),
                bindings,
            )
            for iteration in range(3):
                stream = torch.cuda.Stream() if cuda else None
                with torch.cuda.stream(stream) if cuda else _cpu_context():
                    # Delayed producer and immediate consumer exercise the
                    # backend's caller-stream completion dependency on CUDA.
                    if cuda:
                        torch.cuda._sleep(1_000_000)
                    value = (logical + 1) * (iteration + 1)
                    full = torch.empty(16, device=device)
                    if operation == "all_gather":
                        source = (
                            full.narrow(0, members.index(logical) * 8, 8)
                            if inplace
                            else torch.empty(8, device=device)
                        )
                    else:
                        source = full
                    source.fill_(value)
                    result = task(
                        source, full if operation == "all_gather" and inplace else None
                    )
                    assert len(result) == (not inplace)
                    output = result[0] if result else full
                    observed = output.clone()
                if cuda:
                    stream.synchronize()
                if operation == "all_gather":
                    expected = torch.tensor(members).add(1).float().repeat_interleave(8)
                else:
                    expected = torch.full(
                        (output.numel(),), sum(rank + 1 for rank in members)
                    )
                torch.testing.assert_close(
                    observed.cpu(), expected.float() * (iteration + 1)
                )
            # Invalid local layouts fail before enqueuing a collective. Every
            # participant tries the same failures to avoid unmatched calls.
            with pytest.raises(ValueError, match="packed layout"):
                task(torch.empty((2, 8), device=device))
            if operation == "all_gather" and inplace:
                with pytest.raises(ValueError, match="ordered output view"):
                    task(torch.empty(8, device=device), full)
            elif operation == "all_reduce":
                with pytest.raises(ValueError, match="second tensor"):
                    task(full, full)
    finally:
        if bindings is not None:
            bindings.close()
        dist.destroy_process_group()


def _cpu_context():
    from contextlib import nullcontext

    return nullcontext()


@pytest.mark.parametrize("cuda", [False, True], ids=["gloo", "nccl"])
@pytest.mark.parametrize("world_size", [2, 4], ids=["pair", "two-pairs"])
def test_packed_collective_tasks(tmp_path, cuda, world_size):
    import torch
    import torch.multiprocessing as mp

    if cuda and torch.cuda.device_count() < world_size:
        pytest.skip(f"needs {world_size} GPUs to qualify NCCL task completion")
    workers = mp.spawn(
        _worker,
        args=((tmp_path / "rendezvous").as_uri(), cuda, world_size),
        nprocs=world_size,
        join=False,
    )
    try:
        deadline = time.monotonic() + 90
        while not workers.join(timeout=1):
            if time.monotonic() >= deadline:
                raise TimeoutError("collective workers did not finish")
    finally:
        for process in workers.processes:
            if process.is_alive():
                process.terminate()
        for process in workers.processes:
            process.join(timeout=5)
