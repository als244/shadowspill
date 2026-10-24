"""Native host sizes, bounded loading and asynchronous descriptor roundtrips."""

from time import monotonic

import pytest

from tests.shadowspill.execution._sharded_storage import program, worker


@pytest.mark.cuda
@pytest.mark.parametrize("mode", ["host", "local_copy", "nccl"])
@pytest.mark.parametrize("world_size", [2, 4], ids=["one-replica", "two-replicas"])
def test_native_worker_storage(tmp_path, mode, world_size):
    import torch
    import torch.multiprocessing as mp

    if not torch.cuda.is_available():
        pytest.skip("needs a GPU for native pinned-storage integration")
    if mode == "nccl" and torch.cuda.device_count() < world_size:
        pytest.skip(f"NCCL shard roundtrips need {world_size} distinct GPUs")
    # Gloo checks host registration only. Singleton transport groups exercise
    # native copies on one GPU; real gather completion is checked with NCCL.
    archive = tmp_path / "program.ssp"
    program(shard_ranks=1 if mode == "local_copy" else 2, world_size=world_size).save(
        archive
    )
    context = mp.spawn(
        worker,
        args=((tmp_path / "group").as_uri(), str(archive), mode),
        nprocs=world_size,
        join=False,
    )
    try:
        deadline = monotonic() + 90
        while not context.join(timeout=1):
            if monotonic() >= deadline:
                pytest.fail("sharded storage worker deadline exceeded")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=5)
