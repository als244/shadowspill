"""Host spill admission retains cgroup accounting and coordinates probe failures."""

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from benchmarking import host_memory
from benchmarking.host_memory.limits import Limit, Resources


def test_cgroup_budget_reuses_clean_unmapped_checkpoint_cache(tmp_path):
    from benchmarking.host_memory.facts import _cgroup_available_memory
    from benchmarking.host_memory.limits import (
        Limit,
        Resources,
        validate_resources,
    )

    gib = 1 << 30
    (tmp_path / "memory.stat").write_text(
        f"anon {2 * gib}\nfile {32 * gib}\n"
        f"active_file {31 * gib}\ninactive_file {gib}\n"
        f"file_mapped {gib}\nfile_dirty 0\nfile_writeback 0\nshmem 0\nunevictable 0\n"
    )
    available = _cgroup_available_memory(tmp_path, 240 * gib, 34 * gib)
    assert available == 237 * gib
    # Two 104 GiB pools plus 2 GiB staging each fit without forcing the user
    # to drop caches or change a previously admitted configuration.
    validate_resources(
        [
            Resources(
                "node", f"gpu{rank}", 104 * gib, 2 * gib, (Limit("cgroup", available),)
            )
            for rank in range(2)
        ]
    )


def test_cgroup_budget_keeps_nonreclaimable_file_memory_charged(tmp_path):
    from benchmarking.host_memory.facts import _cgroup_available_memory

    (tmp_path / "memory.stat").write_text(
        "file 60\nshmem 10\nfile_mapped 10\n"
        "file_dirty 5\nfile_writeback 5\nunevictable 10\n"
    )
    assert _cgroup_available_memory(tmp_path, 100, 90) == 30
    # Counters can race, and exclusions can overlap. Never admit more than
    # the cgroup ceiling or report negative space because of a snapshot.
    assert _cgroup_available_memory(tmp_path, 100, 10) == 100
    (tmp_path / "memory.stat").write_text("file 10\nshmem 20\n")
    assert _cgroup_available_memory(tmp_path, 100, 110) == 0
    (tmp_path / "memory.stat").unlink()
    assert _cgroup_available_memory(tmp_path, 100, 90) == 10


def _worker(rank, rendezvous):
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=15),
    )
    try:

        def probe(device, spill, staging):
            return Resources("node", f"gpu{rank}", spill, staging, (Limit("ram", 10),))

        host_memory.local_resources = probe
        host_memory.preflight(torch.device("cpu"), 4, 1, group=dist.group.WORLD)
        with pytest.raises(ValueError, match="exceeds host-memory"):
            host_memory.preflight(torch.device("cpu"), 5, 1, group=dist.group.WORLD)

        def failed_probe(*args):
            if rank == 0:
                raise OSError("probe failed")
            return probe(*args)

        host_memory.local_resources = failed_probe
        with pytest.raises(ValueError, match="worker 0: OSError: probe failed"):
            host_memory.preflight(torch.device("cpu"), 4, 1, group=dist.group.WORLD)
        assert not torch.cuda.is_initialized()
    finally:
        dist.destroy_process_group()


def test_host_preflight_coordinates_shared_capacity_and_probe_failures(tmp_path):
    mp.spawn(_worker, args=((tmp_path / "rendezvous").as_uri(),), nprocs=2, join=True)
