"""Actual CPU collectives import native weights and coordinate preflight failures."""

from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from engine_tests.checkpoint.test_weights import manifest
from engine_tests.checkpoint.test_weights_conversion import flat_states
from engine_tests.checkpoint.test_weights_load import HostExecutor
from shadowspill.checkpoint import write_weights
from shadowspill.checkpoint.weights_load import load_weights
from shadowspill.communication.binding.groups import GroupBindings


def _worker(launcher_rank, root):
    root = Path(root)
    dist.init_process_group(
        "gloo",
        init_method=(root / "rendezvous").as_uri(),
        rank=launcher_rank,
        world_size=4,
        timeout=timedelta(seconds=30),
    )
    groups = None
    try:
        groups = GroupBindings(
            identity="native-weights",
            groups=(),
            rank_map={0: 3, 1: 1, 2: 0, 3: 2},
            topology={},
            backend="gloo",
            timeout=timedelta(seconds=30),
        )
        rank = groups.logical_rank
        for elements in (1, 15):
            for weights in ("master", "compute"):
                state = flat_states(elements, replicated=True)[rank]
                executor = HostExecutor(state, SimpleNamespace(groups=groups))
                complete = torch.arange(elements, dtype=torch.float32) / 7 + 0.12345

                def fill(request, destination, complete=complete):
                    if request.name.startswith("optimizer."):
                        destination.fill_(0.9)
                    else:
                        destination.copy_(
                            complete[
                                request.offset : request.offset + destination.numel()
                            ]
                        )

                executor.initialize(fill, chunk_bytes=17, restore=True)
                path = root / f"{elements}-{weights}-{rank}.ssp"
                with path.open("wb") as stream:
                    write_weights(executor, stream, chunk_bytes=17, weights=weights)
                info = manifest(path)
                assert bool(info["gathered_tensors"]) == (weights == "master")
                for value in executor.values.values():
                    value.fill_(-9)
                load_weights(executor, path, chunk_bytes=17)
                expected = (
                    complete if weights == "master" else complete.bfloat16().float()
                )
                for name, value in executor.values.items():
                    region = executor._paths[0].storage.regions.get(name)
                    offset, length = (
                        (0, elements)
                        if region is None
                        else (region.offset, region.length)
                    )
                    if name == "moment":
                        assert not value.count_nonzero()
                    else:
                        torch.testing.assert_close(
                            value[:length],
                            expected[offset : offset + length].to(value.dtype),
                            rtol=0,
                            atol=0,
                        )
                        assert not value[length:].count_nonzero()
                assert executor._initialized
                assert (
                    max(row[2] for row in executor._paths[0].storage.spill.transfers)
                    <= 17
                )
                # A checksum failure on one rank is reported before any other
                # rank changes state or enters replica reconstruction.
                if elements == 15 and weights == "master":
                    before = {
                        name: value.clone() for name, value in executor.values.items()
                    }
                    if rank == 1:
                        with path.open("r+b") as stream:
                            stream.seek(512)
                            first = stream.read(1)[0]
                            stream.seek(512)
                            stream.write(bytes([first ^ 255]))
                    dist.barrier()
                    with pytest.raises(RuntimeError, match="checksum differs"):
                        load_weights(executor, path, chunk_bytes=17)
                    torch.testing.assert_close(executor.values, before, rtol=0, atol=0)
                    assert executor._initialized
        assert not torch.cuda.is_initialized()
    finally:
        if groups is not None:
            groups.close()
        dist.destroy_process_group()


def test_actual_gloo_import_resets_moments_and_preserves_empty_master_owners(tmp_path):
    mp.spawn(_worker, args=(str(tmp_path),), nprocs=4, join=True)
