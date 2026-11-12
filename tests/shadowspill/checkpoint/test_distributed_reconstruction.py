"""Actual Gloo reload of replicas, with permuted ranks and independent groups."""

from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from shadowspill import Executor, load_checkpoint, save_checkpoint
from shadowspill.communication.binding.groups import GroupBindings
from tests.shadowspill.checkpoint.test_weights import OwnedExecutor, manifest, rewrite
from tests.shadowspill.execution.test_shard_initialization import fixture


class HostExecutor(OwnedExecutor):
    # Exercise the real initialization/coordinated reconstruction method with
    # host spill storage; no CUDA runtime is needed for these transfer tests.
    initialize = Executor.initialize

    def __init__(self, program, worker):
        super().__init__(program, worker.groups.logical_rank)
        self._distributed = worker
        self._named_paths = {}
        self._paths = [
            SimpleNamespace(
                storage=self.storage,
                controls=SimpleNamespace(initialize=lambda *args, **kwargs: None),
            )
        ]

    def synchronize(self):
        pass


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
            identity="replicated-checkpoint",
            groups=(),
            rank_map={0: 3, 1: 1, 2: 0, 3: 2},
            topology={},
            backend="gloo",
            timeout=timedelta(seconds=30),
        )
        rank = groups.logical_rank
        for elements, weights, replicated in (
            (15, "master", True),
            (1, "master", True),
            (15, "compute", True),
            (15, None, True),
            (15, None, False),
        ):
            program = fixture(elements=elements)
            bindings = dict(program.bindings)
            if replicated:
                bindings["weight_shards"] = {}
            if weights is None:
                bindings["optimizer_bindings"] = [
                    item
                    for item in bindings["optimizer_bindings"]
                    if item["role"] == "state"
                ]
            program = replace(program, bindings=bindings)
            executor = HostExecutor(program, SimpleNamespace(groups=groups))
            base = (rank // 2) * 100
            complete = torch.arange(elements, dtype=torch.float32) / 7 + base
            for name, tensor in executor.values.items():
                region = executor.storage.regions.get(name)
                offset = 0 if region is None else region.offset
                length = elements if region is None else region.length
                tensor[:length].copy_(complete[offset : offset + length])
            expected = {name: value.clone() for name, value in executor.values.items()}
            checkpoint = root / f"{elements}-{weights}-{replicated}-rank-{rank}.ssp"
            save_checkpoint(
                executor, checkpoint, chunk_bytes=20, weights=weights or "master"
            )
            info = manifest(checkpoint)
            assert not info["retained_weight_copies"]
            stored = {entry["name"] for entry in info["tensors"]}
            if weights == "master":
                assert info["gathered_tensors"] == {
                    "layer.weight": {
                        "source": "master.layer.weight",
                        "groups": [[1, 0], [3, 2]],
                    },
                }
                assert "layer.weight" not in stored
            else:
                assert not info["gathered_tensors"]
                assert "master.layer.weight" not in stored
                if weights == "compute":
                    region = executor.storage.regions["master"]
                    expected["master"][: region.length].copy_(
                        expected["weight"][
                            region.offset : region.offset + region.length
                        ]
                    )
            for value in executor.values.values():
                value.zero_()
            load_checkpoint(executor, checkpoint, chunk_bytes=17)
            assert executor._initialized
            for name, actual in executor.values.items():
                region = executor.storage.regions.get(name)
                count = elements if region is None else region.length
                torch.testing.assert_close(
                    actual[:count], expected[name][:count], rtol=0, atol=0
                )
                assert not actual[count:].count_nonzero()
            assert (
                max(transfer[2] for transfer in executor.storage.spill.transfers) <= 20
            )
            # Repeated restoration reuses CPU process groups and bounded buffers.
            count = len(groups._owned)
            load_checkpoint(executor, checkpoint, chunk_bytes=17)
            assert len(groups._owned) == count
            if elements == 15 and weights == "master":
                if rank == 1:

                    def corrupt(info, contents):
                        entry = next(
                            item
                            for item in info["tensors"]
                            if item["name"] == "master.layer.weight"
                        )
                        payload = contents[entry["member"]]
                        contents[entry["member"]] = (
                            bytes([payload[0] ^ 1]) + payload[1:]
                        )

                    rewrite(checkpoint, corrupt)
                dist.barrier()
                with pytest.raises(RuntimeError, match="checksum differs"):
                    load_checkpoint(executor, checkpoint, chunk_bytes=17)
                assert not executor._initialized
        assert not torch.cuda.is_initialized()
    finally:
        if groups is not None:
            groups.close()
        dist.destroy_process_group()


def test_distributed_weight_representations_reload_with_logical_rank_remapping(
    tmp_path,
):
    mp.spawn(_worker, args=(str(tmp_path),), nprocs=4, join=True)
