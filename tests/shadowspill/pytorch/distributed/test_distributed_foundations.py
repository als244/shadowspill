from __future__ import annotations

import copy
import tempfile
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as collectives
import torch.multiprocessing as mp
from torch import nn
from torch._subclasses.fake_tensor import FakeTensorMode

from shadowspill.pytorch.capture.artifacts import GraphArtifact
from shadowspill.pytorch.capture.fake import fake_device_model
from shadowspill.pytorch.distributed import Distributed, borrowed_group_memo, current
from shadowspill.pytorch.distributed._preparation import prepared_model_import
from shadowspill.pytorch.runtime import Runtime


def test_cgroup_budget_reuses_clean_unmapped_checkpoint_cache(tmp_path):
    from shadowspill.pytorch.distributed._bootstrap import _cgroup_available_memory
    from shadowspill.pytorch.distributed._resources import (
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
    from shadowspill.pytorch.distributed._bootstrap import _cgroup_available_memory

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


class GroupModel(nn.Module):
    def __init__(self, group):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(3, 4))
        self.tied = self.weight
        self.expert = nn.Parameter(torch.full((2, 2), float(dist.get_rank())))
        self.register_buffer("rank_buffer", torch.tensor(dist.get_rank()))
        self.group = group

    def forward(self, x):
        return collectives.wait_tensor(
            collectives.all_reduce(x @ self.weight, "sum", self.group)
        )


def _worker(rank, path):
    dist.init_process_group(
        "gloo",
        init_method="file://" + path,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        model = GroupModel(dist.group.WORLD)
        binding = Distributed(
            dist.group.WORLD, replica_overrides=[([model.expert], None)], timeout=20
        )
        prepared = binding._bind(model, dist.group.WORLD, namespace="foundation")
        try:
            with torch.no_grad():
                model.weight.fill_(rank + 1)
            prepared.synchronize_initial(model)
            torch.testing.assert_close(model.weight, torch.ones_like(model.weight))
            assert int(model.rank_buffer) == rank
            assert float(model.expert[0, 0]) == rank
            assert current() is None

            # Exercise preparation without allocating accelerator pools. The
            # GPU regression checks actual import; this fixes the source/copy
            # synchronization contract on machines with only one device too.
            runtime = object.__new__(Runtime)
            runtime._distributed_models = {}
            runtime._distributed_for = lambda *_args: prepared

            @prepared_model_import
            def import_copy(source, *, runtime, distributed):
                return copy.deepcopy(source, borrowed_group_memo())

            with torch.no_grad():
                model.weight.fill_(rank + 1)
            version = model.weight._version
            imported = import_copy(model, runtime=runtime, distributed=binding)
            assert model.weight._version == version
            torch.testing.assert_close(
                model.weight, torch.full_like(model.weight, rank + 1)
            )
            torch.testing.assert_close(
                imported.weight, torch.ones_like(imported.weight)
            )
            assert imported.tied is imported.weight
            assert imported.group is model.group
            assert imported.rank_buffer.item() == rank
            assert prepared.initialized
            # Importing the unchanged source a second time creates another
            # independent copy that must also receive synchronized values.
            again = import_copy(model, runtime=runtime, distributed=binding)
            torch.testing.assert_close(again.weight, imported.weight)
            assert model.weight._version == version
            with prepared.activate():
                assert borrowed_group_memo()[id(dist.group.WORLD)] is dist.group.WORLD
                mode = FakeTensorMode(allow_non_fake_inputs=True)
                fake = fake_device_model(model, mode)
                assert fake is not model and fake.group is model.group
                assert fake.weight is fake.tied
                assert fake.weight.device.type == "cuda"
                assert model.weight.device.type == "cpu"
                records = {p.name: p for p in prepared.parameters}
                assert records["weight"].replicas == (0, 1)
                assert records["expert"].replicas == (rank,)
                assert "rank_buffer" not in records
                sample = torch.ones(2, 3)
                torch.export.export(model, (sample,))

                # Exported state is lifted; use the bound module for an input-
                # only graph around a collective to test automatic alias names.
                class Sum(nn.Module):
                    def forward(self, x):
                        return collectives.wait_tensor(
                            collectives.all_reduce(x, "sum", dist.group.WORLD)
                        )

                graph = torch.export.export(Sum(), (sample,)).graph_module
                artifact = GraphArtifact.capture(
                    kind="inference", graph_module=graph, example_inputs=(sample,)
                )
                assert "shadowspill/group/" in str(artifact.graph_module.graph)
                torch.testing.assert_close(artifact.graph_module(sample)[0], sample * 2)
            assert current() is None
        finally:
            prepared.close()
        # Closing the binding never destroys the caller's process group.
        value = torch.tensor(rank + 1)
        dist.all_reduce(value)
        assert int(value) == 3
        # A model may already produce complete gradients; None means no SUM.
        complete = Distributed(dist.group.WORLD, gradient_group=None, timeout=20)
        prepared = complete._bind(model, dist.group.WORLD, namespace="complete")
        assert all(p.contributions == (rank,) for p in prepared.parameters)
        prepared.close()
    finally:
        dist.destroy_process_group()


def test_actual_copy_and_capture_use_borrowed_groups():
    with tempfile.TemporaryDirectory() as root:
        mp.spawn(_worker, args=(str(Path(root, "store")),), nprocs=2, join=True)


def test_parameter_storage_allows_ties_and_disjoint_slices():
    import pytest

    from shadowspill.pytorch.distributed._ownership import validate_parameter_storage

    bank = torch.arange(20.0)
    model = nn.Module()
    model.left = nn.Parameter(bank[:10])
    model.tied = model.left
    model.right = nn.Parameter(bank[10:])
    validate_parameter_storage(model)
    model.overlap = nn.Parameter(bank[5:15])
    with pytest.raises(ValueError, match="distinct Parameters have overlapping"):
        validate_parameter_storage(model)
