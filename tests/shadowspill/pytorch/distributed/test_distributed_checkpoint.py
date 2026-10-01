from __future__ import annotations

import tempfile
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from shadowspill.pytorch.callables import PlannedTrainStep
from shadowspill.pytorch.distributed import Distributed
from shadowspill.pytorch.distributed._checkpoint import (
    master_aliases,
    masters_from_compute,
    restore_compute_weights,
)
from shadowspill.training import _checkpoint
from shadowspill.training._model import initialize_model


class Execution:
    def __init__(self, model, bound, rank):
        self.model = model
        self.call = PlannedTrainStep.__new__(PlannedTrainStep)
        self.call._state = type("State", (), {"model": model})()
        self.call._step = 7
        self.call._distributed_layout = bound.checkpoint_layout()
        self.master = torch.arange(3, dtype=torch.float32) + rank * 10
        self.optimizer = {
            "state": {0: {"exp_avg": self.master.clone()}},
            "param_groups": [],
        }

    def synchronize(self):
        pass

    def save(self, path, *, weights="master"):
        torch.save(
            self.call._checkpoint_payload(
                self.model.state_dict(),
                self.optimizer,
                {"weight": self.master},
                weights=weights,
            ),
            path,
        )


def worker(rank, root):
    dist.init_process_group(
        "gloo",
        init_method="file://" + str(Path(root, "store")),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    model = nn.Linear(3, 2, bias=False, dtype=torch.bfloat16)
    model.register_buffer("local", torch.tensor(rank))
    bound = Distributed(dist.group.WORLD, timeout=15)._bind(
        model, dist.group.WORLD, namespace="checkpoint"
    )
    try:
        execution = Execution(model, bound, rank)
        loop = {
            "step": 7,
            "source": {"offset": rank * 100},
            "rng": _checkpoint.rng_state(torch.device("cpu")),
        }
        path = Path(root, "complete")
        _checkpoint.save(path, execution, loop, distributed=bound)
        state, restored = _checkpoint.load(path, distributed=bound)
        assert "weight" not in state["model"]
        target = nn.Linear(3, 2, bias=False, dtype=torch.bfloat16, device="meta")
        target.register_buffer(
            "local", torch.empty((), device="meta", dtype=torch.int64)
        )
        initialize_model(
            target,
            state=state["model"],
            missing_parameters=master_aliases(state, bound),
        )
        restore_compute_weights(
            dict(target.named_parameters()), state["masters"], bound, chunk_bytes=8
        )
        expected = torch.tensor([[0, 1, 2], [10, 11, 12]], dtype=torch.bfloat16)
        torch.testing.assert_close(target.weight, expected)
        assert state["masters"]["weight"].shape == (3,)
        assert state["model"]["local"].item() == rank
        assert restored["source"] == loop["source"]
        torch.testing.assert_close(state["masters"]["weight"], execution.master)
        assert sorted(p.name for p in path.iterdir()) == [
            "manifest.json",
            "rank-00000",
            "rank-00001",
        ]

        compute_path = Path(root, "compute")
        with torch.no_grad():
            model.weight.copy_(expected)
        _checkpoint.save(
            compute_path, execution, loop, distributed=bound, weights="compute"
        )
        compute, _ = _checkpoint.load(compute_path, distributed=bound)
        assert not compute["masters"]
        torch.testing.assert_close(compute["model"]["weight"], expected, rtol=0, atol=0)
        owned = masters_from_compute(compute["model"], {"weight"}, bound)
        assert owned["weight"].dtype == torch.bfloat16
        torch.testing.assert_close(
            owned["weight"].float(), execution.master, rtol=0, atol=0
        )
        restored_master = torch.empty_like(execution.master)
        restored_master.copy_(owned["weight"])
        torch.testing.assert_close(restored_master, execution.master, rtol=0, atol=0)

        if rank == 0:
            try:
                _checkpoint.load(path)
            except ValueError as error:
                assert "distributed configuration" in str(error)
            else:
                raise AssertionError("ordinary load accepted rank checkpoint")
        bound.control.agree("checkpoint/finished", True)
    finally:
        bound.close()
        dist.destroy_process_group()


def test_rank_state_and_owned_masters_publish_atomically():
    with tempfile.TemporaryDirectory() as root:
        mp.spawn(worker, args=(root,), nprocs=2, join=True)
