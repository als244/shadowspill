"""Restore one representation with mixed ownership, ties, strides and tail shards."""

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
from shadowspill.training._model import initialize_model


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(
            torch.arange(15, dtype=torch.bfloat16).reshape(3, 5).t()
        )
        self.alias = self.weight
        self.one = nn.Parameter(torch.ones(1, dtype=torch.bfloat16))
        self.local = nn.Parameter(
            torch.full((3,), float(dist.get_rank()), dtype=torch.bfloat16)
        )
        self.register_buffer("counter", torch.tensor(dist.get_rank()))


def worker(rank, root):
    dist.init_process_group(
        "gloo",
        init_method="file://" + str(Path(root, "store")),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=40),
    )
    try:
        for sharded in (False, True):
            model = Model()
            bound = Distributed(
                dist.group.WORLD, replica_overrides=[(["local"], None)]
            )._bind(
                model, dist.group.WORLD, namespace="checkpoint-layouts-" + str(sharded)
            )
            bound.shard_optimizer = sharded
            try:
                full = {
                    name: value.detach().float() + 0.013
                    for name, value in model.named_parameters()
                }
                masters = {}
                for name, value in full.items():
                    if sharded and name != "local":
                        capacity = (value.numel() + 1) // 2
                        owned = torch.zeros(capacity)
                        piece = value.flatten()[rank * capacity : (rank + 1) * capacity]
                        owned[: piece.numel()] = piece
                        masters[name] = owned
                    else:
                        masters[name] = value
                call = PlannedTrainStep.__new__(PlannedTrainStep)
                call._state = type("State", (), {"model": model})()
                call._step = 3
                call._distributed_layout = bound.checkpoint_layout()
                for choice in ("master", "compute"):
                    state = call._checkpoint_payload(
                        model.state_dict(), {}, masters, weights=choice
                    )
                    with torch.device("meta"):
                        target = Model()
                    initialize_model(
                        target,
                        state=state["model"],
                        missing_parameters=master_aliases(state, bound),
                    )
                    restore_compute_weights(
                        dict(target.named_parameters()),
                        state["masters"],
                        bound,
                        chunk_bytes=8,
                    )
                    assert target.weight is target.alias
                    assert target.weight.stride() == model.weight.stride()
                    assert target.counter.item() == rank
                    for name, value in target.named_parameters():
                        want = (
                            full[name].bfloat16()
                            if choice == "master"
                            else model.get_parameter(name)
                        )
                        torch.testing.assert_close(value, want, rtol=0, atol=0)
                    if choice == "compute":
                        assert not state["masters"]
                        restored = masters_from_compute(
                            state["model"], set(masters), bound
                        )
                        for name, value in restored.items():
                            if sharded and name != "local":
                                flat = model.get_parameter(name).detach().flatten()
                                capacity = (flat.numel() + 1) // 2
                                expected = torch.zeros(capacity, dtype=torch.bfloat16)
                                piece = flat[rank * capacity : (rank + 1) * capacity]
                                expected[: piece.numel()] = piece
                            else:
                                expected = model.get_parameter(name).detach()
                            torch.testing.assert_close(value, expected, rtol=0, atol=0)
                    else:
                        assert set(state["model"]) == {"counter"}
            finally:
                bound.close()
    finally:
        dist.destroy_process_group()


def test_mixed_layouts_restore_each_representation():
    with tempfile.TemporaryDirectory() as root:
        mp.spawn(worker, args=(root,), nprocs=2, join=True)
