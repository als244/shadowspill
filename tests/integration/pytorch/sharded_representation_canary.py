"""torchrun: frozen FP8 data/scales use local shards through public training."""

from __future__ import annotations

import argparse
import copy
import json
import os
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from mlops.block_fp8 import BlockFP8Weight
from torch import nn
from torch.nn import functional as F

from shadowspill.pytorch import ProfilingOptions
from shadowspill.pytorch.state.storage import persistent_state
from shadowspill.training import Trainer
from shadowspill.training.backends import ShadowSpill
from workloads.recipes.text.distributed import data_parallel_groups


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        data = (torch.arange(128 * 128) % 8).reshape(1, 128, 128)
        self.weight = nn.Parameter(
            BlockFP8Weight(
                data.to(torch.float8_e4m3fn),
                torch.full((1, 1, 1), 0.125),
                dtype=torch.bfloat16,
            ),
            requires_grad=False,
        )
        self.bias = nn.Parameter(torch.zeros(128))

    def forward(self, x):
        return F.linear(x, self.weight.dequantize(dtype=torch.float32)[0], self.bias)


def objective(model, data):
    x, world = data
    return model(x).square().mean() / world


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-size", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    args.out.mkdir(parents=True, exist_ok=True)
    dist.init_process_group("gloo", timeout=timedelta(seconds=300))
    try:
        with ShadowSpill(
            device="auto",
            execution_gib=2,
            spill_gib=1,
            control_group=dist.group.WORLD,
            preparation_timeout=300,
            artifact_store=args.out / f"rank-{rank}" / "artifacts",
            profiling_options=ProfilingOptions(
                conditioning_seconds=0,
                measurement_seconds=0,
                minimum_samples=3,
            ),
        ) as backend:
            specification, resources = data_parallel_groups(
                device=backend.device,
                shard_size=args.shard_size,
                shard_gradients=True,
                timeout=300,
            )
            with resources():
                model = Network()
                reference = copy.deepcopy(model).to(backend.device)
                generator = torch.Generator().manual_seed(100 + rank)
                data = (torch.randn(3, 128, generator=generator), dist.get_world_size())
                with Trainer(
                    model,
                    objective=objective,
                    optimizer=torch.optim.SGD,
                    optimizer_args={"lr": 0.01},
                    backend=backend,
                    distributed=specification,
                ) as trainer:
                    trainer.prepare(data)
                    state = persistent_state(backend.runtime, trainer.model)
                    assert state is not None
                    components = [
                        item
                        for item in state.storages
                        if any(view.name.startswith("weight.") for view in item.views)
                    ]
                    assert len(components) == 2
                    for item in components:
                        assert item.sharding is not None, item.views
                        assert item.sharding.execution_bytes == (
                            args.shard_size * item.sharding.shard_bytes
                        )
                    loss = objective(reference, (data[0].to(backend.device), data[1]))
                    loss.backward()
                    dist.all_reduce(reference.bias.grad, group=specification.group)
                    with torch.no_grad():
                        reference.bias.add_(reference.bias.grad, alpha=-0.01)
                    trainer.step(data)
                    actual = trainer._execution.call.state_dict(weights="compute")
                    torch.testing.assert_close(
                        actual["model"]["bias"],
                        reference.bias.detach().cpu(),
                        rtol=1e-5,
                        atol=1e-6,
                    )
                    torch.testing.assert_close(
                        actual["model"]["weight"].dequantize(dtype=torch.float32),
                        reference.weight.dequantize(dtype=torch.float32).cpu(),
                        rtol=0,
                        atol=0,
                    )
                    result = {
                        "rank": rank,
                        "shard_size": args.shard_size,
                        "physical_components": len(components),
                        "spill_bytes": sum(
                            item.sharding.shard_bytes for item in components
                        ),
                        "passed": True,
                    }
                    (args.out / f"rank-{rank}.json").write_text(
                        json.dumps(result) + "\n"
                    )
                    print(json.dumps(result), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
