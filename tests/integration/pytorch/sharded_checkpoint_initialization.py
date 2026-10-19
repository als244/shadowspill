"""torchrun: exact checkpoint publication across full and local shard groups."""

from __future__ import annotations

import argparse
import json
from datetime import timedelta

import torch
import torch.distributed as dist
from mlops.block_fp8 import BlockFP8Weight
from torch import nn

from shadowspill.memory import device, pinned_host, transfer_route
from shadowspill.pytorch import (
    Runtime,
    import_model_state,
    initialize_storage_,
    read_model_state,
    release_model_state,
)
from shadowspill.pytorch.accelerator import resolve_device
from workloads.recipes.text.distributed import data_parallel_groups


class State(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(
            BlockFP8Weight(
                torch.empty((1, 128, 128), dtype=torch.float8_e4m3fn),
                torch.empty((1, 1, 1), dtype=torch.float32),
                dtype=torch.bfloat16,
            ),
            requires_grad=False,
        )
        self.bias = nn.Parameter(torch.empty(129))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-size", type=int, required=True)
    args = parser.parse_args()
    if args.shard_size < 1:
        parser.error("--shard-size must be positive")
    dist.init_process_group("gloo", timeout=timedelta(seconds=180))
    rank, world = dist.get_rank(), dist.get_world_size()
    selected = resolve_device("auto")
    try:
        size = args.shard_size
        if world % size:
            raise ValueError("shard size must divide world size")
        with Runtime(
            pools={
                "execution": device(device=selected.index, physical_capacity=2 << 30),
                "spill": pinned_host(capacity=64 << 20),
            },
            routes={
                "fetch": transfer_route(source="spill", destination="execution"),
                "evict": transfer_route(source="execution", destination="spill"),
            },
            calibrate=False,
            control_group=dist.group.WORLD,
        ) as runtime:
            specification, resources = data_parallel_groups(
                device=selected, shard_size=size, shard_gradients=True, timeout=180
            )
            with resources():
                with torch.device("meta"):
                    model = State()

                @torch.no_grad()
                def initialize(target):
                    loaded = []

                    def fill_weight(value):
                        loaded.append("weight")
                        value.copy_(
                            ((torch.arange(128 * 128) + rank) % 8)
                            .reshape(1, 128, 128)
                            .to(torch.float8_e4m3fn)
                        )

                    def fill_scale(value):
                        loaded.append("scale")
                        value.fill_(0.125)

                    def fill_bias(value):
                        loaded.append("bias")
                        value.copy_(torch.arange(129, dtype=torch.float32) + rank)

                    initialize_storage_(target.weight._data, fill_weight)
                    initialize_storage_(target.weight._scale, fill_scale)
                    initialize_storage_(target.bias, fill_bias)
                    assert loaded == (
                        ["weight", "scale", "bias"] if rank % size == 0 else []
                    )
                    target.bias[3:7].add_(10 + rank)

                imported = import_model_state(
                    model,
                    runtime=runtime,
                    pool="spill",
                    distributed=specification,
                    initialize=initialize,
                )
                actual = read_model_state(imported, runtime=runtime)
                expected_weight = (torch.arange(128 * 128) % 8).reshape(1, 128, 128)
                torch.testing.assert_close(
                    actual["weight"].dequantize(dtype=torch.float32),
                    expected_weight.float() * 0.125,
                    rtol=0,
                    atol=0,
                )
                expected_bias = torch.arange(129, dtype=torch.float32)
                expected_bias[3:7] += 10
                torch.testing.assert_close(
                    actual["bias"], expected_bias, rtol=0, atol=0
                )
                release_model_state(imported, runtime=runtime)
                print(
                    json.dumps({"rank": rank, "shard_size": size, "passed": True}),
                    flush=True,
                )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
