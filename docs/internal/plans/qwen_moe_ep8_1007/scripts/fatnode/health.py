"""Bounded CUDA/NCCL check inside the healthy-device container."""

import json
import os
from datetime import timedelta

import torch
import torch.distributed as dist

rank = int(os.environ["LOCAL_RANK"])
world = int(os.environ["WORLD_SIZE"])
assert torch.cuda.device_count() == world, torch.cuda.device_count()
torch.cuda.set_device(rank)
dist.init_process_group(
    "nccl", timeout=timedelta(seconds=60), device_id=torch.device("cuda", rank)
)
try:
    value = torch.tensor(float(rank + 1), device=f"cuda:{rank}")
    dist.all_reduce(value)
    assert value.item() == world * (world + 1) / 2
    print(
        json.dumps(
            {
                "rank": rank,
                "world": world,
                "uuid": str(torch.cuda.get_device_properties(rank).uuid),
                "all_reduce": value.item(),
                "passed": True,
            }
        ),
        flush=True,
    )
finally:
    dist.destroy_process_group()
