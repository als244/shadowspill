"""Check that each worker's first NVTX range survives capture startup."""

import os

import torch
import torch.distributed as dist

dist.init_process_group("gloo")
rank = dist.get_rank()
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
x = torch.randn(4096, device="cuda")
for _ in range(3):
    x = x.square()
torch.cuda.synchronize()
dist.barrier()
torch.cuda.cudart().cudaProfilerStart()
dist.barrier()
with torch.cuda.nvtx.range(f"ep2/profiled_training/rank_{rank}"):
    for step in range(5):
        with torch.cuda.nvtx.range(f"ep2/training/step_{step:06d}/rank_{rank}"):
            x = x + 1
            torch.cuda.synchronize()
dist.barrier()
if rank == 0:
    torch.cuda.cudart().cudaProfilerStop()
print(f"Rank {rank}: five ranges complete", flush=True)
dist.destroy_process_group()
