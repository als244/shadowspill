"""Version-compatible eager tensor collectives used outside captured graphs."""

import torch.distributed as dist

# PyTorch 2.13 renamed these APIs; older supported versions retain the old names.
if hasattr(dist, "all_gather_single"):
    all_gather_single = dist.all_gather_single
else:
    all_gather_single = dist.all_gather_into_tensor

if hasattr(dist, "reduce_scatter_single"):
    reduce_scatter_single = dist.reduce_scatter_single
else:
    reduce_scatter_single = dist.reduce_scatter_tensor
