"""Map a local expert shard onto the checkpoint's global, bounded tensor reader."""

import math
from dataclasses import replace

import torch

from shadowspill.checkpoint.initialization import validate_chunk


class Initialize:
    def __init__(self, base, *, rank, ranks):
        self.base, self.rank, self.ranks = base, rank, ranks

    def __call__(self, request, destination):
        validate_chunk(request, destination)
        name = request.name.removeprefix("master.")
        mapped = name.replace(".mlp.experts.", ".mlp.")
        mapped = mapped.replace(".gate_up_weight", ".w13").replace(
            ".down_weight", ".w2"
        )
        if mapped not in self.base.geometry and mapped.endswith("._data"):
            mapped = mapped.removesuffix("._data")
        shape, dtype = self.base.geometry[mapped]
        shift = 0
        if ".mlp.experts." in name:
            if shape[0] % self.ranks:
                raise ValueError("checkpoint expert axis does not divide over workers")
            expected = (shape[0] // self.ranks, *shape[1:])
            shift = self.rank * math.prod(expected)
        else:
            expected = shape
        if tuple(request.shape) != tuple(expected):
            raise ValueError(f"local checkpoint geometry differs for {request.name}")
        mapped_request = replace(
            request,
            name=mapped,
            shape=shape,
            dtype=dtype,
            offset=request.offset + shift,
            storage_elements=math.prod(shape),
            region_offset=request.region_offset + shift,
        )
        if dtype == destination.dtype:
            self.base(mapped_request, destination)
        elif dtype in (torch.float32, torch.bfloat16) and destination.dtype in (
            torch.float32,
            torch.bfloat16,
        ):
            scratch = torch.empty(destination.shape, dtype=dtype)
            self.base(mapped_request, scratch)
            destination.copy_(scratch)
        else:
            raise ValueError(
                f"unsupported checkpoint dtype conversion for {request.name}"
            )
