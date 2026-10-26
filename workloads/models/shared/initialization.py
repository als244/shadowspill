"""Bounded random generation independent of runtime chunks and rank ownership."""

import math
from hashlib import sha256

import torch


def validate_chunk(request, destination):
    """Recipes fill flat CPU chunks of a contiguous declared tensor."""
    if (
        destination.device.type != "cpu"
        or destination.ndim != 1
        or not destination.is_contiguous()
        or request.storage_offset
        or request.storage_elements != math.prod(request.shape)
        or request.offset < 0
        or request.offset + destination.numel() > request.storage_elements
    ):
        raise ValueError("initialization requires a contiguous CPU tensor chunk")


def fill_generated(request, destination, *, sample, seed, block_elements=256 << 10):
    """Generate fixed RNG blocks, then copy only the requested intersection.

    ``sample(name, shape, start, stop, generator)`` supplies the model's values.
    Scratch is bounded by one block, even for a multi-GiB expert bank. Naming
    each RNG block makes the result independent of shard and loader chunk sizes.
    """
    validate_chunk(request, destination)
    if block_elements <= 0:
        raise ValueError("block_elements must be positive")
    end = request.offset + destination.numel()
    for start in range(
        request.offset // block_elements * block_elements, end, block_elements
    ):
        stop = min(start + block_elements, request.storage_elements)
        key = f"{seed}:{request.name}:{start}".encode()
        generator = torch.Generator().manual_seed(
            int.from_bytes(sha256(key).digest()[:8], "little")
        )
        values = sample(request.name, request.shape, start, stop, generator)
        left, right = max(start, request.offset), min(stop, end)
        destination[left - request.offset : right - request.offset].copy_(
            values[left - start : right - start]
        )
