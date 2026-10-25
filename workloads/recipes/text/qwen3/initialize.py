"""Bounded Qwen3 initialization with identical values across shards and chunk sizes."""

import math
from hashlib import sha256

import torch


class Initialize:
    """Match Qwen's distributions, not its module reset traversal's RNG stream.

    Masters and optimizer state use the Plan's built-in initializers. Scratch is
    bounded to 256K elements; no full model or expert bank is materialized.
    """

    block_elements = 256 << 10

    def __init__(self, config, *, seed=0):
        self.config, self.seed = config, seed

    def __call__(self, request, destination):
        if (
            destination.device.type != "cpu"
            or request.storage_offset
            or (request.storage_elements != math.prod(request.shape))
        ):
            raise ValueError("Qwen initializer requires contiguous CPU spill storage")
        name = request.name
        if name.endswith("norm.weight"):
            destination.fill_(1)
            return
        if name not in {
            "embed.weight",
            "lm_head.weight",
            "rotary.cosine",
            "rotary.sine",
        } and not name.startswith("blocks."):
            raise ValueError(f"no Qwen3 initializer for {name}")
        width, end = self.block_elements, request.offset + destination.numel()
        for start in range(request.offset // width * width, end, width):
            stop = min(start + width, request.storage_elements)
            if name.startswith("rotary."):
                head = self.config.rotary_width
                index = torch.arange(start, stop)
                frequency = 1 / self.config.rope_base ** (
                    2 * (index % (head // 2)).float() / head
                )
                phase = (index // head).float() * frequency
                values = phase.cos_() if name.endswith("cosine") else phase.sin_()
            else:
                seed = int.from_bytes(
                    sha256(f"{self.seed}:{name}:{start}".encode()).digest()[:8],
                    "little",
                )
                values = torch.empty(stop - start).normal_(
                    std=self.config.initializer_range,
                    generator=torch.Generator().manual_seed(seed),
                )
            left, right = max(start, request.offset), min(stop, end)
            destination[left - request.offset : right - request.offset].copy_(
                values[left - start : right - start]
            )
