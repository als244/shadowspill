"""Bounded Llama initialization, independent of traversal and loader chunk size."""

import math
from hashlib import sha256

import torch

from workloads.models.shared.decoder import llama3_frequencies


class Initialize:
    """Fill supplied storage with the model's initial distributions/derived tables.

    Fixed-size RNG blocks keep values identical across loader chunk sizes and
    callback ordering. Scratch tensors hold at most 256K elements, never a whole
    parameter. Masters and optimizer state use the Plan's built-in initializers.
    This reproduces distributions, not nn.Module.reset_parameters' RNG stream.
    """

    block_elements = 256 << 10

    def __init__(self, config, *, seed=0):
        self.config, self.seed = config, seed

    def __call__(self, request, destination):
        if destination.device.type != "cpu":
            raise ValueError("Llama initializer expects CPU spill storage")
        if request.storage_offset or request.storage_elements != math.prod(
            request.shape
        ):
            raise ValueError("Llama initializer requires whole contiguous storage")
        name = request.name
        if name.endswith("norm.weight"):
            destination.fill_(1)
            return
        if name not in {"rotary.cosine", "rotary.sine", "embed.weight"} and not (
            name == "lm_head.weight" or name.startswith("blocks.")
        ):
            raise ValueError(f"no Llama initializer for {name}")
        width = self.block_elements
        end = request.offset + destination.numel()
        for start in range(request.offset // width * width, end, width):
            stop = min(start + width, request.storage_elements)
            if name.startswith("rotary."):
                values = self._rotary(name, start, stop)
            else:
                key = f"{self.seed}:{name}:{start}".encode()
                seed = int.from_bytes(sha256(key).digest()[:8], "little")
                generator = torch.Generator().manual_seed(seed)
                values = torch.empty(stop - start, dtype=torch.float32)
                if name == "embed.weight":
                    values.normal_(generator=generator)
                else:
                    bound = request.shape[-1] ** -0.5
                    values.uniform_(-bound, bound, generator=generator)
            left, right = max(start, request.offset), min(stop, end)
            destination[left - request.offset : right - request.offset].copy_(
                values[left - start : right - start]
            )

    def _rotary(self, name, start, stop):
        width = self.config.head_dim
        frequency = 1 / (
            self.config.rope_base
            ** (torch.arange(0, width, 2, dtype=torch.float32) / width)
        )
        if self.config.rope_scaling is not None:
            frequency = llama3_frequencies(frequency, self.config.rope_scaling)
        indices = torch.arange(start, stop)
        phase = (indices // width).float() * frequency[indices % (width // 2)]
        return phase.cos_() if name.endswith("cosine") else phase.sin_()
