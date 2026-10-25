"""Bounded base loading and chunk-independent initialization of LoRA and RoPE."""

import math
from hashlib import sha256

import torch

from workloads.models.qwen35moe.checkpoint import Qwen35MoECheckpoint


class Initialize:
    block_elements = 256 << 10

    def __init__(self, options):
        self.checkpoint = Qwen35MoECheckpoint(
            options["checkpoint"], max_seq_len=options["sequence_length"]
        )
        self.seed = options["seed"]

    def __call__(self, request, destination):
        if destination.device.type != "cpu":
            raise ValueError("checkpoint initialization requires CPU spill storage")
        if request.storage_offset or request.storage_elements != math.prod(
            request.shape
        ):
            raise ValueError("Qwen initialization requires whole contiguous storage")
        name, leaf = request.name, request.name.rsplit(".", 1)[-1]
        if leaf in {"lora_b", "lora_gate_up_b", "lora_down_b"}:
            destination.zero_()
        elif leaf in {"lora_a", "lora_gate_up_a", "lora_down_a"} or name.startswith(
            "rotary."
        ):
            self._generated(request, destination)
        else:
            self.checkpoint.copy_chunk(name, request.shape, request.offset, destination)

    def _generated(self, request, destination):
        width = self.block_elements
        end = request.offset + destination.numel()
        for start in range(request.offset // width * width, end, width):
            stop = min(start + width, request.storage_elements)
            if request.name.startswith("rotary."):
                values = self._rotary(request.name, start, stop)
            else:
                key = f"{self.seed}:{request.name}:{start}".encode()
                generator = torch.Generator().manual_seed(
                    int.from_bytes(sha256(key).digest()[:8], "little")
                )
                values = torch.empty(stop - start, dtype=torch.float32)
                scale = request.shape[-1] ** -0.5
                if request.name.endswith(".lora_a"):
                    values.uniform_(-scale, scale, generator=generator)
                else:
                    values.normal_(std=scale, generator=generator)
            left, right = max(start, request.offset), min(stop, end)
            destination[left - request.offset : right - request.offset].copy_(
                values[left - start : right - start]
            )

    def _rotary(self, name, start, stop):
        if name not in {"rotary.cosine", "rotary.sine"}:
            raise ValueError(f"no initializer for {name}")
        config = self.checkpoint.config
        width = config.rotary_width
        frequency = 1 / (
            config.rope_base ** (torch.arange(0, width, 2).float() / width)
        )
        indices = torch.arange(start, stop)
        phase = (indices // width).float() * frequency[indices % (width // 2)]
        return phase.cos_() if name.endswith("cosine") else phase.sin_()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.checkpoint.close()
