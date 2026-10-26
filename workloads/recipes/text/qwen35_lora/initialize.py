"""Bounded base loading and chunk-independent initialization of LoRA and RoPE."""

import torch

from workloads.models.qwen35moe.checkpoint import Qwen35MoECheckpoint
from workloads.models.shared.initialization import fill_generated, validate_chunk


class Initialize:
    block_elements = 256 << 10

    def __init__(self, options):
        self.checkpoint = Qwen35MoECheckpoint(
            options["checkpoint"], max_seq_len=options["sequence_length"]
        )
        self.seed = options["seed"]

    def __call__(self, request, destination):
        validate_chunk(request, destination)
        name, leaf = request.name, request.name.rsplit(".", 1)[-1]
        if leaf in {"lora_b", "lora_gate_up_b", "lora_down_b"}:
            destination.zero_()
        elif leaf in {"lora_a", "lora_gate_up_a", "lora_down_a"} or name.startswith(
            "rotary."
        ):
            fill_generated(
                request,
                destination,
                sample=self._values,
                seed=self.seed,
                block_elements=self.block_elements,
            )
        else:
            self.checkpoint.copy_chunk(name, request.shape, request.offset, destination)

    def _values(self, name, shape, start, stop, generator):
        if name.startswith("rotary."):
            return self._rotary(name, start, stop)
        values = torch.empty(stop - start, dtype=torch.float32)
        scale = shape[-1] ** -0.5
        if name.endswith(".lora_a"):
            return values.uniform_(-scale, scale, generator=generator)
        return values.normal_(std=scale, generator=generator)

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
