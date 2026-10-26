"""Bounded catalog initialization; no anonymous full model or expert-bank copy."""

import math

import torch

from workloads.models.shared.decoder import llama3_frequencies
from workloads.models.shared.initialization import fill_generated

FAMILIES = {"llama3", "qwen3", "qwen35", "olmoe", "qwen3moe", "qwen35moe"}


class Initialize:
    """Reproduce model distributions, independently of chunks and rank ownership.

    This intentionally does not reproduce reset_parameters' traversal RNG stream.
    Derived RoPE tables and constant gates/norms retain their model definitions.
    Masters, moments and counters use the Program's own initializers.
    """

    block_elements = 256 << 10

    def __init__(self, family, config, *, seed=0):
        self.family, self.config, self.seed = family, config, seed
        if self.family not in FAMILIES:
            raise ValueError(
                f"{self.family} needs its own checkpoint/initialization recipe"
            )

    def __call__(self, request, destination):
        fill_generated(
            request,
            destination,
            sample=self.values,
            seed=self.seed,
            block_elements=self.block_elements,
        )

    def values(self, name, shape, start, stop, generator):
        values = torch.empty(stop - start, dtype=torch.float32)
        leaf = name.rsplit(".", 1)[-1]
        qwen_moe = self.family in {"qwen3moe", "qwen35moe"}
        if name in {"rotary.cosine", "rotary.sine"}:
            return self.rotary(name, start, stop)
        if name.endswith("norm.weight"):
            centered = getattr(
                self.config, "zero_centered_norm", False
            ) and not name.endswith("lin_norm.weight")
            return values.fill_(0 if centered else 1)
        if leaf in {"lora_b", "lora_gate_up_b", "lora_down_b"}:
            return values.zero_()
        if leaf in {"lora_a", "lora_gate_up_a", "lora_down_a"}:
            bound = shape[-1] ** -0.5
            return (
                values.uniform_(-bound, bound, generator=generator)
                if leaf == "lora_a"
                else values.normal_(std=bound, generator=generator)
            )
        if leaf == "A_log":
            return (
                values.uniform_(0.01, 16, generator=generator).log_()
                if qwen_moe
                else values.zero_()
            )
        if leaf == "dt_bias":
            return values.fill_(1 if qwen_moe else 0)
        if qwen_moe and (leaf in {"w13", "w2"} or leaf == "weight"):
            return values.normal_(
                std=self.config.initializer_range, generator=generator
            )
        if leaf in {"w13_experts", "w2_experts"}:
            return values.normal_(std=shape[-1] ** -0.5, generator=generator)
        if name == "embed.weight":
            return values.normal_(generator=generator)
        if leaf == "weight" and len(shape) >= 2:
            bound = math.prod(shape[1:]) ** -0.5
            return values.uniform_(-bound, bound, generator=generator)
        raise ValueError(f"no catalog initializer for {name}")

    def rotary(self, name, start, stop):
        width = getattr(self.config, "rotary_width", self.config.head_dim)
        frequency = 1 / self.config.rope_base ** (
            torch.arange(0, width, 2, dtype=torch.float32) / width
        )
        scaling = getattr(self.config, "rope_scaling", None)
        if scaling is not None:
            frequency = llama3_frequencies(frequency, scaling)
        indices = torch.arange(start, stop)
        phase = (indices // width).float() * frequency[indices % (width // 2)]
        return phase.cos_() if name.endswith("cosine") else phase.sin_()
