"""Reproducible inputs shared by all backend adapters."""

import torch


def tensor(shape, *, seed, std=0.02, dtype=torch.bfloat16):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return torch.empty(shape, device="cuda", dtype=dtype).normal_(
        std=std, generator=generator
    )


def expert_tensor(shape, *, seed, slot, first_expert=0, std=0.02):
    """Initialize by global expert ID, independent of EP partition and backend."""
    values = torch.empty(shape, device="cuda", dtype=torch.bfloat16)
    for local_id, value in enumerate(values):
        generator = torch.Generator(device="cuda").manual_seed(
            seed + 10000 + 16 * (first_expert + local_id) + slot
        )
        value.normal_(std=std, generator=generator)
    return values
