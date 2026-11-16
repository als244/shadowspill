"""Deterministic small-model initialization directly into bounded CPU chunks."""

from hashlib import sha256

import torch

from shadowspill.checkpoint.initialization import validate_chunk


def fill_tensor(request, destination, *, seed=0):
    shapes = {
        "network.0.weight": (32, 16),
        "network.0.bias": (32,),
        "network.2.weight": (8, 32),
        "network.2.bias": (8,),
        "updates": (),
    }
    if request.name not in shapes or request.shape != shapes[request.name]:
        raise ValueError(f"unknown regression tensor geometry: {request.name}")
    validate_chunk(request, destination)
    if request.name == "updates":
        destination.zero_()
        return
    # One fixed RNG block per small example tensor; offsets make all ranks and
    # callback chunk sizes reconstruct exactly the same initialization. Seed
    # zero preserves the original recipe's parameter-name seeds byte for byte.
    key = request.name if seed == 0 else f"{seed}:{request.name}"
    generator = torch.Generator().manual_seed(
        int.from_bytes(sha256(key.encode()).digest()[:8], "little")
    )
    inputs = 16 if request.name.startswith("network.0.") else 32
    values = torch.empty(request.storage_elements).uniform_(
        -(inputs**-0.5), inputs**-0.5, generator=generator
    )
    destination.copy_(values[request.offset : request.offset + destination.numel()])
