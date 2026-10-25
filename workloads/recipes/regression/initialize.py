"""Deterministic small-model initialization directly into bounded CPU chunks."""

from hashlib import sha256

import torch


def fill_tensor(request, destination):
    if request.name == "updates":
        destination.zero_()
        return
    if request.name not in {
        "network.0.weight",
        "network.0.bias",
        "network.2.weight",
        "network.2.bias",
    }:
        raise ValueError(f"unknown regression tensor: {request.name}")
    # One fixed RNG block per small example tensor; offsets make all ranks and
    # callback chunk sizes reconstruct exactly the same initialization.
    seed = int.from_bytes(sha256(request.name.encode()).digest()[:8], "little")
    generator = torch.Generator().manual_seed(seed)
    inputs = 16 if request.name.startswith("network.0.") else 32
    values = torch.empty(request.storage_elements).uniform_(
        -(inputs**-0.5), inputs**-0.5, generator=generator
    )
    destination.copy_(values[request.offset : request.offset + destination.numel()])
