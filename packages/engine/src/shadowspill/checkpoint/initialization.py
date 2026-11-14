"""Bounded random generation independent of runtime chunks and rank ownership."""

import math
from collections.abc import Callable
from hashlib import sha256

import numpy as np
import torch

from shadowspill.artifacts.tensors import TensorChunk


def validate_chunk(request: TensorChunk, destination: torch.Tensor) -> None:
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


def fill_generated(
    request: TensorChunk,
    destination: torch.Tensor,
    *,
    sample: Callable[[str, tuple[int, ...], int, int, torch.Generator], torch.Tensor],
    seed: int,
    block_elements: int = 256 << 10,
    constant: float | None = None,
) -> None:
    """Generate fixed RNG blocks, then copy only the requested intersection.

    ``sample(name, shape, start, stop, generator)`` supplies the model's values.
    ``constant`` fills the destination directly instead of sampling blocks.
    Random blocks are copied in batches of sixteen; their seeds and values
    remain independent of batching, shard boundaries and loader chunk sizes.
    Default FP32 scratch stays within 32 MiB plus one RNG block, even for a
    multi-GiB expert bank. No process-wide thread settings are changed.
    """
    validate_chunk(request, destination)
    if block_elements <= 0:
        raise ValueError("block_elements must be positive")
    if constant is not None:
        # Match the FP32 sampler's rounding before conversion to storage dtype.
        destination.fill_(torch.tensor(constant, dtype=torch.float32).item())
        return
    end = request.offset + destination.numel()
    pending: list[torch.Tensor] = []
    copied = 0

    def flush() -> None:
        nonlocal copied
        if not pending:
            return
        if len(pending) == 1:
            merged = pending[0]
        else:
            # torch.cat launches CPU work for every small input. Join their
            # unchanged bytes first, then convert/copy one larger destination.
            # Byte views also support BF16, which NumPy cannot represent itself.
            merged = torch.from_numpy(
                np.concatenate([value.view(torch.uint8).numpy() for value in pending])
            ).view(pending[0].dtype)
        destination[copied : copied + merged.numel()].copy_(merged)
        copied += merged.numel()
        pending.clear()

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
        if pending and values.dtype != pending[0].dtype:
            flush()
        pending.append(values[left - start : right - start].contiguous())
        if len(pending) == 16:
            flush()
    flush()
