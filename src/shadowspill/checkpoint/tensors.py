"""Physical tensor geometry and bounded, zero-copy CPU byte views."""

from typing import Any

import torch

from shadowspill.execution.tensors import TensorChunk


def geometry(chunk: TensorChunk) -> dict[str, Any]:
    return {
        "name": chunk.name,
        "shape": list(chunk.shape),
        "dtype": str(chunk.dtype),
        "stride": list(chunk.stride),
        "storage_offset": chunk.storage_offset,
        "storage_elements": chunk.storage_elements,
    }


def byte_view(tensor: torch.Tensor) -> memoryview:
    if tensor.device.type != "cpu" or not tensor.is_contiguous() or tensor.ndim != 1:
        raise ValueError(
            "checkpoint chunks must be contiguous one-dimensional CPU views"
        )
    return memoryview(tensor.detach().view(torch.uint8).numpy())
