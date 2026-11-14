"""Physical tensor geometry and bounded, zero-copy CPU byte views."""

from typing import Any

import torch

from shadowspill.artifacts.tensors import TensorChunk


def geometry(chunk: TensorChunk) -> dict[str, Any]:
    return {
        "name": chunk.name,
        "shape": list(chunk.shape),
        "dtype": str(chunk.dtype),
        "stride": list(chunk.stride),
        "storage_offset": chunk.storage_offset,
        "storage_elements": chunk.storage_elements,
        **(
            {
                "region_offset": chunk.region_offset,
                "region_elements": chunk.region_elements,
            }
            if chunk.region_elements is not None
            and (
                chunk.region_offset != 0
                or chunk.region_elements != chunk.storage_elements
            )
            else {}
        ),
    }


def region_bytes(chunk: TensorChunk) -> int:
    elements = (
        chunk.storage_elements
        if chunk.region_elements is None
        else chunk.region_elements
    )
    return elements * chunk.dtype.itemsize


def byte_view(tensor: torch.Tensor) -> memoryview:
    if tensor.device.type != "cpu" or not tensor.is_contiguous() or tensor.ndim != 1:
        raise ValueError(
            "checkpoint chunks must be contiguous one-dimensional CPU views"
        )
    return memoryview(tensor.detach().view(torch.uint8).numpy())
