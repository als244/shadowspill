"""Bounded C-order host copies without a full contiguous device temporary."""

from collections.abc import Iterator, Sequence
from math import prod

import torch


def storage_runs(
    shape: Sequence[int], stride: Sequence[int], *, max_elements: int
) -> Iterator[tuple[int, int]]:
    """Yield relative storage offsets and contiguous run lengths in C order.

    Prefix coordinates are decoded arithmetically; no Cartesian-product cache or
    full tensor index array is allocated. Broadcast and strided views are valid.
    """
    if (
        len(shape) != len(stride)
        or len(shape) > 32
        or any(type(n) is not int or n < 0 for n in (*shape, *stride))
        or type(max_elements) is not int
        or max_elements < 1
    ):
        raise ValueError(
            "tensor copy requires bounded shape/strides and a positive chunk"
        )
    if 0 in shape:
        return
    suffix, contiguous = len(shape), 1
    while suffix and (shape[suffix - 1] == 1 or stride[suffix - 1] == contiguous):
        suffix -= 1
        contiguous *= shape[suffix]
    for index in range(prod(shape[:suffix])):
        offset = 0
        for axis in range(suffix - 1, -1, -1):
            index, position = divmod(index, shape[axis])
            offset += position * stride[axis]
        for start in range(0, contiguous, max_elements):
            yield offset + start, min(max_elements, contiguous - start)


def host_chunks(tensor: torch.Tensor, *, chunk_bytes: int) -> Iterator[memoryview]:
    """Borrow one CPU scratch block at a time, synchronously ready for file I/O.

    The caller reserves chunk_bytes before iterating and owns the tensor through
    completion. Each yielded view expires on the next iteration. Only contiguous
    source runs cross devices, so no device reshape/packing allocation is needed.
    """
    if (
        tensor.layout != torch.strided
        or tensor.device.type not in {"cpu", "cuda"}
        or tensor.is_conj()
        or tensor.is_neg()
        or type(chunk_bytes) is not int
        or chunk_bytes < tensor.element_size()
    ):
        raise ValueError(
            "tensor copy requires plain CPU/CUDA storage and a valid chunk"
        )
    if not tensor.numel():
        return
    # One fixed-size allocation also avoids filling the pinned-host cache with
    # a succession of differently sized buffers across tensors and requests.
    count = chunk_bytes // tensor.element_size()
    host = torch.empty(
        count, dtype=tensor.dtype, device="cpu", pin_memory=tensor.is_cuda
    )
    source = tensor.detach()
    try:
        for offset, length in storage_runs(
            tuple(source.shape), source.stride(), max_elements=count
        ):
            view = source.as_strided((length,), (1,), source.storage_offset() + offset)
            try:
                host[:length].copy_(view, non_blocking=False)
            finally:
                del view
            # Byte views also preserve BF16 without a NumPy BF16 dependency.
            yield memoryview(host[:length].view(torch.uint8).numpy()).cast("B")
    finally:
        del host, source
