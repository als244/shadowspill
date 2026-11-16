"""Bounded scatter and declared reductions into unpublished tensor files."""

from collections.abc import Iterator, Sequence
from typing import Any, BinaryIO, Literal

import numpy as np
from numpy.typing import NDArray

from shadowspill_client._records import integer
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.inputs.values import _convert
from shadowspill_client.protocol.tensors import storage_dtype, tensor_bytes

from .inputs.frame_reader import FrameRegion
from .inputs.frames import FRAME_CHUNK_BYTES

type Reduction = Literal["sum", "min", "max", "identical"]
REDUCTION_SCRATCH_BYTES = 2 << 20


def _read(stream: BinaryIO, size: int) -> bytearray:
    result, block = bytearray(size), b""
    position = 0
    try:
        while position < size:
            block = stream.read(size - position)
            if not isinstance(block, bytes) or not 0 < len(block) <= size - position:
                raise OSError("unpublished tensor storage became truncated")
            result[position : position + len(block)] = block
            position += len(block)
        return result
    finally:
        # I/O exceptions must not retain an uncharged copy through their traceback.
        del result, block


def _combine(
    previous: bytearray, incoming: memoryview, dtype: str, reduction: Reduction
) -> NDArray[Any] | memoryview:
    left = right = merged = None
    try:
        if reduction == "identical":
            if previous != incoming:
                raise ValueError("declared identical tensor outputs disagree")
            return incoming
        left, right = (
            np.frombuffer(previous, dtype=storage_dtype(dtype)),
            np.frombuffer(incoming, dtype=storage_dtype(dtype)),
        )
        if dtype == "bfloat16":
            left = (left.astype(np.uint32) << 16).view(np.float32)
            right = (right.astype(np.uint32) << 16).view(np.float32)
        operation = {"sum": np.add, "min": np.minimum, "max": np.maximum}[reduction]
        merged = operation(left, right)
        return _convert(merged, dtype)
    finally:
        del left, right, merged, previous, incoming


def _write(
    stream: BinaryIO,
    offset: int,
    data: memoryview,
    dtype: str,
    reduction: Reduction | None,
) -> None:
    previous = merged = view = None
    try:
        stream.seek(offset)
        if reduction is None:
            view = memoryview(data)
        else:
            previous = _read(stream, len(data))
            merged = _combine(previous, data, dtype, reduction)
            view = memoryview(merged).cast("B")
            stream.seek(offset)
        position = 0
        while position < len(view):
            part = view[position:]
            try:
                count = stream.write(part)
            finally:
                part.release()
            if type(count) is not int or not 0 < count <= len(view) - position:
                raise OSError("unpublished tensor write made no progress")
            position += count
    finally:
        if view is not None:
            view.release()
        del previous, merged, view


def write_tensor_region(
    stream: BinaryIO,
    *,
    offset: int,
    shape: tuple[int, ...],
    dtype: str,
    axes: Sequence[int | slice],
    chunks: Iterator[memoryview],
    budget: BufferBudget,
    reduction: Reduction | None = None,
) -> None:
    """Scatter C-ordered borrowed chunks into an admitted unpublished file.

    ``None`` copies the first contribution. Later sums/min/max/identical checks
    operate in the declared dtype, including BF16 rounding after a reduction.
    Cooperating-worker and global-buffer reductions are separate calls chosen
    by OutputLayout. This function adds no normalization or ownership inference.

    The caller owns the destination file and its full disk reservation. A failed
    assembly may leave partial bytes there; it must never publish that file.
    Scratch is fixed regardless of the tensor size; producer views are consumed
    and released before their next advance. The producer always closes on exit.
    """
    try:
        integer(offset, "tensor file offset")
        size = tensor_bytes(dtype, shape)
        region = FrameRegion.select(shape, axes)
        if reduction not in (None, "sum", "min", "max", "identical"):
            raise ValueError("unsupported tensor reduction")
        itemsize = storage_dtype(dtype).itemsize
        with budget.reserve(REDUCTION_SCRATCH_BYTES, timeout=0):
            if stream.seek(0, 2) < offset + size:
                raise ValueError("tensor destination exceeds its admitted file")
            spans = iter(region.spans())
            span = next(spans, None)
            position = 0
            for chunk in chunks:
                try:
                    if (
                        not isinstance(chunk, memoryview)
                        or chunk.ndim != 1
                        or chunk.itemsize != 1
                        or not chunk.c_contiguous
                        or not 0 < chunk.nbytes <= FRAME_CHUNK_BYTES
                        or chunk.nbytes % itemsize
                    ):
                        raise ValueError("tensor producer exceeded its byte bound")
                    consumed = 0
                    while consumed < len(chunk):
                        if span is None:
                            raise ValueError("tensor producer exceeds its destination")
                        _, physical, elements = span
                        count = min(
                            len(chunk) - consumed, elements * itemsize - position
                        )
                        part = chunk[consumed : consumed + count]
                        try:
                            _write(
                                stream,
                                offset + physical * itemsize + position,
                                part,
                                dtype,
                                reduction,
                            )
                        finally:
                            part.release()
                        consumed += count
                        position += count
                        if position == elements * itemsize:
                            span, position = next(spans, None), 0
                finally:
                    if isinstance(chunk, memoryview):
                        chunk.release()
                    del chunk
            if span is not None:
                raise ValueError("tensor producer ended before its destination")
    finally:
        close = getattr(chunks, "close", None)
        if close is not None:
            close()
