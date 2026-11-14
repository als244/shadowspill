"""Contiguous portions of an arbitrary strided tensor, in logical element order."""

from collections.abc import Iterator, Sequence
from math import prod


def spans(
    shape: Sequence[int],
    stride: Sequence[int],
    offset: int = 0,
) -> Iterator[tuple[int, int, int]]:
    """Yield (logical element offset, storage element offset, element count)."""
    count, axis = 1, len(shape) - 1
    while axis >= 0 and (shape[axis] <= 1 or stride[axis] == count):
        count *= shape[axis]
        axis -= 1
    if axis < 0:
        if count:
            yield 0, offset, count
        return
    inner = prod(shape[1:])
    for index in range(shape[0]):
        for logical, physical, length in spans(
            shape[1:], stride[1:], offset + index * stride[0]
        ):
            yield index * inner + logical, physical, length
