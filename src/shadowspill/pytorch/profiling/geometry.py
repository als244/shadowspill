"""Writing into a representative value whatever geometry it has."""

from __future__ import annotations

import torch


def distinct_locations(value: torch.Tensor) -> torch.Tensor:
    """A view of this tensor naming each of its locations exactly once.

    A broadcast dimension has a stride of zero: every index along it reads
    the same element. Writing through such a view is refused whatever the
    value -- generated or copied -- because the writes would race for one
    location and what survived would be whichever landed last.

    Collapsing the broadcast dimensions to a single index leaves a view over
    the same storage in which no location appears twice. Writing into that
    is well defined, and what the broadcast then reads is one value repeated
    along its broadcast axes, which is what a broadcast is. Two tensors of
    one geometry collapse to the same elements in the same order, so a copy
    between their collapsed views is exact.

    A tensor whose locations overlap for some other reason is returned
    unchanged, so writing into it still raises rather than being filled by a
    rule that was not written for it.
    """

    strides = value.stride()
    if all(stride != 0 for stride in strides):
        return value
    collapsed = tuple(
        1 if stride == 0 else size
        for size, stride in zip(value.shape, strides, strict=True)
    )
    return value.as_strided(collapsed, strides, value.storage_offset())


__all__ = ["distinct_locations"]
