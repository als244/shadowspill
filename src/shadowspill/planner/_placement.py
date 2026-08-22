"""Fixed-offset lease placement through the compiled planner library."""

from __future__ import annotations

import ctypes
from collections.abc import Sequence

from ._capi import (
    ABI_VERSION,
    CPlacementProblem,
    CPlacementResult,
    load_planner_library,
)

#: One lease as ``(bytes, alignment, predicted_start_ns, predicted_end_ns,
#: lease_id)``. Lifetimes are half-open, so leases whose intervals merely touch
#: may share an offset.
PlacementItem = tuple[int, int, int, int, int]


def place_lifetimes(
    items: Sequence[PlacementItem],
) -> tuple[tuple[int, ...], int]:
    """Return each lease's offset, in input order, and the bytes required."""

    count = len(items)
    if count == 0:
        return (), 0

    columns = tuple(
        (ctypes.c_uint64 * count)(*column) for column in zip(*items, strict=True)
    )
    offsets = (ctypes.c_uint64 * count)()
    problem = CPlacementProblem(
        abi_version=ABI_VERSION,
        lifetime_count=count,
        bytes=columns[0],
        alignment=columns[1],
        predicted_start_ns=columns[2],
        predicted_end_ns=columns[3],
        lease_id=columns[4],
    )
    result = CPlacementResult(required_bytes=0, offsets=offsets)
    status = int(
        load_planner_library().shadowspill_place_lifetimes(
            ctypes.byref(problem), ctypes.byref(result)
        )
    )
    if status != 0:
        raise RuntimeError(f"lease placement failed with planner status {status}")
    return tuple(offsets), int(result.required_bytes)


__all__ = ["PlacementItem", "place_lifetimes"]
