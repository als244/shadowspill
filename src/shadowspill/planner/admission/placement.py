"""Fixed-offset lease placement through the library.

Placement needs four numbers per lease and nothing else, so that is the whole
contract: `Lifetime` names them, and anything carrying them can be placed. The
entry point never sees lease identity — offsets come back in input
order, and the input index breaks every tie.
"""

from __future__ import annotations

import array
import ctypes
from collections.abc import Sequence
from typing import Protocol

import numpy as np

from shadowspill.status import ABI_VERSION

from ..capi import (
    CLeaseLifetime,
    CPlacementProblem,
    CPlacementResult,
    check_planner_status,
    planner_api,
)


class Lifetime(Protocol):
    """What placement reads from one lease.

    The interval is half-open, so leases whose lifetimes merely touch are not
    live at the same moment and may share an offset.
    """

    @property
    def bytes(self) -> int: ...

    @property
    def alignment(self) -> int: ...

    @property
    def predicted_start_ns(self) -> int: ...

    @property
    def predicted_end_ns(self) -> int: ...


def place_records(
    records: ctypes.Array[CLeaseLifetime],
    count: int,
    excluded: Sequence[bool] | None = None,
) -> tuple[tuple[int, ...], int]:
    """Place the first `count` records of an array the library already holds.

    `excluded` marks records to leave out -- their offsets come back as zero
    and they are outside the bytes required -- for leases placed elsewhere.
    """

    if count == 0:
        return (), 0
    offsets, required = place_record_array(
        records,
        count,
        None if excluded is None else np.fromiter(excluded, dtype=bool, count=count),
    )
    return tuple(offsets.tolist()), required


def place_record_array(
    records: ctypes.Array[CLeaseLifetime],
    count: int,
    excluded: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    """`place_records` with arrays in and out, for a caller holding many.

    This is the path a measurement takes: the lifetime pass leaves the fixed
    leases in a contiguous prefix, so placement runs on them where they lie,
    and the offsets come back as one writable array.
    """

    offsets = np.zeros(count, dtype=np.uint64)
    if count == 0:
        return offsets, 0
    mask = None
    if excluded is not None:
        flags = np.ascontiguousarray(excluded, dtype=np.uint8)
        mask = flags.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    problem = CPlacementProblem(
        abi_version=ABI_VERSION,
        lifetime_count=count,
        lifetimes=records,
        excluded=mask,
    )
    result = CPlacementResult(
        required_bytes=0,
        offsets=offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
    )
    check_planner_status(
        planner_api().shadowspill_place_lifetimes(
            ctypes.byref(problem), ctypes.byref(result)
        ),
        "lease placement",
    )
    return offsets, int(result.required_bytes)


def place_lifetimes(
    lifetimes: Sequence[Lifetime],
    excluded: Sequence[bool] | None = None,
) -> tuple[tuple[int, ...], int]:
    """Place anything carrying the four numbers, copying it across first.

    Callers that already hold the library's own records should use
    `place_records`, which skips the copy.
    """

    count = len(lifetimes)
    if count == 0:
        return (), 0

    # One flat buffer viewed as the record array the library expects, filled in
    # a single pass. `from_buffer` borrows it, so `records` keeps it alive.
    buffer = array.array(
        "Q",
        [
            field
            for item in lifetimes
            for field in (
                item.bytes,
                item.alignment,
                item.predicted_start_ns,
                item.predicted_end_ns,
            )
        ],
    )
    return place_records((CLeaseLifetime * count).from_buffer(buffer), count, excluded)


__all__ = ["Lifetime", "place_lifetimes", "place_records"]
