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

from shadowspill._status import ABI_VERSION

from ._capi import (
    CLeaseLifetime,
    CPlacementProblem,
    CPlacementResult,
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
) -> tuple[tuple[int, ...], int]:
    """Place the first `count` records of an array the library already holds.

    This is the path a measurement takes: the lifetime pass leaves the fixed
    leases in a contiguous prefix, so placement runs on them where they lie.
    """

    if count == 0:
        return (), 0
    offsets = (ctypes.c_uint64 * count)()
    problem = CPlacementProblem(
        abi_version=ABI_VERSION, lifetime_count=count, lifetimes=records
    )
    result = CPlacementResult(required_bytes=0, offsets=offsets)
    _check(
        planner_api().shadowspill_place_lifetimes(
            ctypes.byref(problem), ctypes.byref(result)
        )
    )
    return tuple(offsets[:]), int(result.required_bytes)


def place_lifetimes(
    lifetimes: Sequence[Lifetime],
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
    return place_records((CLeaseLifetime * count).from_buffer(buffer), count)


def _check(status: int) -> None:
    if int(status) != 0:
        raise RuntimeError(f"lease placement failed with planner status {status}")


__all__ = ["Lifetime", "place_lifetimes", "place_records"]
