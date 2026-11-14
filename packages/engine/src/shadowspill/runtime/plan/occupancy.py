"""Describe the allocations blocking worker layout admission."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from shadowspill.runtime.occupancy import allocation_id_for_address, live_allocations

if TYPE_CHECKING:
    from .bridge import RuntimeBridge

_REPORTED_ALLOCATIONS = 24


def _describe_occupants(
    holders: tuple[Any, ...], holding: Mapping[int, tuple[str, ...]]
) -> str:
    """Name what holds a range, and what in turn keeps that holder alive.

    The holder says what the range is in the framework's terms; the retainer
    says where the reference lives, which is the only place it can be dropped.
    """

    if not holders:
        return ", held by no frontend object"
    described = []
    for item in holders[:3]:
        kept = holding.get(id(item), ())
        where = f" via {', '.join(kept[:2])}" if kept else " retained by a library"
        described.append(f"{type(item).__name__}{tuple(item.shape)}{where}")
    more = f" +{len(holders) - 3}" if len(holders) > 3 else ""
    return f", held by {'; '.join(described)}{more}"


def describe_pool_occupants(bridge: RuntimeBridge) -> str:
    """What is occupying the pool, for a layout that could not be placed.

    A refusal for want of a contiguous range is not explained by how many
    allocations are live: a small one in the wrong place costs the largest
    free range and leaves the free total nearly untouched. The offsets are
    the diagnosis, so they are listed in pool order with the scope that made
    each one.
    """

    try:
        held = live_allocations(bridge.runtime)
    except Exception:  # a failure report must not fail
        return ""
    if not held:
        return ""
    shown = held[:_REPORTED_ALLOCATIONS]
    try:
        # Everything in this pool arrived through the framework's allocator,
        # so a range with no frontend object is held by the framework's own
        # internals rather than by anything a caller can drop. Saying which
        # is what separates a reference to release from one to relocate.
        frontend = bridge.runtime.frontend
        holders = frontend.occupants(
            shown,
            lambda address: allocation_id_for_address(bridge.runtime, address),
        )
        occupying = [item for objects in holders.values() for item in objects]
        holding = frontend.retainers(
            occupying, ignore=(holders, occupying, *holders.values())
        )
    except Exception:
        holders = {}
        holding = {}
    lines = "".join(
        f"\n  offset={item.offset} bytes={item.charged_bytes}"
        f" {bridge.objects.role_of(item)} from {item.origin}"
        f"{' scratch' if item.scratch else ''}"
        f"{' planned' if item.plan_owned else ''}"
        f"{' freed-pending-retirement' if item.logical_freed else ''}"
        f"{_describe_occupants(holders.get(item.allocation_id, ()), holding)}"
        for item in shown
    )
    omitted = len(held) - len(shown)
    tail = f"\n  ... {omitted} more" if omitted > 0 else ""
    return f"; the pool is held by:{lines}{tail}"
