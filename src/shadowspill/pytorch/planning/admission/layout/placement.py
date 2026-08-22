"""Deterministic interval placement for a fixed execution-pool slice."""

from __future__ import annotations

from shadowspill.planner._placement import place_lifetimes as _place_offsets

from .model import FixedLayoutPlacement, LeaseLifetime


def place_lifetimes(
    lifetimes: tuple[LeaseLifetime, ...],
) -> tuple[tuple[FixedLayoutPlacement, ...], int]:
    """Place larger and longer-lived leases first, then choose lowest fit."""

    offsets, required = _place_offsets(
        [
            (
                item.bytes,
                item.alignment,
                item.predicted_start_ns,
                item.predicted_end_ns,
                item.lease_id,
            )
            for item in lifetimes
        ]
    )
    placements = tuple(
        FixedLayoutPlacement(
            lease_id=item.lease_id,
            offset=offsets[index],
            bytes=item.bytes,
            alignment=item.alignment,
            predicted_start_ns=item.predicted_start_ns,
            predicted_end_ns=item.predicted_end_ns,
            causal_start=item.causal_start,
            causal_end=item.causal_end,
            purpose=item.purpose,
            task_id=item.task_id,
            alias_group_id=item.alias_group_id,
            action_index=item.action_index,
        )
        for index, item in enumerate(lifetimes)
    )
    return placements, required


__all__ = ["place_lifetimes"]
