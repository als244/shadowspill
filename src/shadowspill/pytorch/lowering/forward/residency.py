"""Derive initial and final residency for a bound forward task graph."""

from __future__ import annotations

from shadowspill.ir import MemoryLocation, ObjectRole, Persistence, ResidencySpec

from .artifacts import ForwardObjects, ForwardTaskGraph


def derive_forward_residency(
    objects: ForwardObjects,
    graph: ForwardTaskGraph,
) -> tuple[tuple[ResidencySpec, ...], tuple[ResidencySpec, ...]]:
    aliases = objects.catalog.alias_groups()
    logical_objects = objects.catalog.objects()
    initial_aliases = {
        item.alias_group_id
        for item in logical_objects
        if item.role
        in {
            ObjectRole.PARAMETER,
            ObjectRole.BUFFER,
            ObjectRole.INPUT,
            ObjectRole.CONTROL,
        }
        and item.alias_group_id not in graph.produced_aliases
    }
    final_host = {
        item.alias_group_id
        for item in logical_objects
        if item.persistence is Persistence.CHECKPOINT
        or item.role in {ObjectRole.INPUT, ObjectRole.CONTROL}
    }
    final_device = {
        objects.catalog.alias_id(object_id) for object_id in graph.public_outputs
    }
    final_host -= final_device
    return (
        tuple(
            ResidencySpec(group.alias_group_id, MemoryLocation.HOST)
            for group in aliases
            if group.alias_group_id in initial_aliases
        ),
        tuple(
            ResidencySpec(
                group.alias_group_id,
                MemoryLocation.DEVICE
                if group.alias_group_id in final_device
                else MemoryLocation.HOST,
            )
            for group in aliases
            if group.alias_group_id in final_host | final_device
        ),
    )


__all__ = ["derive_forward_residency"]
