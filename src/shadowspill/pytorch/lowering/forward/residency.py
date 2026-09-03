"""Derive initial and final residency for a bound forward task graph."""

from __future__ import annotations

from collections.abc import Mapping

from shadowspill.ir import MemoryLocation, ObjectRole, Persistence, ResidencySpec

from .artifacts import ForwardObjects, ForwardTaskGraph


def derive_forward_residency(
    objects: ForwardObjects,
    graph: ForwardTaskGraph,
    *,
    public_output_locations: Mapping[int, MemoryLocation] | None = None,
) -> tuple[tuple[ResidencySpec, ...], tuple[ResidencySpec, ...]]:
    locations = dict(public_output_locations or {})
    invalid = sorted(set(locations) - set(range(len(graph.public_outputs))))
    if invalid:
        raise ValueError(f"public output residency indices are invalid: {invalid}")
    if any(not isinstance(value, MemoryLocation) for value in locations.values()):
        raise TypeError("public output residency must use MemoryLocation")
    aliases = objects.catalog.alias_groups()
    shared_aliases = {
        group.alias_group_id for group in aliases if group.shared_residency is not None
    }
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
        and item.alias_group_id not in shared_aliases
    }
    final_spill = {
        item.alias_group_id
        for item in logical_objects
        if item.persistence is Persistence.CHECKPOINT
        or item.role in {ObjectRole.INPUT, ObjectRole.CONTROL}
    }
    final_device = {
        objects.catalog.alias_id(object_id)
        for index, object_id in enumerate(graph.public_outputs)
        if locations.get(index, MemoryLocation.DEVICE) is MemoryLocation.DEVICE
    }
    final_spill.update(
        objects.catalog.alias_id(object_id)
        for index, object_id in enumerate(graph.public_outputs)
        if locations.get(index, MemoryLocation.DEVICE) is MemoryLocation.SPILL
    )
    final_spill -= final_device
    final_spill -= shared_aliases
    final_device -= shared_aliases
    return (
        tuple(
            ResidencySpec(group.alias_group_id, MemoryLocation.SPILL)
            for group in aliases
            if group.alias_group_id in initial_aliases
        ),
        tuple(
            ResidencySpec(
                group.alias_group_id,
                MemoryLocation.DEVICE
                if group.alias_group_id in final_device
                else MemoryLocation.SPILL,
            )
            for group in aliases
            if group.alias_group_id in final_spill | final_device
        ),
    )


def finalize_forward_shared_residency(
    objects: ForwardObjects,
    graph: ForwardTaskGraph,
) -> None:
    """Classify shared inputs from the writes in the emitted task graph."""

    objects.catalog.finalize_shared_writes(
        object_id
        for task in graph.tasks
        for object_id in (
            *task.outputs,
            *(mutation.object_id for mutation in task.mutations),
        )
    )


__all__ = ["derive_forward_residency", "finalize_forward_shared_residency"]
