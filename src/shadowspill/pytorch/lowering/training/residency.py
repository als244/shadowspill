"""Derive initial and final residency for a bound training task graph."""

from __future__ import annotations

from shadowspill.ir import (
    MemoryLocation,
    ObjectSpec,
    ResidencySpec,
    TaskSpec,
)

from .artifacts import TrainingBoundaries, TrainingObjects


def derive_training_residency(
    objects: TrainingObjects,
    boundaries: TrainingBoundaries,
    tasks: tuple[TaskSpec, ...],
) -> tuple[tuple[ResidencySpec, ...], tuple[ResidencySpec, ...]]:
    catalog_objects = objects.catalog.objects()
    aliases = objects.catalog.alias_groups()
    alias_by_object = {item.object_id: item.alias_group_id for item in catalog_objects}
    parameter_aliases = {
        alias_by_object[binding.parameter_object_id] for binding in objects.gradients
    }
    input_aliases = _external_input_aliases(
        catalog_objects,
        tasks,
        parameter_aliases,
    )
    public_aliases = {
        alias_by_object[object_id]
        for values in boundaries.public_outputs.values()
        for object_id in values
    }
    optimizer_aliases = {
        alias_by_object[binding.object_id] for binding in objects.optimizer_objects
    }
    initial = tuple(
        ResidencySpec(item.alias_group_id, MemoryLocation.SPILL)
        for item in aliases
        if item.alias_group_id in parameter_aliases | input_aliases
    )
    final = tuple(
        ResidencySpec(
            item.alias_group_id,
            MemoryLocation.DEVICE
            if item.alias_group_id in public_aliases
            else MemoryLocation.SPILL,
        )
        for item in aliases
        if item.alias_group_id
        in parameter_aliases | input_aliases | public_aliases | optimizer_aliases
    )
    return initial, final


def _external_input_aliases(
    objects: tuple[ObjectSpec, ...],
    tasks: tuple[TaskSpec, ...],
    parameter_aliases: set[str],
) -> set[str]:
    """Return input storage bundles that no task in the program produces."""

    alias_by_object = {item.object_id: item.alias_group_id for item in objects}
    produced_aliases = {
        alias_by_object[object_id] for task in tasks for object_id in task.outputs
    }
    return {
        alias_by_object[object_id]
        for task in tasks
        for object_id in task.inputs
        if alias_by_object[object_id] not in produced_aliases | parameter_aliases
    }


__all__ = ["derive_training_residency"]
