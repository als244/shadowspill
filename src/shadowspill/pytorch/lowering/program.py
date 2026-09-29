"""Publish a canonical ShadowSpillProgram from one mode's bound task graph."""

from __future__ import annotations

from shadowspill.ir import (
    DeviceSpec,
    ResourceKind,
    ResourceSpec,
    ShadowSpillProgram,
    TaskAlternativeGroup,
    TaskSpec,
)
from shadowspill.pytorch.accelerator import DEVICE_TYPE

from .catalog import ObjectCatalog
from .profiles import TaskProfileCatalog


def invocation_start(profiles: TaskProfileCatalog, device_id: str) -> TaskSpec:
    """An ordinary control task that can trigger fetches before computation."""

    return TaskSpec(
        "task_000000",
        ResourceSpec(device_id, ResourceKind.CONTROL),
        profiles.control_profile_id(),
        requires_entrypoint=False,
        phase="control",
    )


def execution_device_id(device_ordinal: int) -> str:
    """Return the canonical IR identity for the selected execution device."""

    return f"{DEVICE_TYPE}_{device_ordinal}"


def publish_program(
    catalog: ObjectCatalog,
    profiles: TaskProfileCatalog,
    tasks: tuple[TaskSpec, ...],
    *,
    device_ordinal: int,
    task_alternative_groups: tuple[TaskAlternativeGroup, ...] = (),
) -> ShadowSpillProgram:
    """Freeze shared object, profile, and task inventories into canonical IR."""

    device_id = execution_device_id(device_ordinal)
    return ShadowSpillProgram(
        devices=(DeviceSpec(device_id, "process_0", DEVICE_TYPE, device_ordinal),),
        alias_groups=catalog.alias_groups(),
        objects=catalog.objects(),
        profiles=profiles.profiles,
        tasks=tasks,
        task_alternative_groups=task_alternative_groups,
    )


def publish_storage_program(
    catalog: ObjectCatalog,
    *,
    device_ordinal: int,
) -> ShadowSpillProgram:
    """Freeze the pre-optimizer model/input storage inventory."""

    device_id = execution_device_id(device_ordinal)
    return ShadowSpillProgram(
        devices=(DeviceSpec(device_id, "process_0", DEVICE_TYPE, device_ordinal),),
        alias_groups=catalog.alias_groups(),
        objects=catalog.objects(),
        profiles=(),
        tasks=(),
    )


__all__ = ["execution_device_id", "publish_program", "publish_storage_program"]
