"""Conservative slab replay for one selected PyTorch execution schedule."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import IntEnum

from shadowspill.ir import MemoryActionKind, MemoryLocation
from shadowspill.planner import PressureFitResult
from shadowspill.runtime import (
    AllocationEvent,
    AllocationOperation,
    SlabReplay,
    replay_slab_timeline,
)
from shadowspill.simulator import TransferDirection

from .profiling import TaskMeasurement


class _SpatialEventKind(IntEnum):
    FREE_ALIAS = 0
    FREE_WORKSPACE = 1
    ALLOCATE_PREFETCH = 2
    ALLOCATE_WORKSPACE = 3
    ALLOCATE_OUTPUT = 4


@dataclass(frozen=True, slots=True)
class _SpatialEvent:
    time_ns: int
    sequence: int
    kind: _SpatialEventKind
    identity: str
    bytes: int


def replay_selected_schedule(
    selected: PressureFitResult,
    measurements: Mapping[str, TaskMeasurement],
    *,
    slab_bytes: int,
    alignment: int = 256,
) -> SlabReplay:
    """Replay selected object lifetimes and profiled task workspace spatially.

    Task outputs are ordinary framework allocations and therefore use the
    anonymous best-fit/high-address policy. Prefetch destinations use planned
    low-address placement. Workspace peak extents are conservatively live
    together for their task interval.
    """

    program = selected.program
    alias_size = {
        item.alias_group_id: max(1, item.size_bytes) for item in program.alias_groups
    }
    alias_by_object = {item.object_id: item.alias_group_id for item in program.objects}
    profile_by_id = {item.profile_id: item for item in program.profiles}
    task_by_id = {
        item.task_id: item for item in program.selected_tasks(selected.selections)
    }
    interval_by_task = {
        item.task_id: item for item in selected.simulation.task_intervals
    }
    events: list[_SpatialEvent] = []
    sequence = 0

    def append(
        time_ns: int,
        kind: _SpatialEventKind,
        identity: str,
        bytes_: int,
    ) -> None:
        nonlocal sequence
        events.append(_SpatialEvent(time_ns, sequence, kind, identity, bytes_))
        sequence += 1

    for residency in selected.schedule.initial_residency:
        if residency.location is MemoryLocation.DEVICE:
            append(
                0,
                _SpatialEventKind.ALLOCATE_PREFETCH,
                residency.alias_group_id,
                alias_size[residency.alias_group_id],
            )

    for task_id, task in task_by_id.items():
        interval = interval_by_task[task_id]
        output_aliases = tuple(
            dict.fromkeys(alias_by_object[object_id] for object_id in task.outputs)
        )
        for alias_id in output_aliases:
            append(
                interval.start_ns,
                _SpatialEventKind.ALLOCATE_OUTPUT,
                alias_id,
                alias_size[alias_id],
            )
        profile = profile_by_id[task.profile_id]
        try:
            measurement = measurements[profile.compatibility_digest]
        except KeyError as exc:
            raise ValueError(
                "spatial admission lacks task measurement "
                f"{profile.compatibility_digest!r}"
            ) from exc
        extents = list(measurement.workspace_extent_bytes)
        unclassified = profile.workspace_bytes - sum(extents)
        if unclassified < 0:
            raise ValueError("profile workspace extents exceed charged workspace")
        if unclassified:
            extents.append(unclassified)
        for ordinal, bytes_ in enumerate(extents):
            identity = f"workspace:{task_id}:{ordinal}"
            append(
                interval.start_ns,
                _SpatialEventKind.ALLOCATE_WORKSPACE,
                identity,
                max(1, bytes_),
            )
            append(
                interval.end_ns,
                _SpatialEventKind.FREE_WORKSPACE,
                identity,
                max(1, bytes_),
            )

    transfer_keys: set[tuple[str, str, TransferDirection]] = set()
    for transfer in selected.simulation.transfer_intervals:
        key = (
            transfer.trigger_task_id,
            transfer.alias_group_id,
            transfer.direction,
        )
        if key in transfer_keys:
            raise ValueError("spatial admission found a duplicate transfer interval")
        transfer_keys.add(key)
        if transfer.direction is TransferDirection.HOST_TO_DEVICE:
            append(
                transfer.start_ns,
                _SpatialEventKind.ALLOCATE_PREFETCH,
                transfer.alias_group_id,
                alias_size[transfer.alias_group_id],
            )
        else:
            append(
                transfer.end_ns,
                _SpatialEventKind.FREE_ALIAS,
                transfer.alias_group_id,
                alias_size[transfer.alias_group_id],
            )

    for action in selected.schedule.actions:
        interval = interval_by_task[action.trigger_task_id]
        if action.kind is MemoryActionKind.RELEASE:
            append(
                interval.end_ns,
                _SpatialEventKind.FREE_ALIAS,
                action.alias_group_id,
                alias_size[action.alias_group_id],
            )

    ordered = sorted(
        events,
        key=lambda item: (item.time_ns, int(item.kind), item.sequence),
    )
    live_aliases: dict[str, str] = {}
    generations: dict[str, int] = {}
    allocation_events: list[AllocationEvent] = []
    for position, item in enumerate(ordered):
        if item.kind in {
            _SpatialEventKind.ALLOCATE_PREFETCH,
            _SpatialEventKind.ALLOCATE_OUTPUT,
        }:
            if item.identity in live_aliases:
                continue
            generation = generations.get(item.identity, 0)
            generations[item.identity] = generation + 1
            allocation_id = f"{item.identity}:{generation}"
            live_aliases[item.identity] = allocation_id
            allocation_events.append(
                AllocationEvent(
                    position,
                    allocation_id,
                    AllocationOperation.ALLOCATE,
                    item.bytes,
                    alignment=alignment,
                    planned=item.kind is _SpatialEventKind.ALLOCATE_PREFETCH,
                )
            )
        elif item.kind is _SpatialEventKind.FREE_ALIAS:
            if item.identity not in live_aliases:
                raise ValueError(
                    f"spatial admission frees nonresident alias {item.identity!r}"
                )
            allocation_id = live_aliases.pop(item.identity)
            allocation_events.append(
                AllocationEvent(
                    position,
                    allocation_id,
                    AllocationOperation.FREE,
                    item.bytes,
                    alignment=alignment,
                )
            )
        elif item.kind is _SpatialEventKind.ALLOCATE_WORKSPACE:
            allocation_events.append(
                AllocationEvent(
                    position,
                    item.identity,
                    AllocationOperation.ALLOCATE,
                    item.bytes,
                    alignment=alignment,
                )
            )
        else:
            allocation_events.append(
                AllocationEvent(
                    position,
                    item.identity,
                    AllocationOperation.FREE,
                    item.bytes,
                    alignment=alignment,
                )
            )
    return replay_slab_timeline(slab_bytes, tuple(allocation_events))


__all__ = ["replay_selected_schedule"]
