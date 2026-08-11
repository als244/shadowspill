"""Conservative slab replay for one selected PyTorch execution schedule."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import IntEnum

from shadowspill.ir import MemoryActionKind, MemoryLocation, TaskSpec
from shadowspill.planner import PressureFitResult
from shadowspill.runtime import (
    AllocationEvent,
    AllocationOperation,
    SlabReplay,
    replay_slab_timeline,
)
from shadowspill.simulator import TransferDirection

from .lowering import TaskEntrypoint
from .profiling import TaskAllocationOperation, TaskMeasurement
from .training_lowering import TrainingTaskEntrypoint


class _SpatialEventKind(IntEnum):
    FREE_ALIAS = 0
    FREE_TEMPORARY_OUTPUT = 1
    ALLOCATE_PREFETCH = 2
    TASK_ALLOCATION = 3
    TASK_FREE = 4
    TASK_REUSE = 5


@dataclass(frozen=True, slots=True)
class _SpatialEvent:
    time_ns: int
    sequence: int
    kind: _SpatialEventKind
    identity: str
    bytes: int
    planned: bool = False
    alias_output: bool = False
    source_identity: str | None = None


@dataclass(frozen=True, slots=True)
class TaskOutputBinding:
    """Map one returned tensor leaf to its persistent Program alias group."""

    leaf_index: int
    alias_group_id: str

    def __post_init__(self) -> None:
        if self.leaf_index < 0:
            raise ValueError("output leaf index must be non-negative")
        if not self.alias_group_id:
            raise ValueError("output alias group ID must be non-empty")


def replay_selected_schedule(
    selected: PressureFitResult,
    measurements: Mapping[str, TaskMeasurement],
    *,
    slab_bytes: int,
    alignment: int = 256,
    output_bindings: Mapping[str, tuple[TaskOutputBinding, ...]] | None = None,
) -> SlabReplay:
    """Replay selected object lifetimes and profiled task workspace spatially.

    Task allocations replay the exact profiled callback sequence through the
    ordinary best-fit/high-address policy. Prefetch destinations use planned
    low-address placement. Returned tensor leaves identify allocations that
    survive as Program objects; unbound returned tensors are task temporaries.
    """

    program = selected.program
    alias_size = {
        item.alias_group_id: max(1, item.size_bytes) for item in program.alias_groups
    }
    alias_by_object = {item.object_id: item.alias_group_id for item in program.objects}
    object_size = {item.object_id: item.size_bytes for item in program.objects}
    profile_by_id = {item.profile_id: item for item in program.profiles}
    task_by_id = {
        item.task_id: item for item in program.selected_tasks(selected.selections)
    }
    interval_by_task = {
        item.task_id: item for item in selected.simulation.task_intervals
    }
    events: list[_SpatialEvent] = []
    sequence = 0
    bindings_by_task = dict(output_bindings or {})

    def append(
        time_ns: int,
        kind: _SpatialEventKind,
        identity: str,
        bytes_: int,
        *,
        planned: bool = False,
        alias_output: bool = False,
        source_identity: str | None = None,
    ) -> None:
        nonlocal sequence
        events.append(
            _SpatialEvent(
                time_ns,
                sequence,
                kind,
                identity,
                bytes_,
                planned,
                alias_output,
                source_identity,
            )
        )
        sequence += 1

    for residency in selected.schedule.initial_residency:
        if residency.location is MemoryLocation.DEVICE:
            append(
                0,
                _SpatialEventKind.ALLOCATE_PREFETCH,
                residency.alias_group_id,
                alias_size[residency.alias_group_id],
                planned=True,
            )

    for task_id, task in task_by_id.items():
        interval = interval_by_task[task_id]
        profile = profile_by_id[task.profile_id]
        try:
            measurement = measurements[profile.compatibility_digest]
        except KeyError as exc:
            raise ValueError(
                "spatial admission lacks task measurement "
                f"{profile.compatibility_digest!r}"
            ) from exc
        task_bindings = bindings_by_task.get(task_id, ())
        bound_output_aliases = {item.alias_group_id for item in task_bindings}
        output_aliases = tuple(
            dict.fromkeys(alias_by_object[object_id] for object_id in task.outputs)
        )
        for alias_id in output_aliases:
            if alias_id in bound_output_aliases:
                continue
            append(
                interval.start_ns,
                _SpatialEventKind.TASK_ALLOCATION,
                alias_id,
                alias_size[alias_id],
                alias_output=True,
            )
        task_events = _task_allocation_events(
            task,
            measurement,
            task_bindings,
            alias_size,
        )
        for event in task_events:
            append(
                interval.start_ns,
                event.kind,
                event.identity,
                event.bytes,
                alias_output=event.alias_output,
                source_identity=event.source_identity,
            )
        _validate_profile_workspace(
            task, profile.workspace_bytes, measurement, object_size
        )
        for event in task_events:
            if event.kind not in {
                _SpatialEventKind.TASK_ALLOCATION,
                _SpatialEventKind.TASK_REUSE,
            }:
                continue
            if not event.identity.startswith(f"temporary-output:{task_id}:"):
                continue
            append(
                interval.end_ns,
                _SpatialEventKind.FREE_TEMPORARY_OUTPUT,
                event.identity,
                event.bytes,
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
                planned=True,
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
        key=lambda item: (
            item.time_ns,
            3
            if item.kind
            in {
                _SpatialEventKind.TASK_ALLOCATION,
                _SpatialEventKind.TASK_FREE,
                _SpatialEventKind.TASK_REUSE,
            }
            else int(item.kind),
            item.sequence,
        ),
    )
    live_aliases: dict[str, str] = {}
    generations: dict[str, int] = {}
    allocation_events: list[AllocationEvent] = []
    for position, item in enumerate(ordered):
        if item.kind is _SpatialEventKind.ALLOCATE_PREFETCH:
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
                    planned=True,
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
        elif item.kind in {
            _SpatialEventKind.TASK_ALLOCATION,
            _SpatialEventKind.TASK_REUSE,
        }:
            identity = item.identity
            if item.alias_output:
                if identity in live_aliases:
                    raise ValueError(
                        f"task allocates resident output alias {identity!r}"
                    )
                generation = generations.get(identity, 0)
                generations[identity] = generation + 1
                allocation_id = f"{identity}:{generation}"
                live_aliases[identity] = allocation_id
            else:
                allocation_id = identity
            if item.kind is _SpatialEventKind.TASK_REUSE:
                if item.source_identity is None:
                    raise AssertionError("task reuse lacks a source identity")
                allocation_events.append(
                    AllocationEvent(
                        position,
                        allocation_id,
                        AllocationOperation.REUSE,
                        item.bytes,
                        alignment=alignment,
                        source_allocation_id=item.source_identity,
                    )
                )
            else:
                allocation_events.append(
                    AllocationEvent(
                        position,
                        allocation_id,
                        AllocationOperation.ALLOCATE,
                        item.bytes,
                        alignment=alignment,
                    )
                )
        else:
            allocation_id = item.identity
            allocation_events.append(
                AllocationEvent(
                    position,
                    allocation_id,
                    AllocationOperation.FREE,
                    item.bytes,
                    alignment=alignment,
                )
            )
    return replay_slab_timeline(slab_bytes, tuple(allocation_events))


def _validate_profile_workspace(
    task: TaskSpec,
    workspace_bytes: int,
    measurement: TaskMeasurement,
    object_size: Mapping[str, int],
) -> None:
    """Ensure the scalar simulator charge has complete physical evidence."""

    extents = measurement.workspace_extent_bytes
    unclassified = workspace_bytes - sum(extents)
    if unclassified < 0:
        raise ValueError("profile workspace extents exceed charged workspace")
    if not unclassified:
        return
    contribution_extents = tuple(object_size[item.object_id] for item in task.mutations)
    if sum(contribution_extents) != unclassified:
        raise ValueError(
            "task workspace has no complete physical extent distribution: "
            f"task={task.task_id}, "
            f"unclassified={unclassified}, mutations={sum(contribution_extents)}"
        )
    return


@dataclass(frozen=True, slots=True)
class _TaskSpatialAllocation:
    kind: _SpatialEventKind
    identity: str
    bytes: int
    alias_output: bool = False
    source_identity: str | None = None


def _task_allocation_events(
    task: TaskSpec,
    measurement: TaskMeasurement,
    bindings: Sequence[TaskOutputBinding],
    alias_size: Mapping[str, int],
) -> tuple[_TaskSpatialAllocation, ...]:
    """Bind a structural ABI trace to one task's concrete Program outputs."""

    alias_by_leaf = {item.leaf_index: item.alias_group_id for item in bindings}
    if len(alias_by_leaf) != len(bindings):
        raise ValueError(f"task {task.task_id} binds an output leaf twice")
    local_identity: dict[int, str] = {}
    temporary_outputs: set[int] = set()
    result: list[_TaskSpatialAllocation] = []
    bound_aliases: set[str] = set()
    reused_ordinals = {
        event.reuses_ordinal
        for event in measurement.allocation_trace
        if event.reuses_ordinal is not None
    }
    for event in measurement.allocation_trace:
        if event.operation is TaskAllocationOperation.ALLOCATE:
            aliases = {
                alias_by_leaf[index]
                for index in event.output_leaf_indices
                if index in alias_by_leaf
            }
            if len(aliases) > 1:
                raise ValueError(
                    f"task {task.task_id} maps one storage to multiple aliases"
                )
            if aliases:
                identity = next(iter(aliases))
                expected = alias_size[identity]
                if event.charged_bytes != expected:
                    raise ValueError(
                        f"task {task.task_id} output {identity!r} allocated "
                        f"{event.charged_bytes} bytes; expected {expected}"
                    )
                bound_aliases.add(identity)
            elif event.output_leaf_indices:
                identity = f"temporary-output:{task.task_id}:{event.allocation_ordinal}"
                temporary_outputs.add(event.allocation_ordinal)
            else:
                identity = f"workspace:{task.task_id}:{event.allocation_ordinal}"
            local_identity[event.allocation_ordinal] = identity
            source_identity = (
                None
                if event.reuses_ordinal is None
                else local_identity.get(event.reuses_ordinal)
            )
            if event.reuses_ordinal is not None and source_identity is None:
                raise ValueError(
                    f"task {task.task_id} reuses an unknown profile extent"
                )
            result.append(
                _TaskSpatialAllocation(
                    _SpatialEventKind.TASK_REUSE
                    if source_identity is not None
                    else _SpatialEventKind.TASK_ALLOCATION,
                    identity,
                    event.charged_bytes,
                    bool(aliases),
                    source_identity,
                )
            )
            continue
        freed_identity = local_identity.get(event.allocation_ordinal)
        if freed_identity is None:
            raise ValueError(f"task {task.task_id} frees an unknown profile extent")
        if event.allocation_ordinal in temporary_outputs:
            raise ValueError(
                f"task {task.task_id} releases a returned output inside its trace"
            )
        if event.allocation_ordinal in reused_ordinals:
            continue
        result.append(
            _TaskSpatialAllocation(
                _SpatialEventKind.TASK_FREE,
                freed_identity,
                event.charged_bytes,
            )
        )
    required_aliases = {item.alias_group_id for item in bindings}
    if bound_aliases != required_aliases:
        missing = sorted(required_aliases - bound_aliases)
        raise ValueError(
            f"task {task.task_id} profile does not allocate outputs {missing}"
        )
    live_ordinals = set(local_identity)
    live_ordinals.difference_update(
        event.allocation_ordinal
        for event in measurement.allocation_trace
        if event.operation is TaskAllocationOperation.FREE
    )
    expected_live = temporary_outputs | {
        event.allocation_ordinal
        for event in measurement.allocation_trace
        if any(index in alias_by_leaf for index in event.output_leaf_indices)
    }
    if live_ordinals != expected_live:
        raise ValueError(
            f"task {task.task_id} profile retains unclassified allocations"
        )
    return tuple(result)


def output_bindings_for_entrypoints(
    tasks: Sequence[TaskSpec],
    entrypoints: Sequence[TaskEntrypoint | TrainingTaskEntrypoint],
    alias_by_object: Mapping[str, str],
) -> dict[str, tuple[TaskOutputBinding, ...]]:
    """Describe which returned tensor allocations become persistent outputs."""

    task_by_id = {task.task_id: task for task in tasks}
    result: dict[str, tuple[TaskOutputBinding, ...]] = {}
    for entrypoint in entrypoints:
        task = task_by_id.get(entrypoint.task_id)
        if task is None:
            continue
        slots = (
            entrypoint.gradient_output_slots
            if isinstance(entrypoint, TrainingTaskEntrypoint)
            and entrypoint.phase == "backward"
            else entrypoint.output_slots
        )
        output_objects = set(task.outputs)
        seen_aliases: set[str] = set()
        bindings: list[TaskOutputBinding] = []
        for slot in slots:
            if slot.object_id not in output_objects:
                continue
            alias_id = alias_by_object[slot.object_id]
            if alias_id in seen_aliases:
                continue
            bindings.append(TaskOutputBinding(slot.leaf_index, alias_id))
            seen_aliases.add(alias_id)
        result[task.task_id] = tuple(bindings)
    return result


__all__ = [
    "TaskOutputBinding",
    "output_bindings_for_entrypoints",
    "replay_selected_schedule",
]
