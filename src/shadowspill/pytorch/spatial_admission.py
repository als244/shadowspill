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
    HANDOFF_ALIAS = 0
    REPLACE_ALIAS = 1
    FREE_ALIAS = 2
    FREE_TEMPORARY_OUTPUT = 3
    ALLOCATE_PREFETCH = 4
    TASK_ALLOCATION = 5
    TASK_FREE = 6
    TASK_REUSE = 7


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
    task_id: str | None = None


@dataclass(frozen=True, slots=True)
class TaskOutputBinding:
    """Map one returned tensor leaf to its persistent Program alias group."""

    leaf_index: int
    alias_group_id: str
    replacement: bool = False
    source_alias_group_id: str | None = None

    def __post_init__(self) -> None:
        if self.leaf_index < 0:
            raise ValueError("output leaf index must be non-negative")
        if not self.alias_group_id:
            raise ValueError("output alias group ID must be non-empty")
        if self.source_alias_group_id == self.alias_group_id:
            raise ValueError("storage handoff source and destination must differ")


def replay_selected_schedule(
    selected: PressureFitResult,
    measurements: Mapping[str, TaskMeasurement],
    *,
    execution_pool_bytes: int,
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
        item.alias_group_id: item.size_bytes for item in program.alias_groups
    }
    zero_aliases = {
        alias_id for alias_id, size_bytes in alias_size.items() if size_bytes == 0
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
        task_id: str | None = None,
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
                task_id,
            )
        )
        sequence += 1

    for residency in selected.schedule.initial_residency:
        if residency.alias_group_id in zero_aliases:
            continue
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
            if alias_id in bound_output_aliases or alias_id in zero_aliases:
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
            zero_aliases,
        )
        for event in task_events:
            append(
                (
                    interval.end_ns
                    if event.kind is _SpatialEventKind.HANDOFF_ALIAS
                    else interval.start_ns
                ),
                event.kind,
                event.identity,
                event.bytes,
                alias_output=event.alias_output,
                source_identity=event.source_identity,
                task_id=task_id,
            )
        _validate_profile_workspace(
            task,
            profile.workspace_bytes,
            measurement,
            object_size,
            task_bindings,
        )
        for event in task_events:
            if event.kind not in {
                _SpatialEventKind.TASK_ALLOCATION,
                _SpatialEventKind.TASK_REUSE,
            }:
                continue
            if event.replacement_alias is not None:
                append(
                    interval.end_ns,
                    _SpatialEventKind.REPLACE_ALIAS,
                    event.replacement_alias,
                    event.bytes,
                    source_identity=event.identity,
                )
            elif event.identity.startswith(f"temporary-output:{task_id}:"):
                append(
                    interval.end_ns,
                    _SpatialEventKind.FREE_TEMPORARY_OUTPUT,
                    event.identity,
                    event.bytes,
                )

    transfer_keys: set[tuple[str, str, TransferDirection]] = set()
    for transfer in selected.simulation.transfer_intervals:
        if transfer.alias_group_id in zero_aliases:
            continue
        key = (
            transfer.trigger_task_id,
            transfer.alias_group_id,
            transfer.direction,
        )
        if key in transfer_keys:
            raise ValueError("spatial admission found a duplicate transfer interval")
        transfer_keys.add(key)
        if transfer.direction is TransferDirection.FETCH:
            append(
                interval_by_task[transfer.trigger_task_id].end_ns,
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

    handoff_releases = {
        (task_id, binding.source_alias_group_id)
        for task_id, bindings in bindings_by_task.items()
        for binding in bindings
        if binding.source_alias_group_id is not None
        and binding.alias_group_id not in zero_aliases
    }
    scheduled_releases = {
        (action.trigger_task_id, action.alias_group_id)
        for action in selected.schedule.actions
        if action.kind is MemoryActionKind.RELEASE
    }
    missing_handoff_releases = sorted(handoff_releases - scheduled_releases)
    if missing_handoff_releases:
        raise ValueError(
            "task-local storage handoff lacks its causal same-task release: "
            f"{missing_handoff_releases}"
        )

    for action in selected.schedule.actions:
        if action.alias_group_id in zero_aliases:
            continue
        interval = interval_by_task[action.trigger_task_id]
        if action.kind is MemoryActionKind.RELEASE:
            if (action.trigger_task_id, action.alias_group_id) in handoff_releases:
                continue
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
    live_alias_origins: dict[str, _SpatialEvent] = {}
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
            live_alias_origins[item.identity] = item
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
            live_alias_origins.pop(item.identity)
            allocation_events.append(
                AllocationEvent(
                    position,
                    allocation_id,
                    AllocationOperation.FREE,
                    item.bytes,
                    alignment=alignment,
                )
            )
        elif item.kind is _SpatialEventKind.REPLACE_ALIAS:
            if item.identity not in live_aliases or item.source_identity is None:
                raise ValueError(
                    f"spatial admission replaces nonresident alias {item.identity!r}"
                )
            old_allocation_id = live_aliases[item.identity]
            allocation_events.append(
                AllocationEvent(
                    position,
                    old_allocation_id,
                    AllocationOperation.FREE,
                    item.bytes,
                    alignment=alignment,
                )
            )
            live_aliases[item.identity] = item.source_identity
            live_alias_origins[item.identity] = item
        elif item.kind is _SpatialEventKind.HANDOFF_ALIAS:
            if item.source_identity is None:
                raise AssertionError("storage handoff lacks a source alias")
            if item.source_identity not in live_aliases:
                raise ValueError(
                    "storage handoff source is not resident: "
                    f"task={item.task_id!r}, source={item.source_identity!r}, "
                    f"destination={item.identity!r}"
                )
            if item.identity in live_aliases:
                raise ValueError(
                    "storage handoff destination is already resident: "
                    f"task={item.task_id!r}, source={item.source_identity!r}, "
                    f"destination={item.identity!r}"
                )
            allocation_id = live_aliases.pop(item.source_identity)
            live_alias_origins.pop(item.source_identity)
            live_aliases[item.identity] = allocation_id
            live_alias_origins[item.identity] = item
        elif item.kind in {
            _SpatialEventKind.TASK_ALLOCATION,
            _SpatialEventKind.TASK_REUSE,
        }:
            identity = item.identity
            if item.alias_output:
                if identity in live_aliases:
                    origin = live_alias_origins[identity]
                    raise ValueError(
                        "task allocates resident output alias "
                        f"{identity!r}: task={item.task_id!r}, "
                        f"time_ns={item.time_ns}, event={position}; "
                        "prior allocation="
                        f"{live_aliases[identity]!r}, "
                        f"prior_task={origin.task_id!r}, "
                        f"prior_time_ns={origin.time_ns}, "
                        f"prior_kind={origin.kind.name}"
                    )
                generation = generations.get(identity, 0)
                generations[identity] = generation + 1
                allocation_id = f"{identity}:{generation}"
                live_aliases[identity] = allocation_id
                live_alias_origins[identity] = item
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
    return replay_slab_timeline(execution_pool_bytes, tuple(allocation_events))


def _validate_profile_workspace(
    task: TaskSpec,
    workspace_bytes: int,
    measurement: TaskMeasurement,
    object_size: Mapping[str, int],
    bindings: Sequence[TaskOutputBinding] = (),
) -> None:
    """Ensure the scalar simulator charge has complete physical evidence."""

    extents = measurement.workspace_extent_bytes
    unclassified = workspace_bytes - sum(extents)
    if unclassified < 0:
        raise ValueError("profile workspace extents exceed charged workspace")
    if not unclassified:
        return
    replacement_leaves = {item.leaf_index for item in bindings if item.replacement}
    replacement_ordinals: set[int] = set()
    replacement_bytes = 0
    for event in measurement.allocation_trace:
        if event.operation is not TaskAllocationOperation.ALLOCATE:
            continue
        if (
            replacement_leaves.intersection(event.output_leaf_indices)
            and event.allocation_ordinal not in replacement_ordinals
        ):
            replacement_ordinals.add(event.allocation_ordinal)
            replacement_bytes += event.charged_bytes
    mutation_bytes = sum(object_size[item.object_id] for item in task.mutations)
    if unclassified not in {replacement_bytes, mutation_bytes}:
        raise ValueError(
            "task workspace has no complete physical extent distribution: "
            f"task={task.task_id}, "
            f"unclassified={unclassified}, replacements={replacement_bytes}, "
            f"mutations={mutation_bytes}"
        )
    return


@dataclass(frozen=True, slots=True)
class _TaskSpatialAllocation:
    kind: _SpatialEventKind
    identity: str
    bytes: int
    alias_output: bool = False
    source_identity: str | None = None
    replacement_alias: str | None = None


def _task_allocation_events(
    task: TaskSpec,
    measurement: TaskMeasurement,
    bindings: Sequence[TaskOutputBinding],
    alias_size: Mapping[str, int],
    zero_aliases: set[str] | None = None,
) -> tuple[_TaskSpatialAllocation, ...]:
    """Bind a structural ABI trace to one task's concrete Program outputs."""

    zero_aliases = zero_aliases or set()
    alias_by_leaf = {item.leaf_index: item.alias_group_id for item in bindings}
    replacement_by_leaf = {
        item.leaf_index: item.alias_group_id for item in bindings if item.replacement
    }
    handoff_by_leaf = {
        item.leaf_index: item
        for item in bindings
        if item.source_alias_group_id is not None
    }
    if len(alias_by_leaf) != len(bindings):
        raise ValueError(f"task {task.task_id} binds an output leaf twice")
    local_identity: dict[int, str] = {}
    temporary_outputs: set[int] = set()
    result: list[_TaskSpatialAllocation] = []
    required_aliases = {item.alias_group_id for item in bindings}
    bound_aliases = required_aliases.intersection(zero_aliases)
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
                    f"task {task.task_id} allocation "
                    f"{event.allocation_ordinal} maps output leaves "
                    f"{event.output_leaf_indices} to multiple aliases "
                    f"{sorted(aliases)}"
                )
            if aliases:
                alias_identity = next(iter(aliases))
                replacement_aliases = {
                    replacement_by_leaf[index]
                    for index in event.output_leaf_indices
                    if index in replacement_by_leaf
                }
                if replacement_aliases and replacement_aliases != {alias_identity}:
                    raise ValueError(
                        f"task {task.task_id} allocation mixes replacement aliases"
                    )
                expected = alias_size[alias_identity]
                if event.charged_bytes != expected:
                    raise ValueError(
                        f"task {task.task_id} output {alias_identity!r} allocated "
                        f"{event.charged_bytes} bytes; expected {expected}"
                    )
                bound_aliases.add(alias_identity)
                if replacement_aliases:
                    identity = (
                        f"replacement-output:{task.task_id}:{event.allocation_ordinal}"
                    )
                    replacement_alias = alias_identity
                else:
                    identity = alias_identity
                    replacement_alias = None
            elif event.output_leaf_indices:
                identity = f"temporary-output:{task.task_id}:{event.allocation_ordinal}"
                temporary_outputs.add(event.allocation_ordinal)
                replacement_alias = None
            else:
                identity = f"workspace:{task.task_id}:{event.allocation_ordinal}"
                replacement_alias = None
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
                    bool(aliases) and replacement_alias is None,
                    source_identity,
                    replacement_alias,
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
    donated_leaves = {
        item.output_leaf_index for item in measurement.output_input_bindings
    }
    for leaf_index, binding in handoff_by_leaf.items():
        if binding.alias_group_id in zero_aliases:
            continue
        if leaf_index not in donated_leaves:
            raise ValueError(
                f"task {task.task_id} handoff leaf {leaf_index} is not backed "
                "by a compiled input allocation"
            )
        source_alias = binding.source_alias_group_id
        if source_alias is None:
            raise AssertionError("filtered handoff binding lacks a source")
        if binding.alias_group_id in bound_aliases:
            continue
        result.append(
            _TaskSpatialAllocation(
                _SpatialEventKind.HANDOFF_ALIAS,
                binding.alias_group_id,
                alias_size[binding.alias_group_id],
                alias_output=True,
                source_identity=source_alias,
            )
        )
        bound_aliases.add(binding.alias_group_id)
    if bound_aliases != required_aliases:
        missing = sorted(required_aliases - bound_aliases)
        missing_bindings = [
            {
                "leaf_index": item.leaf_index,
                "alias_group_id": item.alias_group_id,
                "alias_bytes": alias_size[item.alias_group_id],
                "replacement": item.replacement,
                "source_alias_group_id": item.source_alias_group_id,
            }
            for item in bindings
            if item.alias_group_id in missing
        ]
        observed_allocations = [
            {
                "ordinal": event.allocation_ordinal,
                "bytes": event.charged_bytes,
                "output_leaves": event.output_leaf_indices,
            }
            for event in measurement.allocation_trace
            if event.operation is TaskAllocationOperation.ALLOCATE
            and event.output_leaf_indices
        ]
        raise ValueError(
            f"task {task.task_id} profile does not allocate outputs {missing}; "
            f"missing_bindings={missing_bindings!r}; "
            f"observed_allocations={observed_allocations!r}; "
            f"donated_leaves={sorted(donated_leaves)!r}"
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
        replacement_leaves = set(entrypoint.replacement_output_leaves)
        handoff_by_leaf = {
            item.leaf_index: item for item in entrypoint.storage_handoffs
        }
        seen_aliases: set[str] = set()
        bindings: list[TaskOutputBinding] = []
        for slot in slots:
            replacement = slot.leaf_index in replacement_leaves
            if slot.object_id not in output_objects and not replacement:
                continue
            alias_id = alias_by_object[slot.object_id]
            if alias_id in seen_aliases:
                continue
            bindings.append(
                TaskOutputBinding(
                    slot.leaf_index,
                    alias_id,
                    replacement,
                    (
                        alias_by_object[
                            handoff_by_leaf[slot.leaf_index].source_object_id
                        ]
                        if slot.leaf_index in handoff_by_leaf
                        else None
                    ),
                )
            )
            seen_aliases.add(alias_id)
        result[task.task_id] = tuple(bindings)
    return result


__all__ = [
    "TaskOutputBinding",
    "output_bindings_for_entrypoints",
    "replay_selected_schedule",
]
