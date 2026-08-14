"""Conservative slab replay for one selected PyTorch execution schedule."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import IntEnum

from shadowspill.ir import (
    MemoryActionKind,
    MemoryLocation,
    ObjectRole,
    TaskProfile,
    TaskSpec,
)
from shadowspill.planner import PressureFitResult
from shadowspill.pytorch.profiling import (
    TaskAllocationEvent,
    TaskAllocationOperation,
    TaskMeasurement,
)
from shadowspill.pytorch.runtime_adapter.bridge import TaskAllocationPlacementHint
from shadowspill.runtime import (
    AllocationEvent,
    AllocationOperation,
    SlabLayout,
    SlabReplay,
    plan_slab_layout,
)
from shadowspill.simulator import TaskInterval, TransferDirection

from ...lowering.forward import TaskEntrypoint
from ...lowering.training import TrainingTaskEntrypoint


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
    allocation_ordinal: int | None = None
    requested_bytes: int | None = None
    caller_owned: bool = False


@dataclass(frozen=True, slots=True)
class _ReplayIndex:
    alias_size: Mapping[str, int]
    zero_aliases: frozenset[str]
    alias_by_object: Mapping[str, str]
    object_size: Mapping[str, int]
    profile_by_id: Mapping[str, TaskProfile]
    task_by_id: Mapping[str, TaskSpec]
    interval_by_task: Mapping[str, TaskInterval]
    bindings_by_task: Mapping[str, tuple[TaskOutputBinding, ...]]
    caller_output_aliases: frozenset[str]


class _SpatialTimeline:
    """Append-only causal event stream before deterministic time ordering."""

    def __init__(self) -> None:
        self.events: list[_SpatialEvent] = []

    def append(
        self,
        time_ns: int,
        kind: _SpatialEventKind,
        identity: str,
        bytes_: int,
        *,
        planned: bool = False,
        alias_output: bool = False,
        source_identity: str | None = None,
        task_id: str | None = None,
        allocation_ordinal: int | None = None,
        requested_bytes: int | None = None,
        caller_owned: bool = False,
    ) -> None:
        self.events.append(
            _SpatialEvent(
                time_ns=time_ns,
                sequence=len(self.events),
                kind=kind,
                identity=identity,
                bytes=bytes_,
                planned=planned,
                alias_output=alias_output,
                source_identity=source_identity,
                task_id=task_id,
                allocation_ordinal=allocation_ordinal,
                requested_bytes=requested_bytes,
                caller_owned=caller_owned,
            )
        )


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


@dataclass(frozen=True, slots=True)
class PrefetchPlacement:
    """Exact execution-pool destination for one planned prefetch action."""

    alias_group_id: str
    offset: int
    trigger_task_id: str | None = None


@dataclass(frozen=True, slots=True)
class SelectedSpatialLayout:
    """Runtime guidance derived from one complete selected allocation timeline."""

    slab: SlabLayout
    task_allocations: tuple[tuple[str, TaskAllocationPlacementHint], ...]
    prefetches: tuple[PrefetchPlacement, ...]

    @property
    def replay(self) -> SlabReplay:
        return self.slab.replay

    def task_hints(self) -> dict[str, tuple[TaskAllocationPlacementHint, ...]]:
        result: dict[str, list[TaskAllocationPlacementHint]] = {}
        for task_id, placement in self.task_allocations:
            result.setdefault(task_id, []).append(placement)
        return {
            task_id: tuple(sorted(items, key=lambda item: item.allocation_ordinal))
            for task_id, items in result.items()
        }

    def initial_prefetch_offsets(self) -> dict[str, int]:
        return {
            item.alias_group_id: item.offset
            for item in self.prefetches
            if item.trigger_task_id is None
        }

    def action_prefetch_offsets(self) -> dict[tuple[str, str], int]:
        return {
            (item.trigger_task_id, item.alias_group_id): item.offset
            for item in self.prefetches
            if item.trigger_task_id is not None
        }


def replay_selected_schedule(
    selected: PressureFitResult,
    measurements: Mapping[str, TaskMeasurement],
    *,
    execution_pool_bytes: int,
    alignment: int = 256,
    output_bindings: Mapping[str, tuple[TaskOutputBinding, ...]] | None = None,
) -> SlabReplay:
    """Replay selected object lifetimes and task workspace through one slab."""

    slab, _sources = _selected_slab_layout(
        selected,
        measurements,
        execution_pool_bytes=execution_pool_bytes,
        alignment=alignment,
        output_bindings=output_bindings,
    )
    return slab.replay


def build_selected_spatial_layout(
    selected: PressureFitResult,
    measurements: Mapping[str, TaskMeasurement],
    *,
    execution_pool_bytes: int,
    alignment: int = 256,
    output_bindings: Mapping[str, tuple[TaskOutputBinding, ...]] | None = None,
) -> SelectedSpatialLayout:
    """Assign immutable offsets to every selected execution-pool lifetime."""

    slab, sources = _selected_slab_layout(
        selected,
        measurements,
        execution_pool_bytes=execution_pool_bytes,
        alignment=alignment,
        output_bindings=output_bindings,
    )
    return _runtime_spatial_layout(slab, sources)


def _selected_slab_layout(
    selected: PressureFitResult,
    measurements: Mapping[str, TaskMeasurement],
    *,
    execution_pool_bytes: int,
    alignment: int,
    output_bindings: Mapping[str, tuple[TaskOutputBinding, ...]] | None,
) -> tuple[SlabLayout, Mapping[str, _AllocationSource]]:
    """Build the static slab layout before frontend callback reconciliation."""

    index = _index_selected_schedule(selected, output_bindings)
    timeline = _build_spatial_timeline(selected, measurements, index)
    translated = _translate_spatial_timeline(timeline.events, alignment)
    slab = plan_slab_layout(
        execution_pool_bytes,
        translated.events,
        dynamic_allocation_ids=translated.caller_owned_allocation_ids,
    )
    return slab, translated.sources


def _index_selected_schedule(
    selected: PressureFitResult,
    output_bindings: Mapping[str, tuple[TaskOutputBinding, ...]] | None,
) -> _ReplayIndex:
    program = selected.program
    alias_size = {item.alias_group_id: item.size_bytes for item in program.alias_groups}
    return _ReplayIndex(
        alias_size=alias_size,
        zero_aliases=frozenset(
            alias_id for alias_id, size_bytes in alias_size.items() if size_bytes == 0
        ),
        alias_by_object={
            item.object_id: item.alias_group_id for item in program.objects
        },
        object_size={item.object_id: item.size_bytes for item in program.objects},
        profile_by_id={item.profile_id: item for item in program.profiles},
        task_by_id={
            item.task_id: item for item in program.selected_tasks(selected.selections)
        },
        interval_by_task={
            item.task_id: item for item in selected.simulation.task_intervals
        },
        bindings_by_task=dict(output_bindings or {}),
        caller_output_aliases=frozenset(
            item.alias_group_id
            for item in program.objects
            if item.role is ObjectRole.OUTPUT
        ),
    )


def _build_spatial_timeline(
    selected: PressureFitResult,
    measurements: Mapping[str, TaskMeasurement],
    index: _ReplayIndex,
) -> _SpatialTimeline:
    timeline = _SpatialTimeline()
    _append_initial_residency(selected, index, timeline)
    _append_task_lifetimes(measurements, index, timeline)
    _append_transfer_lifetimes(selected, index, timeline)
    _append_release_lifetimes(selected, index, timeline)
    return timeline


def _append_initial_residency(
    selected: PressureFitResult,
    index: _ReplayIndex,
    timeline: _SpatialTimeline,
) -> None:
    for residency in selected.schedule.initial_residency:
        if (
            residency.alias_group_id not in index.zero_aliases
            and residency.location is MemoryLocation.DEVICE
        ):
            timeline.append(
                0,
                _SpatialEventKind.ALLOCATE_PREFETCH,
                residency.alias_group_id,
                index.alias_size[residency.alias_group_id],
                planned=True,
            )


def _append_task_lifetimes(
    measurements: Mapping[str, TaskMeasurement],
    index: _ReplayIndex,
    timeline: _SpatialTimeline,
) -> None:
    for task_id, task in index.task_by_id.items():
        interval = index.interval_by_task[task_id]
        profile = index.profile_by_id[task.profile_id]
        measurement = _measurement_for_profile(measurements, profile)
        bindings = index.bindings_by_task.get(task_id, ())
        _append_implicit_task_outputs(task, interval, bindings, index, timeline)
        allocations = _task_allocation_events(
            task,
            measurement,
            bindings,
            index.alias_size,
            set(index.zero_aliases),
            set(index.caller_output_aliases),
        )
        _append_profiled_task_events(task_id, interval, allocations, timeline)
        _validate_profile_workspace(
            task,
            profile.workspace_bytes,
            measurement,
            index.object_size,
            bindings,
        )


def _measurement_for_profile(
    measurements: Mapping[str, TaskMeasurement],
    profile: TaskProfile,
) -> TaskMeasurement:
    try:
        return measurements[profile.compatibility_digest]
    except KeyError as error:
        raise ValueError(
            f"spatial admission lacks task measurement {profile.compatibility_digest!r}"
        ) from error


def _append_implicit_task_outputs(
    task: TaskSpec,
    interval: TaskInterval,
    bindings: Sequence[TaskOutputBinding],
    index: _ReplayIndex,
    timeline: _SpatialTimeline,
) -> None:
    bound_aliases = {item.alias_group_id for item in bindings}
    output_aliases = dict.fromkeys(
        index.alias_by_object[object_id] for object_id in task.outputs
    )
    for alias_id in output_aliases:
        if alias_id in bound_aliases or alias_id in index.zero_aliases:
            continue
        timeline.append(
            interval.start_ns,
            _SpatialEventKind.TASK_ALLOCATION,
            alias_id,
            index.alias_size[alias_id],
            alias_output=True,
            task_id=task.task_id,
            caller_owned=alias_id in index.caller_output_aliases,
        )


def _append_profiled_task_events(
    task_id: str,
    interval: TaskInterval,
    allocations: Sequence[_TaskSpatialAllocation],
    timeline: _SpatialTimeline,
) -> None:
    for event in allocations:
        event_time = (
            interval.end_ns
            if event.kind is _SpatialEventKind.HANDOFF_ALIAS
            else interval.start_ns
        )
        timeline.append(
            event_time,
            event.kind,
            event.identity,
            event.bytes,
            planned=(
                event.kind is _SpatialEventKind.TASK_ALLOCATION
                and (event.alias_output or event.replacement_alias is not None)
            ),
            alias_output=event.alias_output,
            source_identity=event.source_identity,
            task_id=task_id,
            allocation_ordinal=event.allocation_ordinal,
            requested_bytes=event.requested_bytes,
            caller_owned=event.caller_owned,
        )
    for event in allocations:
        _append_task_terminal_event(event, task_id, interval, timeline)


def _append_task_terminal_event(
    event: _TaskSpatialAllocation,
    task_id: str,
    interval: TaskInterval,
    timeline: _SpatialTimeline,
) -> None:
    if event.kind not in {
        _SpatialEventKind.TASK_ALLOCATION,
        _SpatialEventKind.TASK_REUSE,
    }:
        return
    if event.replacement_alias is not None:
        timeline.append(
            interval.end_ns,
            _SpatialEventKind.REPLACE_ALIAS,
            event.replacement_alias,
            event.bytes,
            source_identity=event.identity,
        )
    elif event.identity.startswith(f"temporary-output:{task_id}:"):
        timeline.append(
            interval.end_ns,
            _SpatialEventKind.FREE_TEMPORARY_OUTPUT,
            event.identity,
            event.bytes,
        )


def _append_transfer_lifetimes(
    selected: PressureFitResult,
    index: _ReplayIndex,
    timeline: _SpatialTimeline,
) -> None:
    observed: set[tuple[str, str, TransferDirection]] = set()
    for transfer in selected.simulation.transfer_intervals:
        if transfer.alias_group_id in index.zero_aliases:
            continue
        key = (
            transfer.trigger_task_id,
            transfer.alias_group_id,
            transfer.direction,
        )
        if key in observed:
            raise ValueError("spatial admission found a duplicate transfer interval")
        observed.add(key)
        if transfer.direction is TransferDirection.FETCH:
            timeline.append(
                index.interval_by_task[transfer.trigger_task_id].end_ns,
                _SpatialEventKind.ALLOCATE_PREFETCH,
                transfer.alias_group_id,
                index.alias_size[transfer.alias_group_id],
                planned=True,
                task_id=transfer.trigger_task_id,
            )
        else:
            timeline.append(
                transfer.end_ns,
                _SpatialEventKind.FREE_ALIAS,
                transfer.alias_group_id,
                index.alias_size[transfer.alias_group_id],
            )


def _append_release_lifetimes(
    selected: PressureFitResult,
    index: _ReplayIndex,
    timeline: _SpatialTimeline,
) -> None:
    handoff_releases = _handoff_releases(index)
    scheduled_releases = {
        (action.trigger_task_id, action.alias_group_id)
        for action in selected.schedule.actions
        if action.kind is MemoryActionKind.RELEASE
    }
    missing = sorted(handoff_releases - scheduled_releases)
    if missing:
        raise ValueError(
            f"task-local storage handoff lacks its causal same-task release: {missing}"
        )
    for action in selected.schedule.actions:
        key = (action.trigger_task_id, action.alias_group_id)
        if (
            action.alias_group_id in index.zero_aliases
            or action.kind is not MemoryActionKind.RELEASE
            or key in handoff_releases
        ):
            continue
        timeline.append(
            index.interval_by_task[action.trigger_task_id].end_ns,
            _SpatialEventKind.FREE_ALIAS,
            action.alias_group_id,
            index.alias_size[action.alias_group_id],
        )


def _handoff_releases(index: _ReplayIndex) -> set[tuple[str, str]]:
    return {
        (task_id, binding.source_alias_group_id)
        for task_id, bindings in index.bindings_by_task.items()
        for binding in bindings
        if binding.source_alias_group_id is not None
        and binding.alias_group_id not in index.zero_aliases
    }


@dataclass(slots=True)
class _AllocationReplayState:
    live_aliases: dict[str, str]
    live_origins: dict[str, _SpatialEvent]
    generations: dict[str, int]
    events: list[AllocationEvent]
    sources: dict[str, _AllocationSource]
    caller_owned_allocation_ids: set[str]


@dataclass(frozen=True, slots=True)
class _AllocationSource:
    task_id: str | None = None
    alias_group_id: str | None = None
    allocation_ordinal: int | None = None
    requested_bytes: int | None = None
    reuse: bool = False


@dataclass(frozen=True, slots=True)
class _TranslatedSpatialTimeline:
    events: tuple[AllocationEvent, ...]
    sources: Mapping[str, _AllocationSource]
    caller_owned_allocation_ids: frozenset[str]


def _translate_spatial_timeline(
    events: Sequence[_SpatialEvent],
    alignment: int,
) -> _TranslatedSpatialTimeline:
    state = _AllocationReplayState({}, {}, {}, [], {}, set())
    for position, event in enumerate(sorted(events, key=_spatial_event_order)):
        _translate_spatial_event(state, event, position, alignment)
    return _TranslatedSpatialTimeline(
        tuple(state.events),
        state.sources,
        frozenset(state.caller_owned_allocation_ids),
    )


def _runtime_spatial_layout(
    slab: SlabLayout,
    sources: Mapping[str, _AllocationSource],
) -> SelectedSpatialLayout:
    offsets = slab.offset_by_allocation()
    dynamic_ids = frozenset(slab.dynamic_allocation_ids)
    task_allocations: list[tuple[str, TaskAllocationPlacementHint]] = []
    prefetches: list[PrefetchPlacement] = []
    for allocation_id, source in sources.items():
        offset = offsets[allocation_id]
        if source.alias_group_id is not None:
            prefetches.append(
                PrefetchPlacement(
                    source.alias_group_id,
                    offset,
                    source.task_id,
                )
            )
            continue
        if (
            source.task_id is None
            or source.allocation_ordinal is None
            or source.requested_bytes is None
        ):
            raise ValueError(
                f"allocation {allocation_id!r} lacks runtime placement identity: "
                f"producer_task={source.task_id!r}"
            )
        task_allocations.append(
            (
                source.task_id,
                TaskAllocationPlacementHint(
                    source.allocation_ordinal,
                    source.requested_bytes,
                    (
                        slab.static_layout_bytes
                        if allocation_id in dynamic_ids
                        else offset
                    ),
                    source.reuse,
                    allocation_id in dynamic_ids,
                ),
            )
        )
    return SelectedSpatialLayout(
        slab,
        tuple(
            sorted(
                task_allocations,
                key=lambda item: (item[0], item[1].allocation_ordinal),
            )
        ),
        tuple(
            sorted(
                prefetches,
                key=lambda item: (
                    item.trigger_task_id is not None,
                    item.trigger_task_id or "",
                    item.alias_group_id,
                ),
            )
        ),
    )


def _spatial_event_order(event: _SpatialEvent) -> tuple[int, int, int]:
    task_local = event.kind in {
        _SpatialEventKind.TASK_ALLOCATION,
        _SpatialEventKind.TASK_FREE,
        _SpatialEventKind.TASK_REUSE,
    }
    return event.time_ns, 3 if task_local else int(event.kind), event.sequence


def _translate_spatial_event(
    state: _AllocationReplayState,
    event: _SpatialEvent,
    position: int,
    alignment: int,
) -> None:
    if event.kind is _SpatialEventKind.ALLOCATE_PREFETCH:
        _translate_prefetch(state, event, position, alignment)
    elif event.kind is _SpatialEventKind.FREE_ALIAS:
        _translate_alias_free(state, event, position, alignment)
    elif event.kind is _SpatialEventKind.REPLACE_ALIAS:
        _translate_alias_replacement(state, event, position, alignment)
    elif event.kind is _SpatialEventKind.HANDOFF_ALIAS:
        _translate_alias_handoff(state, event)
    elif event.kind in {
        _SpatialEventKind.TASK_ALLOCATION,
        _SpatialEventKind.TASK_REUSE,
    }:
        _translate_task_allocation(state, event, position, alignment)
    else:
        state.events.append(
            AllocationEvent(
                position,
                event.identity,
                AllocationOperation.FREE,
                event.bytes,
                alignment=alignment,
            )
        )


def _translate_prefetch(
    state: _AllocationReplayState,
    event: _SpatialEvent,
    position: int,
    alignment: int,
) -> None:
    if event.identity in state.live_aliases:
        return
    generation = state.generations.get(event.identity, 0)
    state.generations[event.identity] = generation + 1
    allocation_id = f"{event.identity}:{generation}"
    state.live_aliases[event.identity] = allocation_id
    state.live_origins[event.identity] = event
    state.events.append(
        AllocationEvent(
            position,
            allocation_id,
            AllocationOperation.ALLOCATE,
            event.bytes,
            alignment=alignment,
            planned=True,
        )
    )
    state.sources[allocation_id] = _AllocationSource(
        task_id=event.task_id,
        alias_group_id=event.identity,
    )


def _translate_alias_free(
    state: _AllocationReplayState,
    event: _SpatialEvent,
    position: int,
    alignment: int,
) -> None:
    if event.identity not in state.live_aliases:
        raise ValueError(
            f"spatial admission frees nonresident alias {event.identity!r}"
        )
    allocation_id = state.live_aliases.pop(event.identity)
    state.live_origins.pop(event.identity)
    state.events.append(
        AllocationEvent(
            position,
            allocation_id,
            AllocationOperation.FREE,
            event.bytes,
            alignment=alignment,
        )
    )


def _translate_alias_replacement(
    state: _AllocationReplayState,
    event: _SpatialEvent,
    position: int,
    alignment: int,
) -> None:
    if event.identity not in state.live_aliases or event.source_identity is None:
        raise ValueError(
            f"spatial admission replaces nonresident alias {event.identity!r}"
        )
    state.events.append(
        AllocationEvent(
            position,
            state.live_aliases[event.identity],
            AllocationOperation.FREE,
            event.bytes,
            alignment=alignment,
        )
    )
    state.live_aliases[event.identity] = event.source_identity
    state.live_origins[event.identity] = event


def _translate_alias_handoff(
    state: _AllocationReplayState,
    event: _SpatialEvent,
) -> None:
    source = event.source_identity
    if source is None:
        raise AssertionError("storage handoff lacks a source alias")
    if source not in state.live_aliases:
        raise ValueError(
            "storage handoff source is not resident: "
            f"task={event.task_id!r}, source={source!r}, "
            f"destination={event.identity!r}"
        )
    if event.identity in state.live_aliases:
        raise ValueError(
            "storage handoff destination is already resident: "
            f"task={event.task_id!r}, source={source!r}, "
            f"destination={event.identity!r}"
        )
    allocation_id = state.live_aliases.pop(source)
    state.live_origins.pop(source)
    state.live_aliases[event.identity] = allocation_id
    state.live_origins[event.identity] = event


def _translate_task_allocation(
    state: _AllocationReplayState,
    event: _SpatialEvent,
    position: int,
    alignment: int,
) -> None:
    allocation_id = _task_allocation_identity(state, event, position)
    if event.caller_owned:
        state.caller_owned_allocation_ids.add(allocation_id)
    operation = (
        AllocationOperation.REUSE
        if event.kind is _SpatialEventKind.TASK_REUSE
        else AllocationOperation.ALLOCATE
    )
    if operation is AllocationOperation.REUSE and event.source_identity is None:
        raise AssertionError("task reuse lacks a source identity")
    state.events.append(
        AllocationEvent(
            position,
            allocation_id,
            operation,
            event.bytes,
            alignment=alignment,
            planned=event.planned,
            source_allocation_id=(
                event.source_identity
                if operation is AllocationOperation.REUSE
                else None
            ),
        )
    )
    state.sources[allocation_id] = _AllocationSource(
        task_id=event.task_id,
        allocation_ordinal=event.allocation_ordinal,
        requested_bytes=event.requested_bytes,
        reuse=operation is AllocationOperation.REUSE,
    )


def _task_allocation_identity(
    state: _AllocationReplayState,
    event: _SpatialEvent,
    position: int,
) -> str:
    if not event.alias_output:
        return event.identity
    if event.identity in state.live_aliases:
        origin = state.live_origins[event.identity]
        raise ValueError(
            "task allocates resident output alias "
            f"{event.identity!r}: task={event.task_id!r}, "
            f"time_ns={event.time_ns}, event={position}; "
            f"prior allocation={state.live_aliases[event.identity]!r}, "
            f"prior_task={origin.task_id!r}, "
            f"prior_time_ns={origin.time_ns}, "
            f"prior_kind={origin.kind.name}"
        )
    generation = state.generations.get(event.identity, 0)
    state.generations[event.identity] = generation + 1
    allocation_id = f"{event.identity}:{generation}"
    state.live_aliases[event.identity] = allocation_id
    state.live_origins[event.identity] = event
    return allocation_id


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
    allocation_ordinal: int | None = None
    requested_bytes: int | None = None
    alias_output: bool = False
    source_identity: str | None = None
    replacement_alias: str | None = None
    caller_owned: bool = False


@dataclass(slots=True)
class _TaskAllocationBindingIndex:
    alias_by_leaf: dict[int, str]
    replacement_by_leaf: dict[int, str]
    handoff_by_leaf: dict[int, TaskOutputBinding]
    required_aliases: set[str]
    bound_aliases: set[str]
    local_identity: dict[int, str]
    temporary_outputs: set[int]
    reused_ordinals: set[int]
    events: list[_TaskSpatialAllocation]
    caller_output_aliases: set[str]


def _task_allocation_events(
    task: TaskSpec,
    measurement: TaskMeasurement,
    bindings: Sequence[TaskOutputBinding],
    alias_size: Mapping[str, int],
    zero_aliases: set[str] | None = None,
    caller_output_aliases: set[str] | None = None,
) -> tuple[_TaskSpatialAllocation, ...]:
    """Bind one structural allocation trace to concrete Program outputs."""

    index = _index_task_output_bindings(
        task,
        measurement,
        bindings,
        zero_aliases or set(),
        caller_output_aliases or set(),
    )
    for event in measurement.allocation_trace:
        _apply_task_allocation_event(task, event, alias_size, index)
    _append_storage_handoffs(task, measurement, alias_size, index)
    _validate_bound_task_outputs(task, measurement, bindings, alias_size, index)
    _validate_task_allocation_lifetimes(task, measurement, index)
    return tuple(index.events)


def _index_task_output_bindings(
    task: TaskSpec,
    measurement: TaskMeasurement,
    bindings: Sequence[TaskOutputBinding],
    zero_aliases: set[str],
    caller_output_aliases: set[str],
) -> _TaskAllocationBindingIndex:
    alias_by_leaf = {item.leaf_index: item.alias_group_id for item in bindings}
    if len(alias_by_leaf) != len(bindings):
        raise ValueError(f"task {task.task_id} binds an output leaf twice")
    required_aliases = {item.alias_group_id for item in bindings}
    return _TaskAllocationBindingIndex(
        alias_by_leaf=alias_by_leaf,
        replacement_by_leaf={
            item.leaf_index: item.alias_group_id
            for item in bindings
            if item.replacement
        },
        handoff_by_leaf={
            item.leaf_index: item
            for item in bindings
            if item.source_alias_group_id is not None
        },
        required_aliases=required_aliases,
        bound_aliases=required_aliases.intersection(zero_aliases),
        local_identity={},
        temporary_outputs=set(),
        reused_ordinals={
            event.reuses_ordinal
            for event in measurement.allocation_trace
            if event.reuses_ordinal is not None
        },
        events=[],
        caller_output_aliases=caller_output_aliases,
    )


def _apply_task_allocation_event(
    task: TaskSpec,
    event: TaskAllocationEvent,
    alias_size: Mapping[str, int],
    index: _TaskAllocationBindingIndex,
) -> None:
    if event.operation is TaskAllocationOperation.ALLOCATE:
        _record_task_allocation(task, event, alias_size, index)
    else:
        _record_task_free(task, event, index)


def _record_task_allocation(
    task: TaskSpec,
    event: TaskAllocationEvent,
    alias_size: Mapping[str, int],
    index: _TaskAllocationBindingIndex,
) -> None:
    alias = _allocation_output_alias(task, event, index)
    identity, replacement = _allocation_identity(task, event, alias, alias_size, index)
    source = (
        None
        if event.reuses_ordinal is None
        else index.local_identity.get(event.reuses_ordinal)
    )
    if event.reuses_ordinal is not None and source is None:
        raise ValueError(f"task {task.task_id} reuses an unknown profile extent")
    index.local_identity[event.allocation_ordinal] = identity
    index.events.append(
        _TaskSpatialAllocation(
            kind=(
                _SpatialEventKind.TASK_REUSE
                if source is not None
                else _SpatialEventKind.TASK_ALLOCATION
            ),
            identity=identity,
            bytes=event.charged_bytes,
            allocation_ordinal=event.allocation_ordinal,
            requested_bytes=event.requested_bytes,
            alias_output=alias is not None and replacement is None,
            source_identity=source,
            replacement_alias=replacement,
            caller_owned=alias in index.caller_output_aliases,
        )
    )


def _allocation_output_alias(
    task: TaskSpec,
    event: TaskAllocationEvent,
    index: _TaskAllocationBindingIndex,
) -> str | None:
    aliases = {
        index.alias_by_leaf[leaf]
        for leaf in event.output_leaf_indices
        if leaf in index.alias_by_leaf
    }
    if len(aliases) > 1:
        raise ValueError(
            f"task {task.task_id} allocation {event.allocation_ordinal} maps "
            f"output leaves {event.output_leaf_indices} to multiple aliases "
            f"{sorted(aliases)}"
        )
    return next(iter(aliases), None)


def _allocation_identity(
    task: TaskSpec,
    event: TaskAllocationEvent,
    alias: str | None,
    alias_size: Mapping[str, int],
    index: _TaskAllocationBindingIndex,
) -> tuple[str, str | None]:
    if alias is None:
        if event.output_leaf_indices:
            index.temporary_outputs.add(event.allocation_ordinal)
            return (
                f"temporary-output:{task.task_id}:{event.allocation_ordinal}",
                None,
            )
        return f"workspace:{task.task_id}:{event.allocation_ordinal}", None
    replacements = {
        index.replacement_by_leaf[leaf]
        for leaf in event.output_leaf_indices
        if leaf in index.replacement_by_leaf
    }
    if replacements and replacements != {alias}:
        raise ValueError(f"task {task.task_id} allocation mixes replacement aliases")
    expected = alias_size[alias]
    if event.charged_bytes != expected:
        raise ValueError(
            f"task {task.task_id} output {alias!r} allocated "
            f"{event.charged_bytes} bytes; expected {expected}"
        )
    index.bound_aliases.add(alias)
    if replacements:
        return f"replacement-output:{task.task_id}:{event.allocation_ordinal}", alias
    return alias, None


def _record_task_free(
    task: TaskSpec,
    event: TaskAllocationEvent,
    index: _TaskAllocationBindingIndex,
) -> None:
    identity = index.local_identity.get(event.allocation_ordinal)
    if identity is None:
        raise ValueError(f"task {task.task_id} frees an unknown profile extent")
    if event.allocation_ordinal in index.temporary_outputs:
        raise ValueError(
            f"task {task.task_id} releases a returned output inside its trace"
        )
    if event.allocation_ordinal not in index.reused_ordinals:
        index.events.append(
            _TaskSpatialAllocation(
                _SpatialEventKind.TASK_FREE,
                identity,
                event.charged_bytes,
                allocation_ordinal=event.allocation_ordinal,
                requested_bytes=event.requested_bytes,
            )
        )


def _append_storage_handoffs(
    task: TaskSpec,
    measurement: TaskMeasurement,
    alias_size: Mapping[str, int],
    index: _TaskAllocationBindingIndex,
) -> None:
    donated_leaves = {
        item.output_leaf_index for item in measurement.output_input_bindings
    }
    for leaf_index, binding in index.handoff_by_leaf.items():
        if binding.alias_group_id in index.bound_aliases:
            continue
        if leaf_index not in donated_leaves:
            raise ValueError(
                f"task {task.task_id} handoff leaf {leaf_index} is not backed "
                "by a compiled input allocation"
            )
        source = binding.source_alias_group_id
        if source is None:
            raise AssertionError("filtered handoff binding lacks a source")
        index.events.append(
            _TaskSpatialAllocation(
                _SpatialEventKind.HANDOFF_ALIAS,
                binding.alias_group_id,
                alias_size[binding.alias_group_id],
                alias_output=True,
                source_identity=source,
            )
        )
        index.bound_aliases.add(binding.alias_group_id)


def _validate_bound_task_outputs(
    task: TaskSpec,
    measurement: TaskMeasurement,
    bindings: Sequence[TaskOutputBinding],
    alias_size: Mapping[str, int],
    index: _TaskAllocationBindingIndex,
) -> None:
    if index.bound_aliases == index.required_aliases:
        return
    missing = sorted(index.required_aliases - index.bound_aliases)
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
    observed = [
        {
            "ordinal": event.allocation_ordinal,
            "bytes": event.charged_bytes,
            "output_leaves": event.output_leaf_indices,
        }
        for event in measurement.allocation_trace
        if event.operation is TaskAllocationOperation.ALLOCATE
        and event.output_leaf_indices
    ]
    donated = sorted(
        item.output_leaf_index for item in measurement.output_input_bindings
    )
    raise ValueError(
        f"task {task.task_id} profile does not allocate outputs {missing}; "
        f"missing_bindings={missing_bindings!r}; "
        f"observed_allocations={observed!r}; donated_leaves={donated!r}"
    )


def _validate_task_allocation_lifetimes(
    task: TaskSpec,
    measurement: TaskMeasurement,
    index: _TaskAllocationBindingIndex,
) -> None:
    live_ordinals = set(index.local_identity)
    live_ordinals.difference_update(
        event.allocation_ordinal
        for event in measurement.allocation_trace
        if event.operation is TaskAllocationOperation.FREE
    )
    expected_live = index.temporary_outputs | {
        event.allocation_ordinal
        for event in measurement.allocation_trace
        if any(leaf in index.alias_by_leaf for leaf in event.output_leaf_indices)
    }
    if live_ordinals != expected_live:
        raise ValueError(
            f"task {task.task_id} profile retains unclassified allocations"
        )


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
    "PrefetchPlacement",
    "SelectedSpatialLayout",
    "TaskOutputBinding",
    "build_selected_spatial_layout",
    "output_bindings_for_entrypoints",
    "replay_selected_schedule",
]
