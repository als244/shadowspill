"""Readable deterministic simulator used as the differential oracle."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
from typing import Never

from shadowspill.ir import (
    MemoryAction,
    MemoryActionKind,
    MemoryLocation,
    MemorySchedule,
    Program,
    RecomputationSelection,
    ResourceKind,
    TaskSpec,
)
from shadowspill.simulator.model import (
    DeviceMemoryPeak,
    MemorySnapshot,
    SimulationAdmission,
    SimulationConfig,
    SimulationInfeasibleError,
    SimulationResult,
    TaskInterval,
    TransferDirection,
    TransferInterval,
)

_NANOSECONDS_PER_SECOND = 1_000_000_000


@dataclass(slots=True)
class _AliasState:
    size_bytes: int
    device_id: str
    initial_version: int
    retain_spill_copy: bool
    device_allocated: bool = False
    device_ready: bool = False
    device_version: int = 0
    spill_allocated: bool = False
    spill_ready: bool = False
    spill_version: int = 0
    fetch_pending: bool = False
    evict_pending: bool = False


@dataclass(slots=True)
class _PendingTransfer:
    action_index: int
    alias_group_id: str
    trigger_task_id: str
    direction: TransferDirection
    ready_ns: int
    sequence: int
    stall_reasons: set[str] = field(default_factory=set)


@dataclass(slots=True)
class _ActiveTransfer:
    pending: _PendingTransfer
    start_ns: int
    end_ns: int


@dataclass(slots=True)
class _ActiveTask:
    task: TaskSpec
    ready_ns: int
    start_ns: int
    end_ns: int
    workspace_bytes: int
    output_aliases: tuple[str, ...]
    stall_reasons: tuple[str, ...]


@dataclass(slots=True)
class _TaskWait:
    ready_ns: int | None = None
    reasons: set[str] = field(default_factory=set)


class _Simulator:
    def __init__(
        self,
        program: Program,
        schedule: MemorySchedule,
        selections: tuple[RecomputationSelection, ...],
        config: SimulationConfig,
        *,
        admission: SimulationAdmission | None,
        record_timeline: bool,
    ) -> None:
        self.program = program
        self.schedule = schedule
        self.selections = selections
        self.config = config
        self.admission = admission
        self.record_timeline = record_timeline
        self.tasks = program.selected_tasks(selections)
        self.task_by_id = {task.task_id: task for task in self.tasks}
        self.task_order = {task.task_id: index for index, task in enumerate(self.tasks)}
        self.profile_by_id = {
            profile.profile_id: profile for profile in program.profiles
        }
        self.alias_by_id = {item.alias_group_id: item for item in program.alias_groups}
        self.object_alias = {
            item.object_id: item.alias_group_id for item in program.objects
        }
        self.device_config = {item.device_id: item for item in config.devices}
        if admission is not None and admission.device_capacity_bytes:
            capacities = dict(admission.device_capacity_bytes)
            if set(capacities) != set(self.device_config):
                raise ValueError(
                    "simulation admission capacities must exactly match Program "
                    f"devices; expected {sorted(self.device_config)}, "
                    f"got {sorted(capacities)}"
                )
            self.device_config = {
                device_id: replace(item, capacity_bytes=capacities[device_id])
                for device_id, item in self.device_config.items()
            }
        self.task_physical_deltas = {
            item.task_id: item
            for item in (() if admission is None else admission.task_deltas)
        }
        self.action_physical_deltas = {
            item.action_index: item
            for item in (() if admission is None else admission.action_deltas)
        }
        self.task_reuse_dependencies: dict[str, tuple[int, ...]] = {}
        self.action_reuse_dependencies: dict[int, tuple[int, ...]] = {}
        if admission is not None:
            task_dependencies: dict[str, list[int]] = {}
            action_dependencies: dict[int, list[int]] = {}
            for dependency in admission.reuse_dependencies:
                if dependency.successor_task_id is not None:
                    task_dependencies.setdefault(
                        dependency.successor_task_id, []
                    ).append(dependency.predecessor_action_index)
                else:
                    assert dependency.successor_action_index is not None
                    action_dependencies.setdefault(
                        dependency.successor_action_index, []
                    ).append(dependency.predecessor_action_index)
            self.task_reuse_dependencies = {
                key: tuple(values) for key, values in task_dependencies.items()
            }
            self.action_reuse_dependencies = {
                key: tuple(values) for key, values in action_dependencies.items()
            }
        self._validate_inputs()
        self.alias_state = {
            item.alias_group_id: _AliasState(
                size_bytes=item.size_bytes,
                device_id=item.device_id,
                initial_version=item.initial_version,
                retain_spill_copy=item.retain_spill_copy,
                device_version=item.initial_version,
                spill_version=item.initial_version,
            )
            for item in program.alias_groups
        }
        self.next_action_index = 0
        self.now_ns = 0
        self.unlaunched = {task.task_id for task in self.tasks}
        self.completed: dict[str, int] = {}
        self.completed_transfer_actions: set[int] = set()
        self.active_tasks: dict[tuple[str, ResourceKind, int], _ActiveTask] = {}
        self.task_waits = {task.task_id: _TaskWait() for task in self.tasks}
        self.pending_fetch = {
            device.device_id: deque[_PendingTransfer]() for device in config.devices
        }
        self.pending_evict = {
            device.device_id: deque[_PendingTransfer]() for device in config.devices
        }
        self.active_fetch: dict[str, _ActiveTransfer] = {}
        self.active_evict: dict[str, _ActiveTransfer] = {}
        self.transfer_sequence: dict[tuple[str, TransferDirection], int] = {}
        self.device_object_bytes = {device.device_id: 0 for device in config.devices}
        self.device_workspace_bytes = {device.device_id: 0 for device in config.devices}
        self.device_physical_bytes = {device.device_id: 0 for device in config.devices}
        self.device_object_peaks = {device.device_id: 0 for device in config.devices}
        self.device_workspace_peaks = {device.device_id: 0 for device in config.devices}
        self.device_total_peaks = {device.device_id: 0 for device in config.devices}
        self.spill_bytes = 0
        self.spill_peak_bytes = 0
        self.task_intervals: list[TaskInterval] = []
        self.transfer_intervals: list[TransferInterval] = []
        self.memory_timeline: list[MemorySnapshot] = []

    def _validate_inputs(self) -> None:
        program_devices = {item.device_id for item in self.program.devices}
        configured_devices = set(self.device_config)
        if configured_devices != program_devices:
            raise ValueError(
                "simulation devices must exactly match Program devices; "
                f"expected {sorted(program_devices)}, got {sorted(configured_devices)}"
            )
        self.schedule.validate(self.program, self.selections)
        if self.admission is None:
            return
        initial_devices = dict(self.admission.initial_physical_bytes)
        if set(initial_devices) != configured_devices:
            raise ValueError(
                "admission initial physical devices must exactly match simulation "
                f"devices; expected {sorted(configured_devices)}, got "
                f"{sorted(initial_devices)}"
            )
        unknown_tasks = sorted(set(self.task_physical_deltas) - self.task_by_id.keys())
        if unknown_tasks:
            raise ValueError(f"admission contains unknown tasks {unknown_tasks}")
        action_count = len(self.schedule.actions)
        unknown_actions = sorted(
            index
            for index in self.action_physical_deltas
            if index >= action_count
        )
        if unknown_actions:
            raise ValueError(
                f"admission contains unknown action indices {unknown_actions}"
            )
        for dependency in self.admission.reuse_dependencies:
            predecessor = dependency.predecessor_action_index
            if predecessor >= action_count:
                raise ValueError(
                    f"memory-reuse predecessor action {predecessor} is unknown"
                )
            if self.schedule.actions[predecessor].kind is not MemoryActionKind.OFFLOAD:
                raise ValueError(
                    f"memory-reuse predecessor action {predecessor} is not an eviction"
                )
            successor_task = dependency.successor_task_id
            if successor_task is not None and successor_task not in self.task_by_id:
                raise ValueError(
                    f"memory-reuse successor task {successor_task!r} is unknown"
                )
            successor_action = dependency.successor_action_index
            if successor_action is not None and successor_action >= action_count:
                raise ValueError(
                    f"memory-reuse successor action {successor_action} is unknown"
                )

    def _snapshot(self) -> None:
        for device_id in self.device_object_bytes:
            objects = self.device_object_bytes[device_id]
            workspace = self.device_workspace_bytes[device_id]
            self.device_object_peaks[device_id] = max(
                self.device_object_peaks[device_id], objects
            )
            self.device_workspace_peaks[device_id] = max(
                self.device_workspace_peaks[device_id], workspace
            )
            self.device_total_peaks[device_id] = max(
                self.device_total_peaks[device_id],
                self.device_physical_bytes[device_id],
            )
        self.spill_peak_bytes = max(self.spill_peak_bytes, self.spill_bytes)
        if self.record_timeline:
            self.memory_timeline.append(
                MemorySnapshot(
                    time_ns=self.now_ns,
                    device_object_bytes=tuple(self.device_object_bytes.items()),
                    device_workspace_bytes=tuple(self.device_workspace_bytes.items()),
                    spill_bytes=self.spill_bytes,
                    device_physical_bytes=tuple(self.device_physical_bytes.items()),
                )
            )

    def _initialize_memory(self) -> None:
        for state in self.alias_state.values():
            if state.size_bytes == 0:
                # A zero-length tensor has semantic identity and dependency
                # edges, but no payload whose residency can be absent.
                state.device_allocated = True
                state.device_ready = True
                continue
            if state.retain_spill_copy:
                state.spill_allocated = True
                state.spill_ready = True
                self.spill_bytes += state.size_bytes
        for residency in self.schedule.initial_residency:
            state = self.alias_state[residency.alias_group_id]
            if residency.location is MemoryLocation.DEVICE:
                state.device_allocated = True
                state.device_ready = True
                self.device_object_bytes[state.device_id] += state.size_bytes
            else:
                if not state.spill_allocated:
                    state.spill_allocated = True
                    self.spill_bytes += state.size_bytes
                state.spill_ready = True
        if self.admission is None:
            self.device_physical_bytes.update(self.device_object_bytes)
        else:
            self.device_physical_bytes.update(
                dict(self.admission.initial_physical_bytes)
            )
        self._snapshot()
        for device_id, used in self.device_physical_bytes.items():
            capacity = self.device_config[device_id].capacity_bytes
            if used > capacity:
                self._raise_capacity(
                    kind="initial-device-capacity",
                    location=f"device:{device_id}",
                    capacity=capacity,
                    used=used,
                    requested=0,
                )
        if self.spill_bytes > self.config.spill_capacity_bytes:
            self._raise_capacity(
                kind="initial-spill-capacity",
                location="host",
                capacity=self.config.spill_capacity_bytes,
                used=self.spill_bytes,
                requested=0,
            )

    def _raise_capacity(
        self,
        *,
        kind: str,
        location: str,
        capacity: int,
        used: int,
        requested: int,
        task_id: str | None = None,
        aliases: tuple[str, ...] = (),
    ) -> None:
        raise SimulationInfeasibleError(
            f"{kind} at {self.now_ns} ns: {used} used + {requested} requested "
            f"exceeds {capacity} bytes at {location}",
            kind=kind,
            time_ns=self.now_ns,
            task_id=task_id,
            alias_group_ids=aliases,
            location=location,
            capacity_bytes=capacity,
            used_bytes=used,
            requested_bytes=requested,
        )

    def _apply_physical_delta(self, device_id: str, delta: int) -> None:
        updated = self.device_physical_bytes[device_id] + delta
        if updated < 0:
            raise ValueError(
                "simulation admission underflows physical execution memory: "
                f"device={device_id!r}, "
                f"current={self.device_physical_bytes[device_id]}, "
                f"delta={delta}"
            )
        self.device_physical_bytes[device_id] = updated

    def _task_start_delta(self, task: TaskSpec, default: int) -> int:
        admitted = self.task_physical_deltas.get(task.task_id)
        return default if admitted is None else admitted.start_bytes

    def _task_completion_delta(self, task: TaskSpec, default: int) -> int:
        admitted = self.task_physical_deltas.get(task.task_id)
        return default if admitted is None else admitted.completion_bytes

    def _action_trigger_delta(self, action_index: int, default: int) -> int:
        admitted = self.action_physical_deltas.get(action_index)
        return default if admitted is None else admitted.trigger_bytes

    def _action_completion_delta(self, action_index: int, default: int) -> int:
        admitted = self.action_physical_deltas.get(action_index)
        return default if admitted is None else admitted.completion_bytes

    def _reuse_dependencies_complete(self, predecessors: tuple[int, ...]) -> bool:
        return all(
            predecessor in self.completed_transfer_actions
            for predecessor in predecessors
        )

    def _resource_key(self, task: TaskSpec) -> tuple[str, ResourceKind, int]:
        return (task.resource.device_id, task.resource.kind, task.resource.lane)

    def _first_unlaunched_on_lane(self, task: TaskSpec) -> bool:
        key = self._resource_key(task)
        for candidate in self.tasks:
            if candidate.task_id not in self.unlaunched:
                continue
            if self._resource_key(candidate) == key:
                return candidate.task_id == task.task_id
        return False

    def _task_output_aliases(self, task: TaskSpec) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(self.object_alias[object_id] for object_id in task.outputs)
        )

    def _task_missing_inputs(self, task: TaskSpec) -> tuple[str, ...]:
        missing: list[str] = []
        for alias_id in dict.fromkeys(
            self.object_alias[object_id] for object_id in task.inputs
        ):
            state = self.alias_state[alias_id]
            if not state.device_ready or state.fetch_pending or state.evict_pending:
                missing.append(alias_id)
        return tuple(missing)

    def _try_launch_tasks(self) -> bool:
        changed = False
        for task in self.tasks:
            if task.task_id not in self.unlaunched:
                continue
            if not self._first_unlaunched_on_lane(task):
                continue
            key = self._resource_key(task)
            if key in self.active_tasks:
                continue
            if any(
                dependency not in self.completed for dependency in task.dependencies
            ):
                continue
            wait = self.task_waits[task.task_id]
            dependency_ready = max(
                (self.completed[item] for item in task.dependencies),
                default=0,
            )
            if wait.ready_ns is None:
                wait.ready_ns = max(self.now_ns, dependency_ready)
            reuse_dependencies = self.task_reuse_dependencies.get(task.task_id, ())
            if not self._reuse_dependencies_complete(reuse_dependencies):
                wait.reasons.add("memory-reuse")
                continue
            missing = self._task_missing_inputs(task)
            if missing:
                wait.reasons.add("input-residency")
                continue
            profile = self.profile_by_id[task.profile_id]
            output_aliases = self._task_output_aliases(task)
            new_output_bytes = sum(
                self.alias_state[alias_id].size_bytes
                for alias_id in output_aliases
                if not self.alias_state[alias_id].device_allocated
            )
            device_id = task.resource.device_id
            logical_requested = new_output_bytes + profile.workspace_bytes
            physical_delta = self._task_start_delta(task, logical_requested)
            used = self.device_physical_bytes[device_id]
            requested = max(physical_delta, 0)
            if used + requested > self.device_config[device_id].capacity_bytes:
                wait.reasons.add("device-capacity")
                continue
            for alias_id in output_aliases:
                state = self.alias_state[alias_id]
                if not state.device_allocated:
                    state.device_allocated = True
                    self.device_object_bytes[device_id] += state.size_bytes
                state.device_ready = False
                state.fetch_pending = False
                state.evict_pending = False
                state.spill_ready = False
            self.device_workspace_bytes[device_id] += profile.workspace_bytes
            self._apply_physical_delta(device_id, physical_delta)
            start = self.now_ns
            end = start + profile.runtime_ns
            active = _ActiveTask(
                task=task,
                ready_ns=wait.ready_ns,
                start_ns=start,
                end_ns=end,
                workspace_bytes=profile.workspace_bytes,
                output_aliases=output_aliases,
                stall_reasons=tuple(sorted(wait.reasons)),
            )
            self.active_tasks[key] = active
            self.unlaunched.remove(task.task_id)
            self._snapshot()
            changed = True
        return changed

    def _transfer_runtime_ns(
        self,
        state: _AliasState,
        direction: TransferDirection,
    ) -> int:
        config = self.device_config[state.device_id]
        if direction is TransferDirection.FETCH:
            bandwidth = config.fetch_bandwidth_bytes_per_second
            latency = config.fetch_latency_ns
        else:
            bandwidth = config.evict_bandwidth_bytes_per_second
            latency = config.evict_latency_ns
        transfer = (
            state.size_bytes * _NANOSECONDS_PER_SECOND + bandwidth - 1
        ) // bandwidth
        return latency + transfer

    def _enqueue_transfer(
        self,
        action_index: int,
        action: MemoryAction,
        direction: TransferDirection,
    ) -> None:
        state = self.alias_state[action.alias_group_id]
        key = (state.device_id, direction)
        sequence = self.transfer_sequence.get(key, 0)
        self.transfer_sequence[key] = sequence + 1
        pending = _PendingTransfer(
            action_index=action_index,
            alias_group_id=action.alias_group_id,
            trigger_task_id=action.trigger_task_id,
            direction=direction,
            ready_ns=self.now_ns,
            sequence=sequence,
        )
        if direction is TransferDirection.FETCH:
            state.fetch_pending = True
            self.pending_fetch[state.device_id].append(pending)
        else:
            state.evict_pending = True
            self.pending_evict[state.device_id].append(pending)

    def _release(self, action: MemoryAction) -> None:
        state = self.alias_state[action.alias_group_id]
        if not state.device_allocated or not state.device_ready:
            raise SimulationInfeasibleError(
                f"release of {action.alias_group_id!r} has no ready device copy",
                kind="invalid-release",
                time_ns=self.now_ns,
                task_id=action.trigger_task_id,
                alias_group_ids=(action.alias_group_id,),
                location=f"device:{state.device_id}",
            )
        if state.fetch_pending or state.evict_pending:
            raise SimulationInfeasibleError(
                f"release of {action.alias_group_id!r} conflicts with a transfer",
                kind="release-transfer-conflict",
                time_ns=self.now_ns,
                task_id=action.trigger_task_id,
                alias_group_ids=(action.alias_group_id,),
            )
        state.device_allocated = False
        state.device_ready = False
        self.device_object_bytes[state.device_id] -= state.size_bytes
        if state.spill_allocated and not state.retain_spill_copy:
            state.spill_allocated = False
            state.spill_ready = False
            self.spill_bytes -= state.size_bytes
        self._snapshot()

    def _submit_ready_actions(self) -> None:
        while self.next_action_index < len(self.schedule.actions):
            action_index = self.next_action_index
            action = self.schedule.actions[action_index]
            if action.trigger_task_id not in self.completed:
                return
            state = self.alias_state[action.alias_group_id]
            device_id = state.device_id
            if action.kind is MemoryActionKind.RELEASE:
                default_delta = -state.size_bytes
            elif action.kind is MemoryActionKind.PREFETCH:
                default_delta = 0 if state.device_allocated else state.size_bytes
            else:
                default_delta = 0
            physical_delta = self._action_trigger_delta(
                action_index, default_delta
            )
            requested = max(physical_delta, 0)
            capacity = self.device_config[device_id].capacity_bytes
            if self.device_physical_bytes[device_id] + requested > capacity:
                self._raise_capacity(
                    kind="prefetch-device-capacity",
                    location=f"device:{device_id}",
                    capacity=capacity,
                    used=self.device_physical_bytes[device_id],
                    requested=requested,
                    task_id=action.trigger_task_id,
                    aliases=(action.alias_group_id,),
                )
            if action.kind is MemoryActionKind.RELEASE:
                self._release(action)
            elif action.kind is MemoryActionKind.OFFLOAD:
                if not state.device_allocated or not state.device_ready:
                    raise SimulationInfeasibleError(
                        f"offload of {action.alias_group_id!r} lacks a device source",
                        kind="invalid-offload",
                        time_ns=self.now_ns,
                        task_id=action.trigger_task_id,
                        alias_group_ids=(action.alias_group_id,),
                    )
                if not state.spill_allocated:
                    if (
                        self.spill_bytes + state.size_bytes
                        > self.config.spill_capacity_bytes
                    ):
                        self._raise_capacity(
                            kind="offload-spill-capacity",
                            location="host",
                            capacity=self.config.spill_capacity_bytes,
                            used=self.spill_bytes,
                            requested=state.size_bytes,
                            task_id=action.trigger_task_id,
                            aliases=(action.alias_group_id,),
                        )
                    state.spill_allocated = True
                    state.spill_ready = False
                    self.spill_bytes += state.size_bytes
                self._enqueue_transfer(
                    action_index,
                    action,
                    TransferDirection.EVICT,
                )
            else:
                if state.device_allocated and not state.evict_pending:
                    raise SimulationInfeasibleError(
                        f"prefetch of {action.alias_group_id!r} already has a "
                        "device copy",
                        kind="invalid-prefetch",
                        time_ns=self.now_ns,
                        task_id=action.trigger_task_id,
                        alias_group_ids=(action.alias_group_id,),
                    )
                if not state.spill_ready and not state.evict_pending:
                    raise SimulationInfeasibleError(
                        f"prefetch of {action.alias_group_id!r} lacks a spill source",
                        kind="invalid-prefetch",
                        time_ns=self.now_ns,
                        task_id=action.trigger_task_id,
                        alias_group_ids=(action.alias_group_id,),
                    )
                if not state.device_allocated:
                    state.device_allocated = True
                    state.device_ready = False
                    self.device_object_bytes[device_id] += state.size_bytes
                self._enqueue_transfer(
                    action_index,
                    action,
                    TransferDirection.FETCH,
                )
            self._apply_physical_delta(device_id, physical_delta)
            self._snapshot()
            self.next_action_index += 1

    def _complete_task(self, key: tuple[str, ResourceKind, int]) -> None:
        active = self.active_tasks.pop(key)
        task = active.task
        device_id = task.resource.device_id
        self.device_workspace_bytes[device_id] -= active.workspace_bytes
        self._apply_physical_delta(
            device_id,
            self._task_completion_delta(task, -active.workspace_bytes),
        )
        for alias_id in active.output_aliases:
            state = self.alias_state[alias_id]
            state.device_ready = True
            state.device_version += 1
            state.spill_ready = False
        for mutation in task.mutations:
            alias_id = self.object_alias[mutation.object_id]
            state = self.alias_state[alias_id]
            state.device_version += mutation.version_delta
            state.spill_ready = False
        self.completed[task.task_id] = self.now_ns
        self.task_intervals.append(
            TaskInterval(
                task_id=task.task_id,
                device_id=device_id,
                resource_kind=task.resource.kind,
                resource_lane=task.resource.lane,
                ready_ns=active.ready_ns,
                start_ns=active.start_ns,
                end_ns=active.end_ns,
                workspace_bytes=active.workspace_bytes,
                stall_reasons=active.stall_reasons,
            )
        )
        self._snapshot()
        self._submit_ready_actions()

    def _try_start_direction(
        self,
        device_id: str,
        direction: TransferDirection,
    ) -> bool:
        if direction is TransferDirection.FETCH:
            queue = self.pending_fetch[device_id]
            active_table = self.active_fetch
        else:
            queue = self.pending_evict[device_id]
            active_table = self.active_evict
        if device_id in active_table or not queue:
            return False
        pending = queue[0]
        state = self.alias_state[pending.alias_group_id]
        reuse_dependencies = self.action_reuse_dependencies.get(
            pending.action_index, ()
        )
        if not self._reuse_dependencies_complete(reuse_dependencies):
            pending.stall_reasons.add("memory-reuse")
            return False
        if direction is TransferDirection.FETCH:
            if state.evict_pending or not state.spill_ready:
                pending.stall_reasons.add("source-readiness")
                return False
            if not state.device_allocated:
                raise AssertionError("queued FETCH has no trigger-time reservation")
        else:
            if not state.device_ready:
                pending.stall_reasons.add("source-readiness")
                return False
            if not state.spill_allocated:
                raise AssertionError("queued EVICT has no trigger-time reservation")
        queue.popleft()
        runtime = self._transfer_runtime_ns(state, direction)
        active_table[device_id] = _ActiveTransfer(
            pending=pending,
            start_ns=self.now_ns,
            end_ns=self.now_ns + runtime,
        )
        self._snapshot()
        return True

    def _try_start_transfers(self) -> bool:
        changed = False
        for device_id in self.device_config:
            changed |= self._try_start_direction(device_id, TransferDirection.FETCH)
            changed |= self._try_start_direction(device_id, TransferDirection.EVICT)
        return changed

    def _complete_transfer(
        self,
        device_id: str,
        direction: TransferDirection,
    ) -> None:
        if direction is TransferDirection.FETCH:
            active = self.active_fetch.pop(device_id)
        else:
            active = self.active_evict.pop(device_id)
        pending = active.pending
        state = self.alias_state[pending.alias_group_id]
        default_physical_delta = 0
        if direction is TransferDirection.FETCH:
            state.device_ready = True
            state.device_version = state.spill_version
            state.fetch_pending = False
            if not state.retain_spill_copy:
                state.spill_allocated = False
                state.spill_ready = False
                self.spill_bytes -= state.size_bytes
        else:
            state.spill_ready = True
            state.spill_version = state.device_version
            state.evict_pending = False
            state.device_ready = False
            if not state.fetch_pending:
                state.device_allocated = False
                self.device_object_bytes[device_id] -= state.size_bytes
                default_physical_delta = -state.size_bytes
        self._apply_physical_delta(
            device_id,
            self._action_completion_delta(
                pending.action_index, default_physical_delta
            ),
        )
        self.completed_transfer_actions.add(pending.action_index)
        self.transfer_intervals.append(
            TransferInterval(
                alias_group_id=pending.alias_group_id,
                trigger_task_id=pending.trigger_task_id,
                device_id=device_id,
                direction=direction,
                sequence=pending.sequence,
                ready_ns=pending.ready_ns,
                start_ns=active.start_ns,
                end_ns=active.end_ns,
                bytes=state.size_bytes,
                stall_reasons=tuple(sorted(pending.stall_reasons)),
            )
        )
        self._snapshot()

    def _next_event_time(self) -> int | None:
        ends = [active.end_ns for active in self.active_tasks.values()]
        ends.extend(active.end_ns for active in self.active_fetch.values())
        ends.extend(active.end_ns for active in self.active_evict.values())
        return min(ends) if ends else None

    def _complete_events(self) -> None:
        for device_id in sorted(self.active_fetch):
            if self.active_fetch[device_id].end_ns == self.now_ns:
                self._complete_transfer(device_id, TransferDirection.FETCH)
        for device_id in sorted(self.active_evict):
            if self.active_evict[device_id].end_ns == self.now_ns:
                self._complete_transfer(device_id, TransferDirection.EVICT)
        completed_keys = sorted(
            (
                key
                for key, active in self.active_tasks.items()
                if active.end_ns == self.now_ns
            ),
            key=lambda key: self.task_order[self.active_tasks[key].task.task_id],
        )
        for key in completed_keys:
            self._complete_task(key)

    def _deadlock(self) -> Never:
        for direction, queues in (
            (TransferDirection.FETCH, self.pending_fetch),
            (TransferDirection.EVICT, self.pending_evict),
        ):
            for device_id, queue in queues.items():
                if not queue:
                    continue
                pending = queue[0]
                state = self.alias_state[pending.alias_group_id]
                if direction is TransferDirection.FETCH:
                    used = self.device_physical_bytes[device_id]
                    capacity = self.device_config[device_id].capacity_bytes
                    if (
                        not state.device_allocated
                        and used + state.size_bytes > capacity
                    ):
                        self._raise_capacity(
                            kind="prefetch-device-capacity",
                            location=f"device:{device_id}",
                            capacity=capacity,
                            used=used,
                            requested=state.size_bytes,
                            aliases=(pending.alias_group_id,),
                        )
                elif (
                    not state.spill_allocated
                    and self.spill_bytes + state.size_bytes
                    > self.config.spill_capacity_bytes
                ):
                    self._raise_capacity(
                        kind="offload-spill-capacity",
                        location="host",
                        capacity=self.config.spill_capacity_bytes,
                        used=self.spill_bytes,
                        requested=state.size_bytes,
                        aliases=(pending.alias_group_id,),
                    )
        for task in self.tasks:
            if task.task_id not in self.unlaunched:
                continue
            if any(
                dependency not in self.completed for dependency in task.dependencies
            ):
                continue
            missing = self._task_missing_inputs(task)
            if missing:
                raise SimulationInfeasibleError(
                    f"task {task.task_id!r} has no progress source for inputs "
                    f"{missing}",
                    kind="task-input-deadlock",
                    time_ns=self.now_ns,
                    task_id=task.task_id,
                    alias_group_ids=missing,
                    location=f"device:{task.resource.device_id}",
                )
            profile = self.profile_by_id[task.profile_id]
            output_aliases = self._task_output_aliases(task)
            logical_requested = profile.workspace_bytes + sum(
                self.alias_state[alias_id].size_bytes
                for alias_id in output_aliases
                if not self.alias_state[alias_id].device_allocated
            )
            device_id = task.resource.device_id
            requested = max(self._task_start_delta(task, logical_requested), 0)
            used = self.device_physical_bytes[device_id]
            if used + requested > self.device_config[device_id].capacity_bytes:
                self._raise_capacity(
                    kind="task-device-capacity",
                    location=f"device:{device_id}",
                    capacity=self.device_config[device_id].capacity_bytes,
                    used=used,
                    requested=requested,
                    task_id=task.task_id,
                    aliases=output_aliases,
                )
        for queues in (self.pending_fetch, self.pending_evict):
            for queue in queues.values():
                if not queue:
                    continue
                pending = queue[0]
                raise SimulationInfeasibleError(
                    f"transfer for {pending.alias_group_id!r} has no progress source",
                    kind="transfer-deadlock",
                    time_ns=self.now_ns,
                    alias_group_ids=(pending.alias_group_id,),
                )
        raise SimulationInfeasibleError(
            "task dependencies cannot make progress",
            kind="dependency-deadlock",
            time_ns=self.now_ns,
        )

    def _check_final_residency(self) -> None:
        for residency in self.schedule.final_residency:
            state = self.alias_state[residency.alias_group_id]
            ready = (
                state.device_ready
                if residency.location is MemoryLocation.DEVICE
                else state.spill_ready
            )
            if not ready:
                raise SimulationInfeasibleError(
                    f"final {residency.location.value} residency missing for "
                    f"{residency.alias_group_id!r}",
                    kind="final-residency",
                    time_ns=self.now_ns,
                    alias_group_ids=(residency.alias_group_id,),
                    location=residency.location.value,
                )

    def run(self) -> SimulationResult:
        self._initialize_memory()
        while (
            self.unlaunched
            or self.active_tasks
            or any(self.pending_fetch.values())
            or any(self.pending_evict.values())
            or self.active_fetch
            or self.active_evict
        ):
            changed = True
            while changed:
                changed = self._try_start_transfers()
                changed |= self._try_launch_tasks()
            next_time = self._next_event_time()
            if next_time is None:
                self._deadlock()
            self.now_ns = next_time
            self._complete_events()
        self._check_final_residency()
        peaks = tuple(
            DeviceMemoryPeak(
                device_id=device_id,
                object_bytes=self.device_object_peaks[device_id],
                workspace_bytes=self.device_workspace_peaks[device_id],
                total_bytes=self.device_total_peaks[device_id],
            )
            for device_id in self.device_config
        )
        return SimulationResult(
            makespan_ns=self.now_ns,
            task_intervals=tuple(
                sorted(
                    self.task_intervals,
                    key=lambda item: self.task_order[item.task_id],
                )
            ),
            transfer_intervals=tuple(
                sorted(
                    self.transfer_intervals,
                    key=lambda item: (
                        item.start_ns,
                        item.direction.value,
                        item.device_id,
                        item.sequence,
                    ),
                )
            ),
            device_peaks=peaks,
            spill_peak_bytes=self.spill_peak_bytes,
            memory_timeline=tuple(self.memory_timeline),
        )


def simulate_python(
    program: Program,
    schedule: MemorySchedule,
    *,
    selections: tuple[RecomputationSelection, ...] = (),
    config: SimulationConfig,
    admission: SimulationAdmission | None = None,
    record_timeline: bool = False,
) -> SimulationResult:
    """Replay a validated schedule with the readable reference implementation."""

    return _Simulator(
        program,
        schedule,
        selections,
        config,
        admission=admission,
        record_timeline=record_timeline,
    ).run()


__all__ = ["simulate_python"]
