"""Readable deterministic simulator used as the differential oracle."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
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

from .model import (
    DeviceMemoryPeak,
    MemorySnapshot,
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
    retain_host_backing: bool
    device_allocated: bool = False
    device_ready: bool = False
    device_version: int = 0
    host_allocated: bool = False
    host_ready: bool = False
    host_version: int = 0
    h2d_pending: bool = False
    d2h_pending: bool = False


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
        record_timeline: bool,
    ) -> None:
        self.program = program
        self.schedule = schedule
        self.selections = selections
        self.config = config
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
        self._validate_inputs()
        self.alias_state = {
            item.alias_group_id: _AliasState(
                size_bytes=item.size_bytes,
                device_id=item.device_id,
                initial_version=item.initial_version,
                retain_host_backing=item.retain_host_backing,
                device_version=item.initial_version,
                host_version=item.initial_version,
            )
            for item in program.alias_groups
        }
        self.next_action_index = 0
        self.now_ns = 0
        self.unlaunched = {task.task_id for task in self.tasks}
        self.completed: dict[str, int] = {}
        self.active_tasks: dict[tuple[str, ResourceKind, int], _ActiveTask] = {}
        self.task_waits = {task.task_id: _TaskWait() for task in self.tasks}
        self.pending_h2d = {
            device.device_id: deque[_PendingTransfer]() for device in config.devices
        }
        self.pending_d2h = {
            device.device_id: deque[_PendingTransfer]() for device in config.devices
        }
        self.active_h2d: dict[str, _ActiveTransfer] = {}
        self.active_d2h: dict[str, _ActiveTransfer] = {}
        self.transfer_sequence: dict[tuple[str, TransferDirection], int] = {}
        self.device_object_bytes = {device.device_id: 0 for device in config.devices}
        self.device_workspace_bytes = {device.device_id: 0 for device in config.devices}
        self.device_object_peaks = {device.device_id: 0 for device in config.devices}
        self.device_workspace_peaks = {device.device_id: 0 for device in config.devices}
        self.device_total_peaks = {device.device_id: 0 for device in config.devices}
        self.host_bytes = 0
        self.host_peak_bytes = 0
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
                self.device_total_peaks[device_id], objects + workspace
            )
        self.host_peak_bytes = max(self.host_peak_bytes, self.host_bytes)
        if self.record_timeline:
            self.memory_timeline.append(
                MemorySnapshot(
                    time_ns=self.now_ns,
                    device_object_bytes=tuple(self.device_object_bytes.items()),
                    device_workspace_bytes=tuple(self.device_workspace_bytes.items()),
                    host_bytes=self.host_bytes,
                )
            )

    def _initialize_memory(self) -> None:
        for state in self.alias_state.values():
            if state.retain_host_backing:
                state.host_allocated = True
                state.host_ready = True
                self.host_bytes += state.size_bytes
        for residency in self.schedule.initial_residency:
            state = self.alias_state[residency.alias_group_id]
            if residency.location is MemoryLocation.DEVICE:
                state.device_allocated = True
                state.device_ready = True
                self.device_object_bytes[state.device_id] += state.size_bytes
            else:
                if not state.host_allocated:
                    state.host_allocated = True
                    self.host_bytes += state.size_bytes
                state.host_ready = True
        self._snapshot()
        for device_id, used in self.device_object_bytes.items():
            capacity = self.device_config[device_id].capacity_bytes
            if used > capacity:
                self._raise_capacity(
                    kind="initial-device-capacity",
                    location=f"device:{device_id}",
                    capacity=capacity,
                    used=used,
                    requested=0,
                )
        if self.host_bytes > self.config.host_capacity_bytes:
            self._raise_capacity(
                kind="initial-host-capacity",
                location="host",
                capacity=self.config.host_capacity_bytes,
                used=self.host_bytes,
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
            if not state.device_ready or state.h2d_pending or state.d2h_pending:
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
            used = (
                self.device_object_bytes[device_id]
                + self.device_workspace_bytes[device_id]
            )
            requested = new_output_bytes + profile.workspace_bytes
            if used + requested > self.device_config[device_id].capacity_bytes:
                wait.reasons.add("device-capacity")
                continue
            for alias_id in output_aliases:
                state = self.alias_state[alias_id]
                if not state.device_allocated:
                    state.device_allocated = True
                    self.device_object_bytes[device_id] += state.size_bytes
                state.device_ready = False
                state.h2d_pending = False
                state.d2h_pending = False
                state.host_ready = False
            self.device_workspace_bytes[device_id] += profile.workspace_bytes
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
        if direction is TransferDirection.HOST_TO_DEVICE:
            bandwidth = config.h2d_bandwidth_bytes_per_second
            latency = config.h2d_latency_ns
        else:
            bandwidth = config.d2h_bandwidth_bytes_per_second
            latency = config.d2h_latency_ns
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
        if direction is TransferDirection.HOST_TO_DEVICE:
            state.h2d_pending = True
            self.pending_h2d[state.device_id].append(pending)
        else:
            state.d2h_pending = True
            self.pending_d2h[state.device_id].append(pending)

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
        if state.h2d_pending or state.d2h_pending:
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
        if state.host_allocated and not state.retain_host_backing:
            state.host_allocated = False
            state.host_ready = False
            self.host_bytes -= state.size_bytes
        self._snapshot()

    def _submit_ready_actions(self) -> None:
        while self.next_action_index < len(self.schedule.actions):
            action_index = self.next_action_index
            action = self.schedule.actions[action_index]
            if action.trigger_task_id not in self.completed:
                return
            if action.kind is MemoryActionKind.RELEASE:
                self._release(action)
            elif action.kind is MemoryActionKind.OFFLOAD:
                state = self.alias_state[action.alias_group_id]
                if not state.device_allocated or not state.device_ready:
                    raise SimulationInfeasibleError(
                        f"offload of {action.alias_group_id!r} lacks a device source",
                        kind="invalid-offload",
                        time_ns=self.now_ns,
                        task_id=action.trigger_task_id,
                        alias_group_ids=(action.alias_group_id,),
                    )
                if not state.host_allocated:
                    if (
                        self.host_bytes + state.size_bytes
                        > self.config.host_capacity_bytes
                    ):
                        self._raise_capacity(
                            kind="offload-host-capacity",
                            location="host",
                            capacity=self.config.host_capacity_bytes,
                            used=self.host_bytes,
                            requested=state.size_bytes,
                            task_id=action.trigger_task_id,
                            aliases=(action.alias_group_id,),
                        )
                    state.host_allocated = True
                    state.host_ready = False
                    self.host_bytes += state.size_bytes
                self._enqueue_transfer(
                    action_index,
                    action,
                    TransferDirection.DEVICE_TO_HOST,
                )
            else:
                state = self.alias_state[action.alias_group_id]
                if state.device_allocated and not state.d2h_pending:
                    raise SimulationInfeasibleError(
                        f"prefetch of {action.alias_group_id!r} already has a "
                        "device copy",
                        kind="invalid-prefetch",
                        time_ns=self.now_ns,
                        task_id=action.trigger_task_id,
                        alias_group_ids=(action.alias_group_id,),
                    )
                if not state.host_ready and not state.d2h_pending:
                    raise SimulationInfeasibleError(
                        f"prefetch of {action.alias_group_id!r} lacks a host source",
                        kind="invalid-prefetch",
                        time_ns=self.now_ns,
                        task_id=action.trigger_task_id,
                        alias_group_ids=(action.alias_group_id,),
                    )
                if not state.device_allocated:
                    device_id = state.device_id
                    used = (
                        self.device_object_bytes[device_id]
                        + self.device_workspace_bytes[device_id]
                    )
                    capacity = self.device_config[device_id].capacity_bytes
                    if used + state.size_bytes > capacity:
                        self._raise_capacity(
                            kind="prefetch-device-capacity",
                            location=f"device:{device_id}",
                            capacity=capacity,
                            used=used,
                            requested=state.size_bytes,
                            task_id=action.trigger_task_id,
                            aliases=(action.alias_group_id,),
                        )
                    state.device_allocated = True
                    state.device_ready = False
                    self.device_object_bytes[device_id] += state.size_bytes
                self._enqueue_transfer(
                    action_index,
                    action,
                    TransferDirection.HOST_TO_DEVICE,
                )
            self._snapshot()
            self.next_action_index += 1

    def _complete_task(self, key: tuple[str, ResourceKind, int]) -> None:
        active = self.active_tasks.pop(key)
        task = active.task
        device_id = task.resource.device_id
        self.device_workspace_bytes[device_id] -= active.workspace_bytes
        for alias_id in active.output_aliases:
            state = self.alias_state[alias_id]
            state.device_ready = True
            state.device_version += 1
            state.host_ready = False
        for mutation in task.mutations:
            alias_id = self.object_alias[mutation.object_id]
            state = self.alias_state[alias_id]
            state.device_version += mutation.version_delta
            state.host_ready = False
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
        if direction is TransferDirection.HOST_TO_DEVICE:
            queue = self.pending_h2d[device_id]
            active_table = self.active_h2d
        else:
            queue = self.pending_d2h[device_id]
            active_table = self.active_d2h
        if device_id in active_table or not queue:
            return False
        pending = queue[0]
        state = self.alias_state[pending.alias_group_id]
        if direction is TransferDirection.HOST_TO_DEVICE:
            if state.d2h_pending or not state.host_ready:
                pending.stall_reasons.add("source-readiness")
                return False
            if not state.device_allocated:
                raise AssertionError("queued H2D has no trigger-time reservation")
        else:
            if not state.device_ready:
                pending.stall_reasons.add("source-readiness")
                return False
            if not state.host_allocated:
                raise AssertionError("queued D2H has no trigger-time reservation")
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
            changed |= self._try_start_direction(
                device_id, TransferDirection.HOST_TO_DEVICE
            )
            changed |= self._try_start_direction(
                device_id, TransferDirection.DEVICE_TO_HOST
            )
        return changed

    def _complete_transfer(
        self,
        device_id: str,
        direction: TransferDirection,
    ) -> None:
        if direction is TransferDirection.HOST_TO_DEVICE:
            active = self.active_h2d.pop(device_id)
        else:
            active = self.active_d2h.pop(device_id)
        pending = active.pending
        state = self.alias_state[pending.alias_group_id]
        if direction is TransferDirection.HOST_TO_DEVICE:
            state.device_ready = True
            state.device_version = state.host_version
            state.h2d_pending = False
            if not state.retain_host_backing:
                state.host_allocated = False
                state.host_ready = False
                self.host_bytes -= state.size_bytes
        else:
            state.host_ready = True
            state.host_version = state.device_version
            state.d2h_pending = False
            state.device_ready = False
            if not state.h2d_pending:
                state.device_allocated = False
                self.device_object_bytes[device_id] -= state.size_bytes
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
        ends.extend(active.end_ns for active in self.active_h2d.values())
        ends.extend(active.end_ns for active in self.active_d2h.values())
        return min(ends) if ends else None

    def _complete_events(self) -> None:
        for device_id in sorted(self.active_h2d):
            if self.active_h2d[device_id].end_ns == self.now_ns:
                self._complete_transfer(device_id, TransferDirection.HOST_TO_DEVICE)
        for device_id in sorted(self.active_d2h):
            if self.active_d2h[device_id].end_ns == self.now_ns:
                self._complete_transfer(device_id, TransferDirection.DEVICE_TO_HOST)
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
            (TransferDirection.HOST_TO_DEVICE, self.pending_h2d),
            (TransferDirection.DEVICE_TO_HOST, self.pending_d2h),
        ):
            for device_id, queue in queues.items():
                if not queue:
                    continue
                pending = queue[0]
                state = self.alias_state[pending.alias_group_id]
                if direction is TransferDirection.HOST_TO_DEVICE:
                    used = (
                        self.device_object_bytes[device_id]
                        + self.device_workspace_bytes[device_id]
                    )
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
                    not state.host_allocated
                    and self.host_bytes + state.size_bytes
                    > self.config.host_capacity_bytes
                ):
                    self._raise_capacity(
                        kind="offload-host-capacity",
                        location="host",
                        capacity=self.config.host_capacity_bytes,
                        used=self.host_bytes,
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
            requested = profile.workspace_bytes + sum(
                self.alias_state[alias_id].size_bytes
                for alias_id in output_aliases
                if not self.alias_state[alias_id].device_allocated
            )
            device_id = task.resource.device_id
            used = (
                self.device_object_bytes[device_id]
                + self.device_workspace_bytes[device_id]
            )
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
        for queues in (self.pending_h2d, self.pending_d2h):
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
                else state.host_ready
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
            or any(self.pending_h2d.values())
            or any(self.pending_d2h.values())
            or self.active_h2d
            or self.active_d2h
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
            host_peak_bytes=self.host_peak_bytes,
            memory_timeline=tuple(self.memory_timeline),
        )


def simulate_python(
    program: Program,
    schedule: MemorySchedule,
    *,
    selections: tuple[RecomputationSelection, ...] = (),
    config: SimulationConfig,
    record_timeline: bool = False,
) -> SimulationResult:
    """Replay a validated schedule with the readable reference implementation."""

    return _Simulator(
        program,
        schedule,
        selections,
        config,
        record_timeline=record_timeline,
    ).run()


__all__ = ["simulate_python"]
