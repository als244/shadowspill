"""Public immutable simulator inputs, intervals, and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from shadowspill.ir import ResourceKind


def _require_non_negative(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _require_positive(value: int, name: str) -> None:
    _require_non_negative(value, name)
    if value == 0:
        raise ValueError(f"{name} must be positive")


@dataclass(frozen=True, slots=True)
class DeviceSimulationConfig:
    """Capacity and transfer calibration for one logical device."""

    device_id: str
    capacity_bytes: int
    fetch_bandwidth_bytes_per_second: int
    evict_bandwidth_bytes_per_second: int
    fetch_latency_ns: int = 0
    evict_latency_ns: int = 0

    def __post_init__(self) -> None:
        if not self.device_id or self.device_id.strip() != self.device_id:
            raise ValueError("device_id must be a non-empty normalized string")
        _require_non_negative(self.capacity_bytes, "capacity_bytes")
        _require_positive(
            self.fetch_bandwidth_bytes_per_second,
            "fetch_bandwidth_bytes_per_second",
        )
        _require_positive(
            self.evict_bandwidth_bytes_per_second,
            "evict_bandwidth_bytes_per_second",
        )
        _require_non_negative(self.fetch_latency_ns, "fetch_latency_ns")
        _require_non_negative(self.evict_latency_ns, "evict_latency_ns")


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """Physical capacities and calibrated transfer costs for a replay."""

    devices: tuple[DeviceSimulationConfig, ...]
    host_capacity_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.devices, tuple) or not self.devices:
            raise ValueError("devices must be a non-empty tuple")
        seen: set[str] = set()
        for index, device in enumerate(self.devices):
            if not isinstance(device, DeviceSimulationConfig):
                raise ValueError(f"devices[{index}] has an invalid type")
            if device.device_id in seen:
                raise ValueError(f"devices[{index}] duplicates {device.device_id!r}")
            seen.add(device.device_id)
        _require_non_negative(self.host_capacity_bytes, "host_capacity_bytes")

    @classmethod
    def single_device(
        cls,
        device_id: str,
        *,
        device_capacity_bytes: int,
        host_capacity_bytes: int,
        fetch_bandwidth_bytes_per_second: int,
        evict_bandwidth_bytes_per_second: int,
        fetch_latency_ns: int = 0,
        evict_latency_ns: int = 0,
    ) -> SimulationConfig:
        return cls(
            devices=(
                DeviceSimulationConfig(
                    device_id=device_id,
                    capacity_bytes=device_capacity_bytes,
                    fetch_bandwidth_bytes_per_second=(fetch_bandwidth_bytes_per_second),
                    evict_bandwidth_bytes_per_second=(evict_bandwidth_bytes_per_second),
                    fetch_latency_ns=fetch_latency_ns,
                    evict_latency_ns=evict_latency_ns,
                ),
            ),
            host_capacity_bytes=host_capacity_bytes,
        )


@dataclass(frozen=True, slots=True)
class TaskPhysicalDelta:
    """Physical execution-pool change at one task's causal boundaries."""

    task_id: str
    start_bytes: int
    completion_bytes: int

    def __post_init__(self) -> None:
        if not self.task_id or self.task_id.strip() != self.task_id:
            raise ValueError("task_id must be a non-empty normalized string")
        if isinstance(self.start_bytes, bool) or not isinstance(self.start_bytes, int):
            raise ValueError("start_bytes must be an integer")
        if isinstance(self.completion_bytes, bool) or not isinstance(
            self.completion_bytes, int
        ):
            raise ValueError("completion_bytes must be an integer")


@dataclass(frozen=True, slots=True)
class ActionPhysicalDelta:
    """Physical execution-pool change at one memory action's boundaries."""

    action_index: int
    trigger_bytes: int
    completion_bytes: int

    def __post_init__(self) -> None:
        _require_non_negative(self.action_index, "action_index")
        if isinstance(self.trigger_bytes, bool) or not isinstance(
            self.trigger_bytes, int
        ):
            raise ValueError("trigger_bytes must be an integer")
        if isinstance(self.completion_bytes, bool) or not isinstance(
            self.completion_bytes, int
        ):
            raise ValueError("completion_bytes must be an integer")


@dataclass(frozen=True, slots=True)
class MemoryReuseDependency:
    """Require one eviction action to complete before a successor may run."""

    predecessor_action_index: int
    successor_task_id: str | None = None
    successor_action_index: int | None = None

    def __post_init__(self) -> None:
        _require_non_negative(
            self.predecessor_action_index, "predecessor_action_index"
        )
        if (self.successor_task_id is None) == (
            self.successor_action_index is None
        ):
            raise ValueError(
                "exactly one memory-reuse successor task or action is required"
            )
        if self.successor_task_id is not None and (
            not self.successor_task_id
            or self.successor_task_id.strip() != self.successor_task_id
        ):
            raise ValueError(
                "successor_task_id must be a non-empty normalized string"
            )
        if self.successor_action_index is not None:
            _require_non_negative(
                self.successor_action_index, "successor_action_index"
            )


@dataclass(frozen=True, slots=True)
class SimulationAdmission:
    """Timing-independent physical admission facts consumed by simulation."""

    initial_physical_bytes: tuple[tuple[str, int], ...]
    device_capacity_bytes: tuple[tuple[str, int], ...] = ()
    task_deltas: tuple[TaskPhysicalDelta, ...] = ()
    action_deltas: tuple[ActionPhysicalDelta, ...] = ()
    reuse_dependencies: tuple[MemoryReuseDependency, ...] = ()

    def __post_init__(self) -> None:
        _validate_unique_pairs(self.initial_physical_bytes, "initial_physical_bytes")
        for device_id, bytes_ in self.initial_physical_bytes:
            if not device_id or device_id.strip() != device_id:
                raise ValueError(
                    "initial physical device IDs must be normalized strings"
                )
            _require_non_negative(bytes_, "initial physical bytes")
        _validate_unique_pairs(self.device_capacity_bytes, "device_capacity_bytes")
        for device_id, bytes_ in self.device_capacity_bytes:
            if not device_id or device_id.strip() != device_id:
                raise ValueError(
                    "admission capacity device IDs must be normalized strings"
                )
            _require_positive(bytes_, "admission device capacity bytes")
        if self.device_capacity_bytes and {
            item[0] for item in self.device_capacity_bytes
        } != {item[0] for item in self.initial_physical_bytes}:
            raise ValueError(
                "admission device capacities and initial physical bytes must "
                "name the same devices"
            )
        _validate_unique_values(
            tuple(item.task_id for item in self.task_deltas), "task_deltas"
        )
        _validate_unique_values(
            tuple(item.action_index for item in self.action_deltas), "action_deltas"
        )


def _validate_unique_pairs(
    values: tuple[tuple[str, int], ...],
    field: str,
) -> None:
    _validate_unique_values(tuple(item[0] for item in values), field)


def _validate_unique_values(values: tuple[object, ...], field: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{field} contains duplicate identities")


class TransferDirection(StrEnum):
    FETCH = "fetch"
    EVICT = "evict"


@dataclass(frozen=True, slots=True)
class TaskInterval:
    task_id: str
    device_id: str
    resource_kind: ResourceKind
    resource_lane: int
    ready_ns: int
    start_ns: int
    end_ns: int
    workspace_bytes: int
    stall_reasons: tuple[str, ...] = ()

    @property
    def stall_ns(self) -> int:
        return self.start_ns - self.ready_ns


@dataclass(frozen=True, slots=True)
class TransferInterval:
    alias_group_id: str
    trigger_task_id: str
    device_id: str
    direction: TransferDirection
    sequence: int
    ready_ns: int
    start_ns: int
    end_ns: int
    bytes: int
    stall_reasons: tuple[str, ...] = ()

    @property
    def stall_ns(self) -> int:
        return self.start_ns - self.ready_ns


@dataclass(frozen=True, slots=True)
class DeviceMemoryPeak:
    device_id: str
    object_bytes: int
    workspace_bytes: int
    total_bytes: int


@dataclass(frozen=True, slots=True)
class MemorySnapshot:
    time_ns: int
    device_object_bytes: tuple[tuple[str, int], ...]
    device_workspace_bytes: tuple[tuple[str, int], ...]
    host_bytes: int
    device_physical_bytes: tuple[tuple[str, int], ...] = ()


class SimulationInfeasibleError(ValueError):
    """A deterministic schedule contradiction with machine-readable fields."""

    def __init__(
        self,
        message: str,
        *,
        kind: str,
        time_ns: int,
        task_id: str | None = None,
        alias_group_ids: tuple[str, ...] = (),
        location: str | None = None,
        capacity_bytes: int | None = None,
        used_bytes: int | None = None,
        requested_bytes: int | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.time_ns = time_ns
        self.task_id = task_id
        self.alias_group_ids = alias_group_ids
        self.location = location
        self.capacity_bytes = capacity_bytes
        self.used_bytes = used_bytes
        self.requested_bytes = requested_bytes


@dataclass(frozen=True, slots=True)
class SimulationResult:
    makespan_ns: int
    task_intervals: tuple[TaskInterval, ...]
    transfer_intervals: tuple[TransferInterval, ...]
    device_peaks: tuple[DeviceMemoryPeak, ...]
    host_peak_bytes: int
    memory_timeline: tuple[MemorySnapshot, ...] = ()

    def device_peak(self, device_id: str) -> DeviceMemoryPeak:
        for peak in self.device_peaks:
            if peak.device_id == device_id:
                return peak
        raise KeyError(device_id)


__all__ = [
    "ActionPhysicalDelta",
    "DeviceMemoryPeak",
    "DeviceSimulationConfig",
    "MemoryReuseDependency",
    "MemorySnapshot",
    "SimulationAdmission",
    "SimulationConfig",
    "SimulationInfeasibleError",
    "SimulationResult",
    "TaskInterval",
    "TaskPhysicalDelta",
    "TransferDirection",
    "TransferInterval",
]
