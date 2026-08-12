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
    "DeviceMemoryPeak",
    "DeviceSimulationConfig",
    "MemorySnapshot",
    "SimulationConfig",
    "SimulationInfeasibleError",
    "SimulationResult",
    "TaskInterval",
    "TransferDirection",
    "TransferInterval",
]
