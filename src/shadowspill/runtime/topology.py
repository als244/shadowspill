"""How one initialized runtime arranges its memory, and what it costs.

The pools a plan can be placed into, the routes between them, and the
bandwidth and latency the runtime measured for each route. Planning
reads all of it and has no other way to know any of it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MemoryPool:
    """One initialized runtime pool visible to planning."""

    name: str
    pool_id: int
    kind: str
    capacity: int
    physical_capacity: int | None
    device_ordinal: int | None


@dataclass(frozen=True, slots=True)
class RuntimeRoute:
    """One initialized directed route visible to planning."""

    name: str
    route_id: int
    source: str
    destination: str
    source_pool_id: int
    destination_pool_id: int


@dataclass(frozen=True, slots=True)
class TransferProfile:
    """Measured performance for one directed pool-pair route."""

    source: str
    destination: str
    source_pool_id: int
    destination_pool_id: int
    generation: int
    latency_nanoseconds: int
    bandwidth_bytes_per_second: int
    solo_bandwidth_bytes_per_second: int
    concurrent_bandwidth_bytes_per_second: int
    solo_measurement_nanoseconds: int
    concurrent_measurement_nanoseconds: int
    calibrated_timestamp_nanoseconds: int
    small_copy_bytes: int
    large_copy_bytes: int
    measured_copies: int
    available: bool
    calibrated: bool
    provenance: str
    calibration_mode: str
    concurrent_route_count: int

    def as_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "destination": self.destination,
            "source_pool_id": self.source_pool_id,
            "destination_pool_id": self.destination_pool_id,
            "generation": self.generation,
            "latency_nanoseconds": self.latency_nanoseconds,
            "bandwidth_bytes_per_second": self.bandwidth_bytes_per_second,
            "solo_bandwidth_bytes_per_second": self.solo_bandwidth_bytes_per_second,
            "concurrent_bandwidth_bytes_per_second": (
                self.concurrent_bandwidth_bytes_per_second
            ),
            "solo_measurement_nanoseconds": self.solo_measurement_nanoseconds,
            "concurrent_measurement_nanoseconds": (
                self.concurrent_measurement_nanoseconds
            ),
            "calibrated_timestamp_nanoseconds": self.calibrated_timestamp_nanoseconds,
            "small_copy_bytes": self.small_copy_bytes,
            "large_copy_bytes": self.large_copy_bytes,
            "measured_copies": self.measured_copies,
            "available": self.available,
            "calibrated": self.calibrated,
            "provenance": self.provenance,
            "calibration_mode": self.calibration_mode,
            "concurrent_route_count": self.concurrent_route_count,
        }


@dataclass(frozen=True, slots=True)
class TransferCapabilities:
    """Immutable indexed transfer matrix published by one runtime generation."""

    generation: int
    pool_names: tuple[str, ...]
    profiles: tuple[TransferProfile, ...]
    digest: str

    def route(self, source: str, destination: str) -> TransferProfile:
        try:
            source_id = self.pool_names.index(source)
            destination_id = self.pool_names.index(destination)
        except ValueError as exc:
            raise KeyError((source, destination)) from exc
        return self.profiles[source_id * len(self.pool_names) + destination_id]

    def as_dict(self) -> dict[str, object]:
        return {
            "generation": self.generation,
            "pool_names": list(self.pool_names),
            "profiles": [profile.as_dict() for profile in self.profiles],
            "digest": self.digest,
        }


__all__ = [
    "MemoryPool",
    "RuntimeRoute",
    "TransferCapabilities",
    "TransferProfile",
]
