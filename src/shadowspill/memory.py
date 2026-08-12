"""Framework-neutral configuration values for runtime memory pools."""

from __future__ import annotations

from dataclasses import dataclass

_DEFAULT_PROVIDER_HEADROOM = 512 << 20


def _positive_bytes(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer byte count")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


@dataclass(frozen=True, slots=True)
class DevicePool:
    """Configuration for an accelerator execution-capable memory pool.

    ``physical_capacity`` is the complete process-attributable accelerator
    memory cap, including its context and provider headroom. The runtime
    reports the derived suballocatable pool capacity after initialization.
    """

    physical_capacity: int
    device: int = 0
    provider_headroom: int = _DEFAULT_PROVIDER_HEADROOM

    def __post_init__(self) -> None:
        _positive_bytes(self.physical_capacity, "physical_capacity")
        if isinstance(self.device, bool) or not isinstance(self.device, int):
            raise TypeError("device must be an integer accelerator ordinal")
        if self.device < 0:
            raise ValueError("device must be non-negative")
        if isinstance(self.provider_headroom, bool) or not isinstance(
            self.provider_headroom, int
        ):
            raise TypeError("provider_headroom must be an integer byte count")
        if not 0 <= self.provider_headroom < self.physical_capacity:
            raise ValueError(
                "provider_headroom must be non-negative and smaller than "
                "physical_capacity"
            )


@dataclass(frozen=True, slots=True)
class PinnedHostPool:
    """Configuration for a bounded pinned-memory spill pool."""

    capacity: int

    def __post_init__(self) -> None:
        _positive_bytes(self.capacity, "capacity")


MemoryPoolConfig = DevicePool | PinnedHostPool


def device(
    *,
    physical_capacity: int,
    device: int = 0,
    provider_headroom: int = _DEFAULT_PROVIDER_HEADROOM,
) -> DevicePool:
    """Return a default accelerator-device pool configuration."""

    return DevicePool(
        physical_capacity=physical_capacity,
        device=device,
        provider_headroom=provider_headroom,
    )


def pinned_host(*, capacity: int) -> PinnedHostPool:
    """Return a pinned-host spill-pool configuration."""

    return PinnedHostPool(capacity=capacity)


__all__ = [
    "DevicePool",
    "MemoryPoolConfig",
    "PinnedHostPool",
    "device",
    "pinned_host",
]
