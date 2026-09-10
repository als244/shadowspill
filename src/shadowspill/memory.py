"""Framework-neutral configuration values for runtime pools and routes."""

from __future__ import annotations

from dataclasses import dataclass

# Third-party kernels -- cuBLAS, cuDNN, and any custom kernel a model pulls in
# (flash-attention, fla, and the like) -- allocate their workspaces outside the
# allocator, lazily, the first time each kernel runs. That is after the slab has
# been sized, and a conventional CUDA slab cannot be resized once a provider
# holds even one allocator-owned pointer, so the allowance has to be reserved up
# front. It stays inside the user's physical cap and is reported in PlanReport.
#
# The figure is the worst case measured across the qualification corpus, rounded
# the way the seal rounds: external high-water was 112 MiB for llama3 and olmoe
# and 386 MiB for qwen35, identical at both the numerical and performance
# geometries despite a 6 GiB spread in peak process memory -- the footprint
# follows which kernels are compiled in, not problem size. The seal requires
# round_up(measured + 64 MiB, 64 MiB) -- 192 MiB for llama3 and olmoe, 512 MiB
# for qwen35 -- so this is exactly what the corpus's worst case demands rather
# than a margin above it. A configuration whose provider footprint is known may
# choose a lower value; one whose provider needs more is told by the seal, with
# the figure, rather than left to guess.
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
    memory cap, including its problem and provider headroom. The runtime
    reports the derived suballocatable pool capacity after initialization.

    ``provider_headroom`` of ``0`` means "give the slab everything and tell me
    when a provider wants more": the runtime still bootstraps and still seals,
    but a process that grows past its cap is reported on stderr with the
    measured figures rather than refused. Use it to find out what a model's
    providers actually need; a run that has to honour the cap gives them room.
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


@dataclass(frozen=True, slots=True)
class TransferRoute:
    """One directed transfer capability between two named memory pools.

    The concrete backend is resolved from the endpoint pool implementations at
    runtime construction. Direction is immutable: callers never pass a copy
    direction to an already-created route.
    """

    source: str
    destination: str

    def __post_init__(self) -> None:
        for field, value in (
            ("source", self.source),
            ("destination", self.destination),
        ):
            if not isinstance(value, str) or not value or not value.isidentifier():
                raise ValueError(f"route {field} must be a non-empty pool identifier")
        if self.source == self.destination:
            raise ValueError("a directed transfer route requires distinct pools")


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


def transfer_route(*, source: str, destination: str) -> TransferRoute:
    """Return a directed route configuration between two named pools."""

    return TransferRoute(source=source, destination=destination)


__all__ = [
    "DevicePool",
    "MemoryPoolConfig",
    "PinnedHostPool",
    "TransferRoute",
    "device",
    "pinned_host",
    "transfer_route",
]
