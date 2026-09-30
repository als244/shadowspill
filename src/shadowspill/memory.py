"""Framework-neutral configuration values for runtime pools and routes."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

# External memory is device memory outside ShadowSpill's allocation pool,
# excluding the process baseline measured before the pool is created. Library
# workspaces and loaded kernels can increase it lazily. Reserve space before
# creating the pool because live allocations prevent resizing that pool.
# 512 MiB covers the original qualification hardware; callers can override it
# for other hardware and workloads. Enforcement is a separate setting.
_DEFAULT_EXTERNAL_HEADROOM = 512 << 20


def _positive_bytes(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer byte count")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


@dataclass(frozen=True, slots=True)
class DevicePool:
    """Configuration for an accelerator execution-capable memory pool.

    ``physical_capacity`` is the process-attributable device memory budget.
    Pool sizing subtracts the initial process baseline and ``external_headroom``
    (512 MiB by default). Zero headroom reserves no external allowance.

    ``reject_overbudget=False`` (the default) reports external and whole-process
    memory overruns without rejecting them. Set it to ``True`` to enforce
    those limits. Pool bounds and actual device allocation failures remain
    enforced in either mode; this flag never changes the pool's size.
    """

    physical_capacity: int
    device: int = 0
    external_headroom: int = _DEFAULT_EXTERNAL_HEADROOM
    reject_overbudget: bool = False

    def __post_init__(self) -> None:
        _positive_bytes(self.physical_capacity, "physical_capacity")
        if not isinstance(self.reject_overbudget, bool):
            raise TypeError("reject_overbudget must be a bool")
        if isinstance(self.device, bool) or not isinstance(self.device, int):
            raise TypeError("device must be an integer accelerator ordinal")
        if self.device < 0:
            raise ValueError("device must be non-negative")
        if isinstance(self.external_headroom, bool) or not isinstance(
            self.external_headroom, int
        ):
            raise TypeError("external_headroom must be an integer byte count")
        if not 0 <= self.external_headroom < self.physical_capacity:
            raise ValueError(
                "external_headroom must be non-negative and smaller than "
                "physical_capacity"
            )


@dataclass(frozen=True, slots=True)
class SpillPool:
    """What every pool that is not the execution pool answers.

    A spill pool declares its own kind, and nothing here enumerates which kinds
    exist. That mirrors the runtime, where a pool's kind selects an
    acquire/release pair from a list the runtime seeds and a loaded library
    appends to: there is no branch asking what kind of memory a pool has, and
    adding one is an entry rather than an edit.

    A kind implemented in a separately loaded library subclasses this, names
    the library it needs, and returns whatever that library's ``acquire``
    expects from :meth:`configuration`.
    """

    capacity: int

    #: The ``ShadowSpillPoolKind`` value this pool's memory is looked up by.
    kind: int = 1

    #: What the pool registry reports for it.
    kind_name: str = "pinned_host"

    #: Whether this process can dereference a lease from this pool.
    #:
    #: Mirrors the one fact the runtime states by leaving ``write`` and
    #: ``read`` NULL on a kind's ``pool_memory`` entry, and is declared here
    #: for the same reason it is declared there: it is a property of the kind,
    #: known by whoever implements it, and not something a caller should have
    #: to guess. A ``ClassVar`` rather than a field because no instance may
    #: claim otherwise -- a pool that said it was addressable when it is not
    #: would fault rather than fail.
    #:
    #: False costs a host copy of any state the framework holds: the runtime
    #: copies it in when a plan adopts it and back out when the plan is done,
    #: through the kind's ``write`` and ``read``.
    addressable: ClassVar[bool] = True

    def __post_init__(self) -> None:
        _positive_bytes(self.capacity, "capacity")

    @property
    def library(self) -> Path | None:
        """The shared object supplying this kind, or ``None`` for a built-in.

        Loading happens once per distinct path, before the runtime is created,
        and the library stays open for as long as the runtime does.
        """

        return None

    def configuration(self) -> ctypes.Structure | None:
        """What this pool's kind is told about *this* pool.

        Passed to the kind's ``acquire`` untouched and never read by anything
        in between, which is how a pool says which machine or which device it
        wants without the neutral configuration learning any such word. The
        object returned must outlive the bootstrap call; a built-in kind needs
        none and returns ``None``.
        """

        return None


@dataclass(frozen=True, slots=True)
class PinnedHostPool(SpillPool):
    """Configuration for a bounded pinned-memory spill pool."""


MemoryPoolConfig = DevicePool | SpillPool


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
    external_headroom: int = _DEFAULT_EXTERNAL_HEADROOM,
    reject_overbudget: bool = False,
) -> DevicePool:
    """Return a default accelerator-device pool configuration."""

    return DevicePool(
        physical_capacity=physical_capacity,
        device=device,
        external_headroom=external_headroom,
        reject_overbudget=reject_overbudget,
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
    "SpillPool",
    "TransferRoute",
    "device",
    "pinned_host",
    "transfer_route",
]
