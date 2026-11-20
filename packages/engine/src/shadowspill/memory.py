"""Framework-neutral configuration values for runtime pools and routes."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

# External memory is device memory outside ShadowSpill's allocation pool,
# excluding the process baseline measured before the pool is created. Library
# workspaces and loaded kernels can increase it lazily. Reserve space before
# creating the pool because live allocations prevent resizing that pool.
# 512 MiB is sized for the qualification corpus; callers can override it for
# other hardware and workloads. Enforcement is a separate setting.
_DEFAULT_EXTERNAL_HEADROOM = 512 << 20


def _positive_bytes(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer byte count")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


@dataclass(frozen=True, slots=True)
class DevicePool:
    """The configuration of the device pool, which serves a plan's allocations.

    `~shadowspill.runtime.Runtime` takes exactly one among its ``pools``.
    When the runtime opens, the pool's size is a limit less the larger of the
    process's device memory at that moment and ``external_baseline``, less
    ``external_headroom``, rounded down to the backend's alignment. The limit
    is ``physical_capacity``; with ``physical_limit``, it is the least of
    that, ``physical_limit``, and the device's free memory plus the process's
    own less ``unattributed_reservation``. Opening fails if
    ``physical_capacity`` exceeds the device's memory or no pool is left.
    External memory is the process's device memory outside the pool, beyond
    what it held at open.

    Parameters
    ----------
    physical_capacity : int
        The process's device memory budget, in bytes: the pool and its
        external memory together. Positive.
    device : int, default 0
        Ordinal of the accelerator the pool is on. Nonnegative.
    external_headroom : int, default 512 MiB
        Device memory kept for external memory that grows after the pool
        exists, such as library workspaces and loaded kernels, in bytes.
        Nonnegative and smaller than ``physical_capacity``.
    reject_overbudget : bool, default False
        Whether an overrun fails: the process holding more than
        ``physical_capacity``, or external memory outgrowing what was kept
        for it, when the runtime opens or later checks the budget. By default
        each overrun is reported and execution goes on. Either way the pool
        keeps its size and allocations beyond it fail.
    external_baseline : int, default 0
        Device memory the process is expected to hold outside the pool, in
        bytes, such as the footprint profiling observed once it stopped
        growing; allocations the pool serves are not part of it. The pool
        leaves room for the larger of this and what the process holds at
        open. Zero uses what it holds at open. At least zero and less than
        ``physical_capacity`` less ``external_headroom``.
    physical_limit : int, optional
        Device memory available to the process with the spill pool live, in
        bytes, as `PoolMemoryProfile.physical_limit_bytes` measures it.
        Positive; None leaves ``physical_capacity`` as the limit.
    unattributed_reservation : int, default 0
        Device memory outside the process that the spill pool is measured to
        take (`PoolMemoryProfile.global_reservation_bytes`), in bytes,
        subtracted from what is available at open. Requires
        ``physical_limit``.

    Raises
    ------
    TypeError
        If ``physical_capacity``, ``physical_limit``, ``external_headroom``,
        ``external_baseline`` or ``device`` is not an integer, or
        ``reject_overbudget`` is not a bool.
    ValueError
        If ``physical_capacity`` or ``physical_limit`` is not positive,
        ``device`` is negative, ``external_headroom`` or
        ``external_baseline`` is out of range, or ``unattributed_reservation``
        is not a nonnegative integer or is set without ``physical_limit``.
    """

    physical_capacity: int
    device: int = 0
    external_headroom: int = _DEFAULT_EXTERNAL_HEADROOM
    reject_overbudget: bool = False
    external_baseline: int = field(default=0, kw_only=True)
    physical_limit: int | None = field(default=None, kw_only=True)
    unattributed_reservation: int = field(default=0, kw_only=True)

    def __post_init__(self) -> None:
        _positive_bytes(self.physical_capacity, "physical_capacity")
        if self.physical_limit is not None:
            _positive_bytes(self.physical_limit, "physical_limit")
        if (
            type(self.unattributed_reservation) is not int
            or not 0 <= self.unattributed_reservation < 1 << 64
        ):
            raise ValueError("unattributed_reservation must be a uint64 byte count")
        if self.physical_limit is None and self.unattributed_reservation:
            raise ValueError("unattributed_reservation requires physical_limit")
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
        if isinstance(self.external_baseline, bool) or not isinstance(
            self.external_baseline, int
        ):
            raise TypeError("external_baseline must be an integer byte count")
        if (
            not 0
            <= self.external_baseline
            < (self.physical_capacity - self.external_headroom)
        ):
            raise ValueError(
                "external_baseline must be non-negative and leave room after "
                "external_headroom inside physical_capacity"
            )


@dataclass(frozen=True, slots=True)
class SpillPool:
    """The configuration of a spill pool, which holds values off the device.

    `Executor`, `measure_communication` and `measure_pool_memory` take one as
    ``spill``. `PinnedHostPool` is the built-in kind. A pool declares its own
    kind, which selects the runtime's acquire and release functions. A kind
    implemented in a separately loaded library subclasses this, names the
    library in `library`, and returns from `configuration` what that library's
    ``acquire`` expects.

    Parameters
    ----------
    capacity : int
        Pool size, in bytes. Positive.
    kind : int, default 1
        The runtime's ``ShadowSpillPoolKind`` value for the pool's memory.
    kind_name : str, default "pinned_host"
        The kind's name, as the runtime's pool registry reports it.

    Raises
    ------
    TypeError
        If ``capacity`` is not an integer.
    ValueError
        If ``capacity`` is not positive.
    """

    capacity: int

    #: The ``ShadowSpillPoolKind`` value this pool's memory is looked up by.
    kind: int = 1

    #: What the pool registry reports for it.
    kind_name: str = "pinned_host"

    #: Whether this process can read and write the pool's leases directly.
    #:
    #: A fact about the kind, as the runtime states it by leaving ``write`` and
    #: ``read`` NULL on the kind's ``pool_memory`` entry. Values in a pool that
    #: is not addressable are staged one object at a time through the kind's
    #: ``write`` and ``read``.
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
        """Return what the pool's kind is told about this pool.

        The runtime passes it to the kind's ``acquire`` unread. It must outlive
        the runtime's creation.

        Returns
        -------
        ctypes.Structure or None
            The kind's configuration; None for a built-in kind, which needs
            none.
        """

        return None

    def calibration_identity(self) -> dict[str, Any]:
        """Return the pool's configuration as stable JSON, for pool memory evidence.

        A configured extension pool overrides it to return every option that
        affects acquisition or the transfer lanes, never bytes holding
        pointers; the measuring code adds library hashes. It acquires nothing.

        Returns
        -------
        dict
            The options; empty for a pool without a `configuration`.

        Raises
        ------
        ValueError
            If the pool has a `configuration` and does not override this.
        """
        if self.configuration() is not None:
            raise ValueError("configured spill pools must define calibration_identity")
        return {}

    def accepts_memory_profile_capacity(self, measured_capacity: int) -> bool:
        """Return whether pool memory measured at a capacity applies to this pool.

        Configuration and implementation identity are checked separately.
        By default it requires the exact capacity; a kind accepts others only
        when its implementation guarantees that the measured reservations
        cover this pool.

        Parameters
        ----------
        measured_capacity : int
            Capacity of the measured pool, in bytes.

        Returns
        -------
        bool
            Whether the measurement covers this pool's retained overhead.
        """
        return self.capacity == measured_capacity


@dataclass(frozen=True, slots=True)
class PinnedHostPool(SpillPool):
    """A spill pool in pinned host memory, the built-in kind.

    `pinned_host` returns one. When the runtime opens, it maps the whole
    ``capacity``, faults it in and registers it with the device, so all of it
    stays resident until the runtime closes.

    Parameters
    ----------
    capacity : int
        Pool size, in bytes. Positive.
    kind : int, default 1
        The runtime's ``ShadowSpillPoolKind`` value for pinned host memory.
    kind_name : str, default "pinned_host"
        The kind's name, as the runtime's pool registry reports it.

    Raises
    ------
    TypeError
        If ``capacity`` is not an integer.
    ValueError
        If ``capacity`` is not positive.
    """

    def accepts_memory_profile_capacity(self, measured_capacity: int) -> bool:
        """Return whether pool memory measured at a capacity applies to this pool.

        The pool's lanes and fixed resources do not shrink with its capacity,
        so what was measured for a pool at least as large covers it.

        Parameters
        ----------
        measured_capacity : int
            Capacity of the measured pool, in bytes.

        Returns
        -------
        bool
            Whether ``capacity`` is at most ``measured_capacity``.
        """
        # One host arena is registered in full; lanes and their fixed resources
        # do not grow when the arena shrinks. Retain all larger-pool device
        # reservations instead of scaling or inventing a smaller observation.
        return self.capacity <= measured_capacity


#: A pool configuration, as a runtime takes them in ``pools``: a `DevicePool`
#: or a `SpillPool`.
MemoryPoolConfig = DevicePool | SpillPool


@dataclass(frozen=True, slots=True)
class TransferRoute:
    """A route along which a runtime moves bytes from one pool to another.

    `~shadowspill.runtime.Runtime` takes routes by name in ``routes``. One
    endpoint must be the device pool; the runtime picks the lane that moves
    the bytes from the two pools' kinds when it opens.

    Parameters
    ----------
    source : str
        Name of the pool bytes are copied from: a Python identifier.
    destination : str
        Name of the pool bytes are copied into: a Python identifier other
        than ``source``.

    Raises
    ------
    ValueError
        If a name is not a nonempty identifier, or the two are the same.
    """

    source: str
    destination: str

    def __post_init__(self) -> None:
        for name, value in (
            ("source", self.source),
            ("destination", self.destination),
        ):
            if not isinstance(value, str) or not value or not value.isidentifier():
                raise ValueError(f"route {name} must be a non-empty pool identifier")
        if self.source == self.destination:
            raise ValueError("a directed transfer route requires distinct pools")


def device(
    *,
    physical_capacity: int,
    device: int = 0,
    external_headroom: int = _DEFAULT_EXTERNAL_HEADROOM,
    reject_overbudget: bool = False,
    external_baseline: int = 0,
    physical_limit: int | None = None,
    unattributed_reservation: int = 0,
) -> DevicePool:
    """Return a device pool configuration.

    Parameters
    ----------
    physical_capacity : int
        The process's device memory budget, in bytes, as in `DevicePool`.
    device : int, default 0
        Ordinal of the accelerator the pool is on.
    external_headroom : int, default 512 MiB
        Device memory kept for external memory, in bytes.
    reject_overbudget : bool, default False
        Whether overrunning the budget fails rather than being reported.
    external_baseline : int, default 0
        Device memory the process is expected to hold outside the pool, in
        bytes.
    physical_limit : int, optional
        Device memory available to the process with the spill pool live, in
        bytes.
    unattributed_reservation : int, default 0
        Device memory outside the process that the spill pool takes, in
        bytes. Requires ``physical_limit``.

    Returns
    -------
    DevicePool
        The configuration.

    Raises
    ------
    TypeError
        If an argument has the wrong type, as `DevicePool` describes.
    ValueError
        If an argument is out of range, as `DevicePool` describes.
    """

    return DevicePool(
        physical_capacity=physical_capacity,
        device=device,
        external_headroom=external_headroom,
        reject_overbudget=reject_overbudget,
        external_baseline=external_baseline,
        physical_limit=physical_limit,
        unattributed_reservation=unattributed_reservation,
    )


def pinned_host(*, capacity: int) -> PinnedHostPool:
    """Return a pinned host spill pool configuration.

    Parameters
    ----------
    capacity : int
        Pool size, in bytes. Positive.

    Returns
    -------
    PinnedHostPool
        The configuration.

    Raises
    ------
    TypeError
        If ``capacity`` is not an integer.
    ValueError
        If ``capacity`` is not positive.
    """

    return PinnedHostPool(capacity=capacity)


def transfer_route(*, source: str, destination: str) -> TransferRoute:
    """Return a route configuration from one named pool to another.

    Parameters
    ----------
    source : str
        Name of the pool bytes are copied from.
    destination : str
        Name of the pool bytes are copied into.

    Returns
    -------
    TransferRoute
        The configuration.

    Raises
    ------
    ValueError
        If a name is not a nonempty identifier, or the two are the same.
    """

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
