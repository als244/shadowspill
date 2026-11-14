"""Atomic admission on explicitly assigned node resources, before child launch."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import Condition, Event, RLock
from time import monotonic
from types import MappingProxyType
from typing import Self

from shadowspill_client._records import integer, number
from shadowspill_client.protocol.requests import identifier

from .connected import ConnectedProcess
from .supervision import OwnedProcess


@dataclass(frozen=True)
class AssignedDevice:
    id: str
    visibility: str
    capacity_bytes: int

    def __post_init__(self) -> None:
        identifier(self.id, "assigned device ID")
        identifier(self.visibility, "assigned device visibility")
        integer(self.capacity_bytes, "assigned device capacity", minimum=1)


@dataclass(frozen=True)
class NodeCapacity:
    """Trusted allocation facts/ceilings, never capacities supplied by a request.

    Discovery must limit these to assigned devices and actual node/cgroup memory.
    Host capacity excludes controller/agent reservations. Storage retains its own
    file-lifetime reservations; stopping a process does not delete its artifacts.
    This ledger does not probe hardware or reserve a scheduler allocation.
    """

    node_id: str
    devices: Sequence[AssignedDevice]
    host_bytes: int

    def __post_init__(self) -> None:
        identifier(self.node_id, "assigned node ID")
        integer(self.host_bytes, "assigned host capacity")
        if (
            not isinstance(self.devices, (tuple, list))
            or len(self.devices) > 65536
            or any(not isinstance(device, AssignedDevice) for device in self.devices)
        ):
            raise ValueError("provide a bounded list of assigned device descriptions")
        for field in ("id", "visibility"):
            if len({getattr(device, field) for device in self.devices}) != len(
                self.devices
            ):
                raise ValueError("assigned devices must have unique IDs and visibility")
        object.__setattr__(self, "devices", tuple(self.devices))


@dataclass(frozen=True)
class ResourceRequest:
    """Complete simultaneous node demand, including all live process overhead.

    Host demand includes spill, staging, decoded artifacts, framework/process and
    agent delivery reservations. Disk is admitted by the separate file owners.
    Device amounts are complete process budgets; a claimed slot is exclusive.
    """

    devices: Mapping[str, int]
    host_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.devices, Mapping) or len(self.devices) > 65536:
            raise ValueError("provide bounded assigned device byte requirements")
        for name, size in self.devices.items():
            identifier(name, "requested device ID")
            integer(size, "requested device bytes", minimum=1)
        integer(self.host_bytes, "requested host bytes")
        object.__setattr__(self, "devices", MappingProxyType(dict(self.devices)))


class ResourcesUnavailable(RuntimeError):
    """An admissible request is temporarily blocked by existing live owners."""


class ReservationCancelled(RuntimeError):
    """The owner stopped waiting before it acquired any node resources."""


@dataclass(frozen=True)
class ResourceUsage:
    host_bytes: int
    devices: frozenset[str]
    claims: int


class NodeAdmission:
    def __init__(self, capacity: NodeCapacity, *, max_claims: int = 1024) -> None:
        if not isinstance(capacity, NodeCapacity):
            raise TypeError("node admission requires trusted allocation capacity")
        self.capacity = capacity
        self.max_claims = integer(max_claims, "maximum live claims", minimum=1)
        self.devices = MappingProxyType(
            {device.id: device for device in capacity.devices}
        )
        self._condition = Condition()
        self._claims: set[NodeLease] = set()
        self._host = 0
        self._devices: set[str] = set()
        self.closed = False

    @property
    def used(self) -> ResourceUsage:
        with self._condition:
            return ResourceUsage(
                self._host, frozenset(self._devices), len(self._claims)
            )

    def reserve(
        self,
        request: ResourceRequest,
        *,
        wait: bool = False,
        timeout: float | None = None,
        cancelled: Event | None = None,
    ) -> "NodeLease":
        if not isinstance(request, ResourceRequest):
            raise TypeError("node reservation requires a complete resource request")
        if type(wait) is not bool or (timeout is not None and not wait):
            raise ValueError("a reservation timeout requires wait=True")
        if timeout is not None:
            number(timeout, "resource wait timeout", minimum=0)
        if request.host_bytes > self.capacity.host_bytes or any(
            name not in self.devices or size > self.devices[name].capacity_bytes
            for name, size in request.devices.items()
        ):
            raise ValueError("request cannot fit the assigned node resources")
        deadline = None if timeout is None else monotonic() + timeout
        with self._condition:
            while True:
                if self.closed:
                    raise RuntimeError("node admission is closed")
                if cancelled is not None and cancelled.is_set():
                    raise ReservationCancelled("node resource wait was cancelled")
                if (
                    self._host + request.host_bytes <= self.capacity.host_bytes
                    and not self._devices.intersection(request.devices)
                    and len(self._claims) < self.max_claims
                ):
                    owner = NodeLease(self, request)
                    self._claims.add(owner)
                    self._host += request.host_bytes
                    self._devices.update(request.devices)
                    return owner
                if not wait:
                    raise ResourcesUnavailable("assigned resources have live owners")
                remaining = None if deadline is None else deadline - monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError("timed out waiting for assigned resources")
                # A job owner can stop its own resource wait without closing the
                # shared allocation ledger or touching other live runs.
                if cancelled is not None:
                    remaining = 0.1 if remaining is None else min(remaining, 0.1)
                self._condition.wait(remaining)

    def _release(self, owner: "NodeLease") -> None:
        with self._condition:
            self._claims.remove(owner)
            self._host -= owner.request.host_bytes
            self._devices.difference_update(owner.request.devices)
            self._condition.notify_all()

    def close(self) -> None:
        """Wake/refuse new callers; live leases still own their child cleanup."""
        with self._condition:
            self.closed = True
            self._condition.notify_all()


class NodeLease:
    def __init__(self, owner: NodeAdmission, request: ResourceRequest) -> None:
        self._owner, self.request = owner, request
        self._children: dict[str, ConnectedProcess] = {}
        self._command: OwnedProcess | None = None
        self._lock = RLock()
        self._closing = self.closed = False

    def launch(
        self,
        device_id: str,
        command: Callable[[int], Sequence[str]],
        *,
        environment: Mapping[str, str],
        directory: Path,
        cwd: Path,
        log_limit_bytes: int = 64 << 20,
        startup_timeout: float = 120,
    ) -> ConnectedProcess:
        """Launch once on a reserved slot, setting visibility before any import."""
        with self._lock:
            if self._closing or self.closed:
                raise RuntimeError("resource lease is closing or closed")
            if (
                device_id not in self.request.devices
                or device_id in self._children
                or self._command is not None
            ):
                raise ValueError("worker device is unreserved or already launched")
            child = ConnectedProcess(
                command,
                environment={
                    **environment,
                    "CUDA_VISIBLE_DEVICES": self._owner.devices[device_id].visibility,
                },
                directory=directory,
                cwd=cwd,
                log_limit_bytes=log_limit_bytes,
                startup_timeout=startup_timeout,
            )
            self._children[device_id] = child
            return child

    def launch_command(
        self,
        command: Sequence[str],
        *,
        devices: Sequence[str],
        environment: Mapping[str, str],
        directory: Path,
        cwd: Path,
        log_limit_bytes: int = 64 << 20,
    ) -> OwnedProcess:
        """Own one installed phase/torchrun leader over this entire reservation.

        Device order supplies process-local ordinals before framework import.
        Host-only phases use an empty reservation and an empty visibility mask.
        The same lease cannot also launch persistent per-device workers.
        """
        with self._lock:
            if self._closing or self.closed:
                raise RuntimeError("resource lease is closing or closed")
            if (
                isinstance(devices, (str, bytes))
                or len(set(devices)) != len(devices)
                or set(devices) != set(self.request.devices)
                or self._children
                or self._command is not None
            ):
                raise ValueError(
                    "phase command must own every reserved slot exactly once"
                )
            child = OwnedProcess(
                command,
                environment={
                    **environment,
                    "CUDA_VISIBLE_DEVICES": ",".join(
                        self._owner.devices[device].visibility for device in devices
                    ),
                },
                directory=directory,
                cwd=cwd,
                log_limit_bytes=log_limit_bytes,
            )
            self._command = child
            return child

    def close(self) -> None:
        with self._lock:
            if self.closed:
                return
            self._closing = True
            errors: list[BaseException] = []
            # All peers see EOF before we wait on any collective participant.
            for child in self._children.values():
                try:
                    child.disconnect()
                except BaseException as error:
                    errors.append(error)
            if self._command is not None:
                try:
                    self._command.close()
                    if not self._command.closed:
                        raise RuntimeError("owned phase did not confirm closure")
                except BaseException as error:
                    errors.append(error)
            for child in self._children.values():
                try:
                    child.close()
                    if not child.closed:
                        raise RuntimeError("owned child did not confirm closure")
                except BaseException as error:
                    errors.append(error)
            if errors:
                # Keep every reservation until a subsequent cleanup succeeds.
                raise BaseExceptionGroup("node children did not finish cleanup", errors)
            self._owner._release(self)
            self._children.clear()
            self.closed = True

    def __enter__(self) -> Self:
        if self._closing or self.closed:
            raise RuntimeError("resource lease is closing or closed")
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
