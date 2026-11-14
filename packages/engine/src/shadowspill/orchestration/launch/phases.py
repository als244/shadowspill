"""Allocation-owned phase processes; phase selection remains in its existing driver."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import Event, RLock
from types import MappingProxyType
from typing import Self

from shadowspill_client._records import integer
from shadowspill_client.buffers import BufferBudget

from .admission import NodeAdmission, NodeLease, ReservationCancelled, ResourceRequest
from .log_files import LogFiles
from .supervision import LogStatus

PHASES = ("build", "profile", "communicate", "plan", "execute")


@dataclass(frozen=True)
class PhaseInvocation:
    """Trusted installed command and environment, never decoded request content."""

    phase: str
    command: tuple[str, ...]
    environment: Mapping[str, str]
    workers: int
    root: Path
    log_path: Path

    def __post_init__(self) -> None:
        if self.phase not in PHASES:
            raise ValueError("unknown installed phase")
        integer(self.workers, "phase workers", minimum=1)
        if self.phase == "plan" and self.workers != 1:
            raise ValueError("CPU planning requires one launcher process")
        object.__setattr__(self, "command", tuple(self.command))
        object.__setattr__(
            self, "environment", MappingProxyType(dict(self.environment))
        )


@dataclass(frozen=True)
class PhaseResult:
    returncode: int
    logs: tuple[Path, ...]
    log_statuses: tuple[LogStatus, ...] = ()


@dataclass
class _PhaseOwner:
    lease: NodeLease
    logs: LogFiles | None = None


class NodePhases:
    """One job's sequential phases on explicitly selected assigned devices.

    ``host_bytes`` is a deployment-supplied whole-phase bound, including every
    participant, framework/compiler process and decoded artifact. It is not
    inferred from execution pool sizes. Log scratch comes from the supplied
    node-backed arena; disk quota and files survive process/lease retirement.

    CPU planning reserves host memory only. Each phase terminates its complete
    owned process group before another starts. ``stop`` interrupts only this
    job's resource wait/children; call ``close`` after the launching thread exits
    and retained logs have been published or discarded.
    """

    def __init__(
        self,
        admission: NodeAdmission,
        *,
        devices: Sequence[str],
        host_bytes: int,
        root: Path,
        disk: BufferBudget,
        memory: BufferBudget,
        log_limit_bytes: int = 64 << 20,
        wait_for_resources: bool = False,
    ) -> None:
        if (
            isinstance(devices, (str, bytes))
            or len(set(devices)) != len(devices)
            or any(device not in admission.devices for device in devices)
        ):
            raise ValueError("phase devices must be distinct assigned slots")
        if type(wait_for_resources) is not bool:
            raise TypeError("wait_for_resources must be boolean")
        self.admission, self.devices = admission, tuple(devices)
        self.host_bytes = integer(host_bytes, "whole-phase host bound", minimum=1)
        self.root, self.disk, self.memory = root, disk, memory
        self.log_limit_bytes = integer(log_limit_bytes, "phase log bytes")
        self.wait_for_resources = wait_for_resources
        self._mutex, self._stopped = RLock(), Event()
        self._runs: dict[str, _PhaseOwner] = {}
        self._active = self.closed = False

    @property
    def logs(self) -> Mapping[str, LogFiles]:
        with self._mutex:
            return MappingProxyType(
                {
                    phase: run.logs
                    for phase, run in self._runs.items()
                    if run.logs is not None
                }
            )

    def __call__(self, invocation: PhaseInvocation) -> PhaseResult:
        with self._mutex:
            if self._stopped.is_set() or self.closed:
                raise RuntimeError("phase owner has stopped")
            if self._active or invocation.phase in self._runs:
                raise RuntimeError(
                    "phase owner requires one fresh sequential invocation"
                )
            self._active = True
        try:
            return self._run(invocation)
        finally:
            with self._mutex:
                self._active = False

    def _run(self, invocation: PhaseInvocation) -> PhaseResult:
        if invocation.phase != "plan" and invocation.workers > len(self.devices):
            raise ValueError("phase participants exceed the selected assigned slots")
        devices = (
            () if invocation.phase == "plan" else self.devices[: invocation.workers]
        )
        lease = self.admission.reserve(
            ResourceRequest(
                {
                    device: self.admission.devices[device].capacity_bytes
                    for device in devices
                },
                self.host_bytes,
            ),
            wait=self.wait_for_resources,
            cancelled=self._stopped,
        )
        try:
            with self._mutex:
                owner = _PhaseOwner(lease)
                self._runs[invocation.phase] = owner
                if self._stopped.is_set():
                    raise ReservationCancelled("phase owner stopped before launch")
                owner.logs = LogFiles(
                    lease,
                    None,
                    root=self.root,
                    disk=self.disk,
                    memory=self.memory,
                    limit_bytes=self.log_limit_bytes,
                )
                child = lease.launch_command(
                    invocation.command,
                    devices=devices,
                    environment=invocation.environment,
                    directory=owner.logs.directory,
                    cwd=invocation.root,
                    log_limit_bytes=owner.logs.limit_bytes,
                )
            returncode = child.wait()
        finally:
            # A failed stop retains the lease and log owner for explicit retry.
            lease.close()
        if self._stopped.is_set():
            raise ReservationCancelled("phase owner stopped during execution")
        assert owner.logs is not None
        return PhaseResult(
            returncode,
            tuple(
                owner.logs.directory / f"{stream}.log"
                for stream in ("stdout", "stderr")
            ),
            child.logs,
        )

    def stop(self) -> None:
        self._stopped.set()
        with self._mutex:
            owners = tuple(self._runs.values())
        errors = []
        for owner in owners:
            try:
                owner.lease.close()
            except BaseException as error:
                errors.append(error)
        if errors:
            raise BaseExceptionGroup("phase children did not finish cleanup", errors)

    def close(self) -> None:
        self.stop()
        with self._mutex:
            if self._active:
                raise RuntimeError(
                    "wait for the phase launching thread before closing logs"
                )
            errors = []
            for owner in self._runs.values():
                if owner.logs is not None:
                    try:
                        owner.logs.close()
                    except BaseException as error:
                        errors.append(error)
            if errors:
                raise BaseExceptionGroup("phase logs did not finish cleanup", errors)
            self.closed = True

    def __enter__(self) -> Self:
        if self._stopped.is_set() or self.closed:
            raise RuntimeError("phase owner has stopped")
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
