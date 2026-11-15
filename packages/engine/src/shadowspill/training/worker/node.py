"""Own one node's saved workers, startup artifacts, grants and bounded logs."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Lock, RLock
from types import MappingProxyType
from typing import BinaryIO, Self

from shadowspill.orchestration.capture.files import CaptureFiles, CollectorLimits
from shadowspill.orchestration.capture.nsys import NsysCapture
from shadowspill.orchestration.launch.admission import NodeAdmission, NodeLease
from shadowspill.orchestration.launch.file_grants import FileGrant, FileGrants
from shadowspill.orchestration.launch.host_arena import HostPartition
from shadowspill.orchestration.launch.log_files import LogFiles
from shadowspill.orchestration.launch.publication import OutputFiles
from shadowspill.orchestration.launch.staging import FileDelivery
from shadowspill_client._records import integer
from shadowspill_client.buffers import BufferBudget, BufferLease
from shadowspill_client.protocol.codec import MAX_METADATA_BYTES
from shadowspill_client.protocol.requests import ArtifactRef

from .launch import WorkerProcess, launch_worker
from .protocol import CODEC, WorkerLaunch
from .requests import NodeFiles
from .resources import WorkerHost, worker_request
from .runtime import PreparedWorker


class WorkerNode:
    """A controller-owned lifecycle, without another execution/state machine.

    Construction validates configuration without acquiring resources. ``start``
    reserves the complete physical worker subset and all startup files/logs before
    launching any child. The controller combines participants across nodes using
    the existing startup barrier. It retains this owner even if start/close fails.

    ``stop`` only confirms child termination. WorkerRequests must then finish its
    output/input owners before ``close`` removes this node's startup/file services.
    Request deliveries use the caller's scratch root, so a partially prepared
    request cannot be deleted by cleanup of this node's private startup directory.
    The caller also owns the node-backed memory budget and persistent cache.
    """

    def __init__(
        self,
        admission: NodeAdmission,
        prepared: Mapping[int, PreparedWorker],
        devices: Mapping[int, str],
        host: Mapping[int, WorkerHost],
        *,
        packages: Sequence[str],
        weights: Sequence[ArtifactRef] = (),
        root: Path,
        cache: Path,
        disk: BufferBudget,
        memory: BufferBudget,
        environment: Mapping[str, str],
        log_limit_bytes: int = 64 << 20,
        host_memory: HostPartition | None = None,
        capture_limits: CollectorLimits | None = None,
    ) -> None:
        if (
            not prepared
            or prepared.keys() != devices.keys()
            or prepared.keys() != host.keys()
        ):
            raise ValueError(
                "node workers require matching ranks, devices and host allowances"
            )
        if len(set(devices.values())) != len(devices):
            raise ValueError("physical worker devices cannot be repeated")
        if any(rank != worker.startup.scope.rank for rank, worker in prepared.items()):
            raise ValueError(
                "prepared worker identities differ from assigned logical ranks"
            )
        self._launches = {
            rank: WorkerLaunch(worker.startup, packages, weights, worker.restore)
            for rank, worker in sorted(prepared.items())
        }
        for launch in self._launches.values():
            CODEC.encode(launch)
        self._request = worker_request(
            {
                devices[rank]: (worker.startup, host[rank])
                for rank, worker in prepared.items()
            }
        )
        captures = [
            NsysCapture.from_settings(worker.bindings.setup.config.execution.nsys)
            for worker in prepared.values()
        ]
        self._capture_settings = captures[0]
        if any(settings != self._capture_settings for settings in captures):
            raise ValueError("node workers disagree on Nsight settings")
        if (self._capture_settings is None) != (capture_limits is None):
            raise ValueError("Nsight workers require explicit collector allowances")
        self._capture_limits = capture_limits
        if capture_limits is not None:
            self._request = replace(
                self._request,
                host_bytes=self._request.host_bytes
                + len(prepared) * capture_limits.host_bytes,
            )
        self._admission = admission
        self._host_memory = host_memory
        if host_memory is not None:
            host_memory.require_node(admission)
            if host_memory.budget.max_bytes < self._request.host_bytes:
                raise ValueError(
                    "worker host slice cannot cover the complete node group"
                )
        self.devices = MappingProxyType(dict(devices))
        self.prepared = MappingProxyType(dict(prepared))
        self._parent, self._cache = root.resolve(), cache.resolve()
        self._disk, self._memory = disk, memory
        self._environment = MappingProxyType(dict(environment))
        self._log_limit = integer(log_limit_bytes, "per-stream log bytes")
        self._lock, self._busy = RLock(), Lock()
        self._cancelled = Event()
        self._lease: NodeLease | None = None
        self._metadata: BufferLease | None = None
        self._directory: TemporaryDirectory[str] | None = None
        self._delivery: FileDelivery | None = None
        self._grants: FileGrants | None = None
        self._grant: FileGrant | None = None
        self._outputs: OutputFiles | None = None
        self._logs: dict[int, LogFiles] = {}
        self._captures: dict[int, CaptureFiles] = {}
        self._processes: dict[int, WorkerProcess] = {}
        self._files: NodeFiles | None = None
        self._started = self._stopping = self.launched = self.closed = False

    @property
    def lease(self) -> NodeLease:
        if self._lease is None:
            raise RuntimeError("node workers have no resource reservation")
        return self._lease

    @property
    def files(self) -> NodeFiles:
        if not self.launched or self.closed or self._files is None:
            raise RuntimeError("node file services are not available")
        return self._files

    @property
    def logs(self) -> Mapping[int, LogFiles]:
        return MappingProxyType(dict(self._logs))

    @property
    def captures(self) -> Mapping[int, CaptureFiles]:
        return MappingProxyType(dict(self._captures))

    def _active(self) -> None:
        with self._lock:
            if self._stopping or self.closed:
                raise RuntimeError("node worker ownership is stopping or closed")

    def start(
        self,
        open_source: Callable[[ArtifactRef], AbstractContextManager[BinaryIO]],
        *,
        wait: bool = False,
        timeout: float | None = None,
        startup_timeout: float = 120,
    ) -> Mapping[int, WorkerProcess]:
        """Launch once; a failure leaves reachable owners for explicit cleanup."""
        if not self._busy.acquire(blocking=False):
            raise RuntimeError("node worker creation is already in progress")
        try:
            self._active()
            if self._started:
                raise RuntimeError("node workers can only be started once")
            self._started = True
            self._lease = self._admission.reserve(
                self._request,
                wait=wait,
                timeout=timeout,
                cancelled=self._cancelled,
                host=self._host_memory,
            )
            self._active()
            self._metadata = self._memory.reserve(
                4096 + 8 * MAX_METADATA_BYTES * len(self._launches), timeout=0
            )
            self._directory = TemporaryDirectory(prefix="workers-", dir=self._parent)
            root = Path(self._directory.name)
            first = next(iter(self._launches.values()))
            references = [first.startup.bundle, *first.weights]
            references.extend(
                launch.restore.rank.archive
                for launch in self._launches.values()
                if launch.restore is not None
            )
            self._delivery = FileDelivery(
                references, root=root, disk=self._disk, memory=self._memory
            )
            self._grants = FileGrants(root=root, memory=self._memory)
            self._outputs = OutputFiles(root=root, disk=self._disk, memory=self._memory)
            # Reserve every stream before any participant starts. A full node's
            # later log reservation cannot strand an already-running first rank.
            for rank, device in self.devices.items():
                self._logs[rank] = LogFiles(
                    self._lease,
                    device,
                    root=root,
                    disk=self._disk,
                    memory=self._memory,
                    limit_bytes=self._log_limit,
                )
                if self._capture_settings is not None:
                    assert self._capture_limits is not None
                    scope = self.prepared[rank].startup.scope
                    key = "\0".join((scope.run_id, scope.worker_incarnation, str(rank)))
                    self._captures[rank] = CaptureFiles(
                        self._lease,
                        self._capture_settings,
                        self._capture_limits,
                        artifact_id="nsys." + hashlib.sha256(key.encode()).hexdigest(),
                        root=root,
                        disk=self._disk,
                        memory=self._memory,
                        outputs=self._outputs,
                        device_id=device,
                    )
            self._active()
            self._delivery.receive(open_source)
            self._active()
            self._grant = self._grants.lend(self._delivery, references)
            for rank, launch in self._launches.items():
                self._active()
                self._processes[rank] = launch_worker(
                    self._lease,
                    self.devices[rank],
                    launch,
                    environment=self._environment,
                    files=self._grants.root,
                    outputs=self._outputs.root,
                    cache=self._cache,
                    logs=self._logs[rank],
                    cwd=root,
                    startup_timeout=startup_timeout,
                    capture=self._captures.get(rank),
                )
            self._active()
            self._files = NodeFiles(
                self._parent, self._disk, self._memory, self._grants, self._outputs
            )
            self.launched = True
            return MappingProxyType(dict(self._processes))
        finally:
            self._busy.release()

    def stop(self) -> None:
        """Worker-group stop callback: retain files until their owners finish."""
        self._cancelled.set()
        with self._lock:
            self._stopping = True
            if self._delivery is not None:
                self._delivery.cancel()
            if self._lease is not None:
                self._lease.close()
            if self._busy.locked():
                raise RuntimeError("node worker creation has not finished cleanup")

    def close(self) -> None:
        with self._lock:
            if self.closed:
                return
            self.stop()
            errors: list[BaseException] = []
            for owner in (
                self._grant,
                self._delivery,
                *self._captures.values(),
                self._outputs,
                self._grants,
                *self._logs.values(),
            ):
                if owner is not None:
                    try:
                        owner.close()
                    except BaseException as error:
                        errors.append(error)
            if errors:
                raise BaseExceptionGroup(
                    "node worker files still require cleanup", errors
                )
            if self._directory is not None:
                self._directory.cleanup()
            if self._metadata is not None:
                self._metadata.close()
            self.closed = True

    def __enter__(self) -> Self:
        self._active()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
