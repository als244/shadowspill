"""Public service lifetime around separate creation and live-request components."""

from __future__ import annotations

import os
import ssl
from collections.abc import Sequence
from pathlib import Path
from threading import Event, RLock, Thread
from typing import Self
from uuid import uuid4

from shadowspill.orchestration.launch.admission import NodeAdmission
from shadowspill.orchestration.launch.host_arena import HostArena, HostPartition
from shadowspill.training.definitions import load_registry
from shadowspill_client._records import names
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.preview import TrainingPreview
from shadowspill_client.protocol.setup import TrainingSetup
from shadowspill_client.protocol.training import ResolvedTraining

from .allocation import Allocation
from .authentication import BearerTokens
from .deployment import ServerLimits
from .http import HTTP_SCRATCH_BYTES, HTTPService
from .jobs import RunJobs
from .local_creation import LocalCreationFactory
from .storage import ServiceStorage


class Server:
    """Serve an existing allocation; installed definitions own model behavior.

    Construction is resource-free. ``start`` opens a background listener; ``serve``
    blocks until shutdown. ``close`` ends the service/allocation lifetime, distinct
    from a client disconnect or ordered per-run close. Failed cleanup retains this
    owner and can be retried. Persistent records/artifacts are never deleted here.
    """

    def __init__(
        self,
        *,
        allocation: Allocation,
        storage_root: str | Path,
        workload_packages: Sequence[str] = (),
        cache_root: str | Path | None = None,
        authentication: BearerTokens | None = None,
        limits: ServerLimits | None = None,
    ) -> None:
        if not isinstance(allocation, Allocation):
            raise TypeError("server requires an existing Allocation")
        self.allocation = allocation
        self.root = Path(storage_root)
        self.cache = self.root / ".cache" if cache_root is None else Path(cache_root)
        self.packages = names(workload_packages, "definition packages")
        self.limits = ServerLimits() if limits is None else limits
        if not isinstance(self.limits, ServerLimits):
            raise TypeError("limits requires ServerLimits")
        self.authentication = authentication or BearerTokens(
            {os.environ.get("SHADOWSPILL_TOKEN", ""): "owner"}
        )
        if not isinstance(self.authentication, BearerTokens):
            raise TypeError("authentication requires BearerTokens")
        self.incarnation = uuid4().hex
        self._lock = RLock()
        self._node: NodeAdmission | None = None
        self._host: HostArena | None = None
        self._partitions: list[HostPartition] = []
        self._disk: BufferBudget | None = None
        self._storage: ServiceStorage | None = None
        self._jobs: RunJobs | None = None
        self._http: HTTPService | None = None
        self._thread: Thread | None = None
        self._listener_stopped = Event()
        self._closing = self.closed = False

    @property
    def address(self) -> tuple[str, int]:
        if self._http is None or self._closing:
            raise RuntimeError("server listener is not running")
        return str(self._http.server_address[0]), int(self._http.server_address[1])

    def start(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        *,
        tls: ssl.SSLContext | None = None,
    ) -> Self:
        with self._lock:
            if self._closing or self._http is not None:
                raise RuntimeError("server already started or is closing")
            try:
                self._open(host, port, tls)
            except BaseException:
                self.close()
                raise
            return self

    def _open(self, host: str, port: int, tls: ssl.SSLContext | None) -> None:
        limits = self.limits
        self.cache.mkdir(parents=True, exist_ok=True)
        load_registry(self.packages)  # Reject unavailable installed definitions first.
        self._node = self.allocation._local_node(
            self.cache, controller_host_bytes=limits.controller_host_bytes
        )
        self._host = HostArena(self._node, limits.fixed_host_bytes, 4)
        host_memory = self._host

        def partition(size: int, count: int) -> BufferBudget:
            owner = host_memory.partition(size, count)
            self._partitions.append(owner)
            return owner.budget

        storage_memory = partition(
            limits.storage_bytes,
            1 + limits.max_open_records + 5 * limits.max_artifact_owners,
        )
        jobs_memory = partition(
            limits.job_host_bytes * limits.max_training_runs, limits.max_training_runs
        )
        preview_memory = partition(limits.job_host_bytes, 1)
        http_memory = partition(
            HTTP_SCRATCH_BYTES * limits.max_http_requests, limits.max_http_requests
        )
        self._disk = BufferBudget(limits.temporary_disk_bytes, limits.temporary_files)
        self._storage = ServiceStorage(
            self.root,
            cache_root=self.cache,
            incarnation=self.incarnation,
            memory=storage_memory,
            request_host_bytes=limits.request_host_bytes,
            max_request_owners=limits.max_open_records,
            max_artifact_owners=limits.max_artifact_owners,
            artifact_bytes=limits.artifact_bytes,
            artifact_count=limits.artifact_count,
            upload_ttl_seconds=limits.upload_ttl_seconds,
        )
        self._storage.recover()
        environment = dict(self.allocation.environment)
        environment.pop("SHADOWSPILL_TOKEN", None)
        factory = LocalCreationFactory(
            self._storage,
            self._node,
            packages=self.packages,
            root=self.cache,
            cache=self.cache,
            environment=environment,
            limits=limits.creation,
            disk=self._disk,
            base_host_bytes=limits.fixed_host_bytes,
        )
        self._jobs = RunJobs(
            self._storage,
            factory,
            memory=jobs_memory,
            host_bytes=limits.job_host_bytes,
            max_runs=limits.max_training_runs,
        )
        self._jobs.start()

        def describe(settings: TrainingSetup | ResolvedTraining) -> TrainingPreview:
            with preview_memory.reserve(limits.job_host_bytes, timeout=0):
                return factory.describe(settings)

        self._http = HTTPService(
            (host, port),
            incarnation=self.incarnation,
            authentication=self.authentication,
            runs=self._jobs.access,
            artifacts=self._storage.artifacts,
            jobs=self._jobs,
            describe=describe,
            memory=http_memory,
            request_timeout=limits.request_timeout,
            tls=tls,
        )
        self._thread = Thread(
            target=self._listen,
            name="shadowspill-server",
        )
        self._thread.start()

    def _listen(self) -> None:
        assert self._http is not None
        try:
            self._http.serve_forever(poll_interval=0.05)
        finally:
            self._listener_stopped.set()

    def serve(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        *,
        tls: ssl.SSLContext | None = None,
    ) -> None:
        self.start(host, port, tls=tls)
        try:
            # On CPython, interrupting Thread.join can mark a still-running
            # thread stopped. Keep the shutdown owner intact across SIGINT.
            self._listener_stopped.wait()
        finally:
            self.close()

    def close(self, *, timeout: float = 30) -> None:
        with self._lock:
            self._closing = True
            if self._http is not None:
                if self._thread is not None and self._thread.is_alive():
                    self._http.shutdown()
                    self._thread.join()
                self._http.server_close()
                self._http = None
                self._thread = None
            if self._jobs is not None:
                self._jobs.close(timeout=timeout)
                self._jobs = None
            if self._storage is not None:
                self._storage.close()
                self._storage = None
            if self._disk is not None:
                self._disk.close()
                if self._disk.used != (0, 0):
                    raise RuntimeError("service temporary files still have owners")
            for partition in reversed(self._partitions):
                partition.close()
            if self._host is not None:
                self._host.close()
            if self._node is not None:
                self._node.close()
            self.closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
