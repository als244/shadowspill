"""Public service lifetime around separate creation and live-request components."""

from __future__ import annotations

import os
import ssl
from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from threading import Event, RLock, Thread
from typing import Self
from uuid import uuid4

from shadowspill.orchestration.launch.admission import NodeAdmission
from shadowspill.orchestration.launch.host_arena import HostArena, HostPartition
from shadowspill.orchestration.nodes.authentication import NodeTokens
from shadowspill.orchestration.nodes.protocol import HTTP_BYTES as NODE_HTTP_BYTES
from shadowspill.orchestration.nodes.registry import NodeRegistry
from shadowspill.server.startup.configuration import LocalCreationFactory
from shadowspill.training.components import load_registry
from shadowspill_client._records import names
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.preview import TrainingDescription, TrainingPreview

from .allocation import Allocation
from .authentication import BearerTokens
from .deployment import ServerLimits
from .http import HTTP_SCRATCH_BYTES, HTTPService
from .jobs import RunJobs
from .storage import ServiceStorage


class Server:
    """A training service that clients connect to, serving one allocation.

    The installed component packages supply every model, objective and
    optimizer it runs. Construction opens nothing: `start` listens in a
    background thread and `serve` listens until interrupted. `close` ends
    the service; a client disconnecting or a run closing does not. Records
    and artifacts persist: a service started again on the same
    ``service_root`` serves them, and marks the runs it finds unfinished as
    stopped by the restart. Used as a context manager, it closes on exit.

    Parameters
    ----------
    allocation : Allocation, optional
        The compute to serve. By default `Allocation.from_environment`.
    service_root : str or path
        Directory of the service's mutable state: sessions, request outcomes,
        Plans, reports, outputs, artifacts and checkpoints. Keep it across
        restarts. One service at a time may use it.
    workload_packages : sequence of str, optional
        Distribution names of the installed component packages to enable.
        By default ``shadowspill-workloads`` when it is installed, otherwise
        none.
    model_root : str or path, optional
        An existing directory of pretrained checkpoints, only ever read. A
        pretrained model named without a revision, such as
        ``"organization/model"``, is looked for at
        ``model_root/organization/model`` and ``model_root/model``. None by
        default.
    cache_root : str or path, optional
        Directory for preparation and worker scratch, best on local SSD. By
        default ``.cache`` under ``service_root``.
    authentication : BearerTokens, optional
        Client credentials, each mapped to the owner ID it authenticates as.
        By default the ``SHADOWSPILL_TOKEN`` environment variable, for owner
        ``"owner"``.
    node_authentication : NodeTokens, optional
        Credentials with which node daemons register with the service, each
        distinct from every client credential. None by default: no node
        daemons.
    limits : ServerLimits, optional
        The service's envelopes. By default ``ServerLimits()``.

    Attributes
    ----------
    root : pathlib.Path
        ``service_root``.
    cache : pathlib.Path
        The scratch directory.
    packages : tuple of str
        The enabled component packages.
    incarnation : str
        ID of this service instance, new for each `Server`; clients see it as
        ``ServiceInfo.server_incarnation``.
    closed : bool
        Whether `close` has finished.

    Raises
    ------
    TypeError
        If an argument has the wrong type.
    ValueError
        If ``workload_packages`` repeats or has an empty name; no
        ``authentication`` is given and ``SHADOWSPILL_TOKEN`` does not hold
        32 to 512 visible ASCII characters; or a node credential is also a
        client credential.
    """

    def __init__(
        self,
        *,
        allocation: Allocation | None = None,
        service_root: str | Path,
        workload_packages: Sequence[str] | None = None,
        model_root: str | Path | None = None,
        cache_root: str | Path | None = None,
        authentication: BearerTokens | None = None,
        node_authentication: NodeTokens | None = None,
        limits: ServerLimits | None = None,
    ) -> None:
        if allocation is None:
            allocation = Allocation.from_environment()
        if not isinstance(allocation, Allocation):
            raise TypeError("server requires an existing Allocation")
        self.allocation = allocation
        self.root = Path(service_root)
        self.model_root = None if model_root is None else Path(model_root).absolute()
        self.cache = self.root / ".cache" if cache_root is None else Path(cache_root)
        if workload_packages is None:
            # The supplied catalog, models and optimizer, when installed.
            workload_packages = []
            try:
                distribution("shadowspill-workloads")
            except PackageNotFoundError:
                pass
            else:
                workload_packages.append("shadowspill-workloads")
        self.packages = names(workload_packages, "component packages")
        self.limits = ServerLimits() if limits is None else limits
        if not isinstance(self.limits, ServerLimits):
            raise TypeError("limits requires ServerLimits")
        self.authentication = authentication or BearerTokens(
            {os.environ.get("SHADOWSPILL_TOKEN", ""): "owner"}
        )
        if not isinstance(self.authentication, BearerTokens):
            raise TypeError("authentication requires BearerTokens")
        if node_authentication is not None:
            if not isinstance(node_authentication, NodeTokens):
                raise TypeError("node_authentication requires NodeTokens")
            node_authentication.require_separate(self.authentication)
        self.node_authentication = node_authentication
        self.incarnation = uuid4().hex
        self._nodes: NodeRegistry | None = None
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
        """The listener's host and port, as a pair.

        The port is the bound one, which a start with port 0 picks. Reading
        it raises `RuntimeError` when the listener is not running.
        """
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
        """Open the service and listen in a background thread.

        Opening checks the installed components, discovers the allocation's
        node, reserves the host memory the limits fix and recovers
        ``service_root``. If any of it fails, the service is closed.

        Parameters
        ----------
        host : str, default "127.0.0.1"
            Address to listen on.
        port : int, default 8000
            Port to listen on; 0 picks a free one.
        tls : ssl.SSLContext, optional
            Server TLS context, for a directly reachable HTTPS listener. None
            by default: plain HTTP, which only a loopback ``host`` accepts,
            for local or SSH-forwarded clients.

        Returns
        -------
        Server
            This server.

        Raises
        ------
        RuntimeError
            If the server has started already or is closing, or another
            service is using ``service_root``.
        ValueError
            If ``model_root`` is not an existing directory, a component
            package is not installed, or ``host`` is not a loopback address
            and ``tls`` is None.
        NotImplementedError
            If the allocation spans several nodes.
        OSError
            If the address cannot be bound.
        """
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
        if self.model_root is not None and not self.model_root.is_dir():
            raise ValueError("model_root must be an existing checkpoint directory")
        self.cache.mkdir(parents=True, exist_ok=True)
        load_registry(self.packages)  # Reject unavailable installed components first.
        self._node = self.allocation._local_node(
            self.cache, controller_host_bytes=limits.controller_host_bytes
        )
        connection_slots = limits.max_http_requests + 3 * limits.max_nodes
        connection_bytes = (
            0
            if self.node_authentication is None
            else NODE_HTTP_BYTES * connection_slots
        )
        self._host = HostArena(
            self._node, limits.fixed_host_bytes + connection_bytes, 5
        )
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
        connections = None
        if self.node_authentication is not None:
            self._nodes = NodeRegistry(self.incarnation, max_nodes=limits.max_nodes)
            connections = partition(connection_bytes, connection_slots)
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
            base_host_bytes=limits.fixed_host_bytes + connection_bytes,
            model_root=self.model_root,
            pool=self.allocation.pool,
            nodes=self._nodes,
        )
        self._jobs = RunJobs(
            self._storage,
            factory,
            memory=jobs_memory,
            host_bytes=limits.job_host_bytes,
            max_runs=limits.max_training_runs,
        )
        self._jobs.start()

        def describe(settings: TrainingDescription) -> TrainingPreview:
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
            hardware=factory.inventory,
            memory=http_memory,
            nodes=self._nodes,
            node_authentication=self.node_authentication,
            connections=connections,
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
        """Run the service in this thread until it is interrupted, then close it.

        It returns, or raises the interruption, once the listener stops and
        `close` has run; `close` from another thread also ends it.

        Parameters
        ----------
        host : str, default "127.0.0.1"
            Address to listen on.
        port : int, default 8000
            Port to listen on; 0 picks a free one.
        tls : ssl.SSLContext, optional
            Server TLS context, as for `start`.

        Raises
        ------
        RuntimeError, ValueError, NotImplementedError, OSError
            If `start` raises them.
        """
        self.start(host, port, tls=tls)
        try:
            # On CPython, interrupting Thread.join can mark a still-running
            # thread stopped. Keep the shutdown owner intact across SIGINT.
            self._listener_stopped.wait()
        finally:
            self.close()

    def close(self, *, timeout: float = 30) -> None:
        """Stop the service and release everything it holds.

        Stops the runs it owns and their workers, the listener and storage,
        and returns the allocation's memory. Records and artifacts persist. A
        close that fails keeps what it could not release and can be called
        again.

        Parameters
        ----------
        timeout : float, default 30
            Longest wait for the owned runs to stop, in seconds.

        Raises
        ------
        RuntimeError
            If runs or temporary files still have owners after the wait.
        """
        with self._lock:
            self._closing = True
            if self._nodes is not None and self._jobs is not None:
                # Node daemons still need the listener to acknowledge worker cleanup.
                # A failed close retains both owners for a subsequent retry.
                self._jobs.close(timeout=timeout)
                self._jobs = None
            if self._nodes is not None:
                self._nodes.close()
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
