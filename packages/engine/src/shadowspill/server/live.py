"""Service lifetime for the existing worker, controller and result owners."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from contextlib import AbstractContextManager
from pathlib import Path
from threading import Lock, RLock

from shadowspill.orchestration.capture.nsys import NsysCapture
from shadowspill.server.startup.checkpoint import CheckpointResolver
from shadowspill.server.startup.logs import publish_logs, publish_metadata
from shadowspill.training.preparation.planning.records import PreparedPlans
from shadowspill.training.session.checkpoint import ResolvedCheckpoint
from shadowspill.training.session.controller import RunController
from shadowspill.training.session.prepared import PreparedRequests
from shadowspill.training.session.state import OperationState
from shadowspill.training.worker.group import WorkerGroup
from shadowspill.training.worker.launch import WorkerProcess, start_processes
from shadowspill.training.worker.node import WorkerNode
from shadowspill.training.worker.requests import WorkerRequests, WorkerResults
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.captures import NsightManifest, NsightWindow
from shadowspill_client.protocol.codec import encode
from shadowspill_client.protocol.requests import ArtifactRef, identifier
from shadowspill_client.protocol.service import ServiceError

from .artifacts import ArtifactStore
from .checkpoints import RunCheckpoints
from .nsight import finalize_manifest, publish_manifest, read_manifest
from .records.requests import RequestRecords, RunIdentity
from .results import ResultPublisher
from .storage import ServiceStorage


class LiveRun:
    """Initialize once, retain storage through execution, retire in dependency order.

    The creation owner reserves the decoded family and every supplied node-backed
    budget before constructing this object, and keeps them until ``closed``.
    Construction performs no I/O. Keep this object after failed start or cleanup;
    partial resources remain reachable for retry. HTTP borrows belong to the
    surrounding service registry, which must join them before calling ``close``.

    This is lifecycle composition, not another training dispatcher. Readiness is
    published only after the existing all-worker initialization barrier. Closing
    the local client connection never invokes this owner's allocation ``stop``.
    """

    def __init__(
        self,
        storage: ServiceStorage,
        identity: RunIdentity,
        family: PreparedPlans,
        nodes: Sequence[WorkerNode],
        *,
        plans: ArtifactRef,
        history_id: str,
        root: Path,
        metadata: BufferBudget,
        uploads: BufferBudget,
        publication: BufferBudget,
        disk: BufferBudget,
        checkpoint: ResolvedCheckpoint | None = None,
        parent_checkpoint: ArtifactRef | None = None,
    ) -> None:
        self.history_id = identifier(history_id, "initial history ID")
        if (
            checkpoint is not None
            and self.history_id == checkpoint.manifest.info.history_id
        ):
            raise ValueError("resumed run requires a new history identity")
        self._checkpoint = checkpoint
        if parent_checkpoint is not None and (
            checkpoint is None
            or (
                parent_checkpoint != checkpoint.reference
                and (
                    checkpoint.manifest.info.conversion is None
                    or checkpoint.manifest.info.conversion.source != parent_checkpoint
                )
            )
        ):
            raise ValueError("run parent differs from checkpoint conversion provenance")
        self._parent_checkpoint = parent_checkpoint or (
            None if checkpoint is None else checkpoint.reference
        )
        ranks = [rank for node in nodes for rank in node.prepared]
        if (
            not nodes
            or len(set(ranks)) != len(ranks)
            or set(ranks) != family.plans.keys()
        ):
            raise ValueError("live run requires every prepared rank exactly once")
        for node in nodes:
            for worker in node.prepared.values():
                scope = worker.startup.scope
                if (
                    scope.run_id != identity.run_id
                    or scope.server_incarnation != storage.incarnation
                    or worker.startup.bundle != plans
                    or worker.bindings.prepared is not family
                ):
                    raise ValueError("worker startup differs from this prepared run")
                if (worker.restore is None) != (checkpoint is None) or (
                    worker.restore is not None
                    and checkpoint is not None
                    and (
                        worker.restore.checkpoint != checkpoint.reference
                        or worker.restore.info != checkpoint.manifest.info
                        or worker.restore.rank != checkpoint.manifest.ranks[scope.rank]
                    )
                ):
                    raise ValueError(
                        "worker initialization differs from run checkpoint"
                    )
        self._frozen_base = next(iter(nodes[0].prepared.values())).frozen_base
        if any(
            worker.frozen_base != self._frozen_base
            for node in nodes
            for worker in node.prepared.values()
        ):
            raise ValueError("workers disagree on initialized frozen base")
        execution = family.inventory.layouts.setup.config.execution
        self._nsys = NsysCapture.from_settings(execution.nsys)
        if (
            uploads.max_bytes < execution.max_staging_bytes
            or uploads.max_requests < execution.max_pending_requests
        ):
            raise ValueError("run input arena cannot cover configured staging")
        self.storage, self.identity, self.plans = storage, identity, plans
        self._family: PreparedPlans | None = family
        self._nodes = tuple(nodes)
        self._root, self._metadata, self._uploads = root, metadata, uploads
        self._publication, self._disk = publication, disk
        self._creating, self._retiring, self._mutex = Lock(), RLock(), RLock()
        self._started = self._closing = self._stopped = self.ready = self.closed = False
        self._record_pin: AbstractContextManager[RequestRecords] | None = None
        self._artifact_pin: AbstractContextManager[ArtifactStore] | None = None
        self._records: RequestRecords | None = None
        self._artifacts: ArtifactStore | None = None
        self._group: WorkerGroup | None = None
        self._workers: WorkerRequests | None = None
        self._publisher: ResultPublisher | None = None
        self._controller: RunController[WorkerResults] | None = None

    @property
    def controller(self) -> RunController[WorkerResults]:
        if not self.ready or self.closed or self._controller is None:
            raise RuntimeError("run has no initialized live controller")
        return self._controller

    @property
    def finished(self) -> bool:
        """A service reaper may retire this owner without an HTTP request."""
        return self.closed or (
            self._controller is not None
            and (self._controller.closed or self._controller.failure is not None)
        )

    def _active(self) -> None:
        with self._mutex:
            if self._closing or self.closed:
                raise RuntimeError("run ownership is closing or closed")

    def start(
        self, *, wait_for_resources: bool = False, startup_seconds: float = 120
    ) -> None:
        if not self._creating.acquire(blocking=False):
            raise RuntimeError("run initialization is already in progress")
        try:
            self._active()
            if self._started:
                raise RuntimeError("live run can only initialize once")
            self._started = True
            info = self.storage.catalog.get(self.identity)
            if (
                info is None
                or info.state != "preparing"
                or info.server_incarnation != self.storage.incarnation
            ):
                raise ValueError("live startup requires this server's preparing run")
            assert self._family is not None
            config = encode(self._family.inventory.layouts.setup.config)
            if hashlib.sha256(config).hexdigest() != info.configuration.sha256:
                raise ValueError("prepared configuration differs from registration")
            self._artifact_pin = self.storage.artifacts(self.identity.owner_id)
            self._artifacts = self._artifact_pin.__enter__()
            if self._artifacts.get(self.plans.id) != self.plans:
                raise ValueError("startup bundle is not published in owner storage")
            self._record_pin = self.storage.requests(self.identity)
            self._records = self._record_pin.__enter__()
            if self._records.last_sequence:
                raise ValueError(
                    "live startup cannot replay an existing request journal"
                )
            if info.stage != "initializing":
                self.storage.catalog.progress(self.identity, "initializing")
            if self._nsys is not None:
                for node in self._nodes:
                    for rank in node.prepared:
                        with self._publication.reserve(64 << 10, timeout=0):
                            publish_manifest(
                                self._artifacts, self._capture_record(rank)
                            )
            processes: dict[int, WorkerProcess] = {}
            for node in self._nodes:
                self._active()
                processes.update(
                    node.start(
                        self._artifacts.open,
                        local_source=self._artifacts.local_path,
                        wait=wait_for_resources,
                        startup_timeout=startup_seconds,
                    )
                )
            self._group = start_processes(processes, stop=self._stop_nodes)
            self._active()
            prepared = {
                rank: worker for n in self._nodes for rank, worker in n.prepared.items()
            }
            self._publisher = ResultPublisher(
                self._artifacts,
                session_id=self.identity.session_id,
                root=self._root,
                memory=self._publication,
                disk=self._disk,
            )
            self._workers = WorkerRequests(
                self._group,
                prepared,
                {rank: n.files for n in self._nodes for rank in n.prepared},
                metadata=self._metadata,
                checkpoints=RunCheckpoints(
                    self._artifacts,
                    session_id=self.identity.session_id,
                    configuration=info.configuration,
                    plans=self.plans,
                    world_size=len(prepared),
                    frozen_base=self._frozen_base,
                ),
            )
            self._controller = RunController(
                self._records,
                PreparedRequests(
                    self._family,
                    cache_root=self._root,
                    checkpoint=CheckpointResolver(
                        self._artifacts,
                        session_id=self.identity.session_id,
                        world_size=len(prepared),
                        configuration=self._family.inventory.layouts.setup.config,
                    ),
                ),
                self._workers,
                self._publisher,
                server_incarnation=self.storage.incarnation,
                budget=self._uploads,
                state=OperationState(
                    history_id=self.history_id,
                    completed_updates=0
                    if self._checkpoint is None
                    else self._checkpoint.manifest.info.completed_updates,
                ),
            )
            with self._mutex:
                self._active()
                self.storage.catalog.ready(
                    self.identity,
                    plans=self.plans,
                    initial_history_id=self.history_id,
                    parent_checkpoint=self._parent_checkpoint,
                    initial_checkpoint=None
                    if self._checkpoint is None
                    else self._checkpoint.reference,
                    initial_completed_updates=self._controller.boundary.completed_updates,
                )
                self.ready = True
        finally:
            self._creating.release()

    def _stop_nodes(self) -> None:
        errors = []
        for node in self._nodes:
            try:
                node.stop()
            except BaseException as error:
                errors.append(error)
        if errors:
            raise BaseExceptionGroup("run workers have not confirmed exit", errors)

    def interrupt(self) -> None:
        """Quiesce owned workers, retaining files and records for existing borrows."""
        with self._retiring:
            with self._mutex:
                self._closing = True
                if not self.closed:
                    self._stopped = self._stopped or not (
                        self._controller is not None
                        and self._controller.boundary.closed
                    )
            # Do not wait behind creation while it may wait for these children.
            if self._creating.locked():
                self._stop_nodes()
                raise RuntimeError("join run initialization before retiring its files")
            if self._controller is not None:
                self._controller.stop(self._stop_nodes)
            else:
                self._stop_nodes()

    def stop(self) -> None:
        """End allocation ownership, including its own interrupted startup wait."""
        self.interrupt()
        self.close()

    def _publish_logs(self) -> None:
        if self._artifacts is None:
            return
        phases = []
        for node in self._nodes:
            for rank, logs in node.logs.items():
                phase = f"worker-{rank}"
                phases.append(phase)
                if logs.closed:
                    continue  # A previous retirement already published this worker.
                with self._publication.reserve(64 << 10, timeout=0):
                    publish_logs(self._artifacts, self.identity, phase, logs)
        publish_metadata(
            self._artifacts,
            self.identity,
            "worker-logs",
            {
                "workers": phases,
            },
        )

    def _capture_record(self, rank: int) -> NsightManifest:
        assert self._nsys is not None
        return NsightManifest(
            self.identity.session_id,
            self.identity.run_id,
            rank,
            self.plans,
            NsightWindow(self._nsys.start_step, self._nsys.steps),
        )

    def _publish_captures(self) -> None:
        if self._nsys is None or self._artifacts is None:
            return
        for node in self._nodes:
            if node.closed:
                continue
            for rank in node.prepared:
                with self._publication.reserve(64 << 10, timeout=0):
                    current = read_manifest(self._artifacts, self.identity, rank)
                    if current is not None and current.status != "pending":
                        continue  # A retirement retry must preserve its final record.
                    finalize_manifest(
                        self._artifacts,
                        current or self._capture_record(rank),
                        node.captures.get(rank),
                    )

    def close(self) -> None:
        """Retire an orderly closed/idle run, or retry a prior allocation stop."""
        with self._retiring:
            if self.closed:
                return
            if not self._creating.acquire(blocking=False):
                raise RuntimeError("join run initialization before retiring its files")
            try:
                if self._controller is not None:
                    if self._stopped:
                        self._controller.stop(self._stop_nodes)
                    else:
                        self._controller.close()
                self._closing = True
                self._stop_nodes()
                if self._workers is not None:
                    self._workers.close()
                elif self._group is not None:
                    self._group.close()
                if self._publisher is not None:
                    self._publisher.close()
                self._publish_logs()
                self._publish_captures()
                errors = []
                for node in self._nodes:
                    try:
                        node.close()
                    except BaseException as error:
                        errors.append(error)
                if errors:
                    raise BaseExceptionGroup("run files still require cleanup", errors)
                if self.ready:
                    if self._controller is not None and self._controller.expired:
                        self.storage.catalog.stop(
                            self.identity,
                            ServiceError(
                                "idle_timeout",
                                "Configured idle timeout released workers.",
                            ),
                        )
                    elif self._stopped:
                        self.storage.catalog.stop(
                            self.identity,
                            ServiceError(
                                "allocation_stopped", "Allocation ownership ended."
                            ),
                        )
                    else:
                        self.storage.catalog.finish(self.identity)
                # Drop decoded families/closed owners before the parent can return
                # their host envelope. Pins outlive every producer and publisher.
                self._controller = self._publisher = self._workers = self._group = None
                self._family, self._nodes = None, ()
                if self._records is not None:
                    assert self._record_pin is not None
                    self._record_pin.__exit__(None, None, None)
                    self._records, self._record_pin = None, None
                if self._artifacts is not None:
                    assert self._artifact_pin is not None
                    self._artifact_pin.__exit__(None, None, None)
                    self._artifacts, self._artifact_pin = None, None
                self.closed = True
            finally:
                self._creating.release()
