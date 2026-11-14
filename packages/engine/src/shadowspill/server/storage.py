"""Bounded service storage and explicit startup classification, without workers."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Self

from shadowspill.training.session.recovery import fail_interrupted_requests
from shadowspill.training.session.state import OperationState
from shadowspill_client._records import integer
from shadowspill_client.buffers import BufferBudget, BufferLease
from shadowspill_client.protocol.requests import identifier
from shadowspill_client.protocol.service import ServiceError

from .artifacts import ArtifactStore, _directory
from .owners import OwnerCache
from .records.artifact_index import storage_key
from .records.requests import RequestRecords, RunIdentity
from .records.runs import RunCatalog


@dataclass
class _Requests:
    records: RequestRecords
    memory: BufferLease

    def close(self) -> None:
        self.records.close()
        self.memory.close()


class ServiceStorage:
    """One configured filesystem namespace, never a client-selected server path.

    The catalog lock excludes another server until every cached record/artifact
    owner has closed. Request owners reserve a deployment-specified host envelope
    before journal recovery; it covers their index and journal working memory.
    Returned metadata is additionally charged to its caller's request budget.
    Artifact owners retain their existing fixed index and transfer reservations.

    Call ``recover`` before serving requests. It classifies old unfinished runs
    through the existing request recovery engine, then records their stop. It
    never launches workers, restores mutable state or replays creation/execution.
    Keep this object when recovery/close fails; both operations support retry.
    """

    def __init__(
        self,
        root: Path,
        *,
        cache_root: Path,
        incarnation: str,
        memory: BufferBudget,
        request_host_bytes: int,
        max_request_owners: int,
        max_artifact_owners: int,
        artifact_bytes: int,
        artifact_count: int,
        artifact_transfers: int = 2,
    ) -> None:
        self.incarnation = identifier(incarnation, "server incarnation")
        self.request_host_bytes = integer(
            request_host_bytes, "request owner host bytes", minimum=1
        )
        integer(max_request_owners, "request owners", minimum=1)
        integer(max_artifact_owners, "artifact owners", minimum=1)
        integer(artifact_bytes, "artifact storage bytes", minimum=1)
        integer(artifact_count, "artifact count", minimum=1)
        integer(artifact_transfers, "artifact transfers", minimum=1)
        if request_host_bytes > memory.max_bytes:
            raise ValueError("request owner cannot fit configured host memory")
        self.root, self.cache_root, self._memory = root, cache_root, memory
        # Only the deployment chooses these roots. Below them all names are
        # fixed or hashed validated IDs; no request is interpreted as a path.
        for path in (root, cache_root):
            _directory(path)
        self._mutex = RLock()
        self._closing = self.closed = self._recovering = self.recovered = False
        self.catalog = RunCatalog(
            root / "catalog",
            incarnation=incarnation,
            cache_root=cache_root,
            memory=memory,
        )
        try:
            _directory(root / "requests")
            _directory(root / "artifacts")
            self._requests = OwnerCache(
                self._open_requests, max_owners=max_request_owners
            )
            self._artifacts: OwnerCache[str, ArtifactStore] = OwnerCache(
                lambda owner: ArtifactStore(
                    root / "artifacts" / storage_key(owner),
                    owner_id=owner,
                    cache_root=cache_root,
                    memory=memory,
                    max_bytes=artifact_bytes,
                    max_artifacts=artifact_count,
                    max_transfers=artifact_transfers,
                ),
                max_owners=max_artifact_owners,
            )
        except BaseException:
            self.catalog.close()
            raise

    def _usable(self, *, recovered: bool = True) -> None:
        if self._closing:
            raise RuntimeError("service storage is closing or closed")
        if recovered and not self.recovered:
            raise RuntimeError("classify interrupted runs before using request storage")

    def _request_path(self, identity: RunIdentity) -> Path:
        # IDs exclude NUL, so all three namespaces remain distinct, including
        # equal session/run IDs under different authenticated owners.
        return (
            self.root
            / "requests"
            / storage_key(
                "\0".join((identity.owner_id, identity.session_id, identity.run_id))
            )
        )

    def _open_requests(self, identity: RunIdentity) -> _Requests:
        info = self.catalog.get(identity)
        if info is None:
            raise KeyError("unknown owner/session/run")
        create = (
            info.state == "preparing" and info.server_incarnation == self.incarnation
        )
        directory = self._request_path(identity)
        if directory.is_symlink():
            raise ValueError("request storage requires private real directories")
        lease = self._memory.reserve(self.request_host_bytes, timeout=0)
        try:
            records = RequestRecords(
                directory,
                identity=identity,
                cache_root=self.cache_root,
                create=create,
            )
        except BaseException:
            lease.close()
            raise
        return _Requests(records, lease)

    @contextmanager
    def requests(self, identity: RunIdentity) -> Iterator[RequestRecords]:
        """Pin a live run for its lifetime, or a read-only lookup through its I/O.

        Only a current preparing run may lazily create its request journal. Older
        or initialized runs must already have their committed registration.
        """
        with self._mutex:
            self._usable()
        with self._requests.use(identity) as owner:
            yield owner.records

    @contextmanager
    def artifacts(self, owner: str) -> Iterator[ArtifactStore]:
        identifier(owner, "authenticated owner ID")
        with self._mutex:
            self._usable()
        with self._artifacts.use(owner) as artifacts:
            yield artifacts

    def recover(self) -> None:
        with self._mutex:
            self._usable(recovered=False)
            if self.recovered:
                return
            if self._recovering:
                raise RuntimeError("service storage recovery is already running")
            self._recovering = True
        try:
            after = 0
            while page := self.catalog.interrupted(after=after, limit=1):
                position, info = page[0]
                identity = RunIdentity(info.owner_id, info.session_id, info.run_id)
                path = self._request_path(identity)
                # A creation may have died before requesting a journal. A ready
                # run cannot lose its history and be silently treated as empty.
                if info.state == "ready" or path.exists() or path.is_symlink():
                    with self._requests.use(identity) as owner:
                        fail_interrupted_requests(owner.records, OperationState())
                self.catalog.stop(
                    identity,
                    ServiceError(
                        "server_restarted",
                        "Server restarted; recover training state from a checkpoint.",
                    ),
                )
                after = position
            with self._mutex:
                self._usable(recovered=False)
                self.recovered = True
        finally:
            with self._mutex:
                self._recovering = False

    def close(self) -> None:
        with self._mutex:
            self._closing = True
            if self._recovering:
                raise RuntimeError("join storage recovery before closing its owners")
        errors = []
        for cache in (self._requests, self._artifacts):
            try:
                cache.close()
            except BaseException as error:
                errors.append(error)
        if errors:
            raise BaseExceptionGroup("service storage owners remain", errors)
        with self._mutex:
            self.catalog.close()
            self.closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
