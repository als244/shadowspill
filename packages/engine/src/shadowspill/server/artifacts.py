"""Owner-scoped immutable binary files, committed after durable payload writes."""

from __future__ import annotations

import os
import re
import shutil
import stat
from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import BinaryIO, Self

from shadowspill.orchestration.launch.publication import OutputSpec, OutputWriter
from shadowspill.orchestration.launch.staging import FileDelivery
from shadowspill_client._records import integer
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.requests import ArtifactRef, identifier

from .artifact_group import StagedArtifacts
from .artifact_writer import ProducedArtifacts
from .records.artifact_index import (
    ArtifactAlias,
    ArtifactIndex,
    alias_scope,
    reference_bytes,
    storage_key,
)
from .records.journal import Journal, _sync_directory
from .records.journal_format import JournalLimits

INDEX_BYTES = 4 << 20
MAX_GROUP_ARTIFACTS = 128


@dataclass
class _Publication:
    ids: tuple[str, ...]
    size_bytes: int
    owner: FileDelivery | ProducedArtifacts
    committed: bool = False
    uncertain: bool = False


def _directory(path: Path) -> None:
    path.mkdir(mode=0o700, exist_ok=True)
    if path.is_symlink() or not path.is_dir():
        raise ValueError("artifact storage requires private real directories")
    _sync_directory(path.parent)


class ArtifactStore:
    """A persistent owner namespace, not worker-local output or a tensor codec.

    The service supplies its authenticated owner and configured directory; no
    request chooses a filesystem path. Files are retained after close/restart.
    The existing FileDelivery verifies each bounded stream. Only a subsequent
    durable journal commit makes it discoverable. The index is disposable local
    SQLite; it supplies neither locking nor durability on shared storage.

    Limits cover committed plus pending payload bytes/artifact count. Fixed index
    memory and concurrent transfer scratch come from the supplied host arena.
    A durability error disables this writer until close and journal recovery.
    """

    def __init__(
        self,
        directory: Path,
        *,
        owner_id: str,
        cache_root: Path,
        memory: BufferBudget,
        max_bytes: int,
        max_artifacts: int,
        max_transfers: int = 2,
    ) -> None:
        self.owner_id = identifier(owner_id, "artifact owner ID")
        self.max_bytes = integer(max_bytes, "artifact storage bytes", minimum=1)
        self.max_artifacts = integer(max_artifacts, "artifact count", minimum=1)
        integer(max_transfers, "artifact transfers", minimum=1)
        self._memory = memory
        self._fixed = memory.reserve(INDEX_BYTES, timeout=0)
        self.directory = directory
        self._data, self._scratch = directory / "data", directory / "pending"
        self._mutex = RLock()
        self._disk = BufferBudget(max_bytes, max_transfers)
        self._pending: dict[str, _Publication] = {}
        self._reserved: set[str] = set()
        self._receiving: set[str] = set()
        self._bytes = self._count = self._readers = 0
        self._broken = self.closed = False
        try:
            _directory(directory)
            self._journal = Journal(
                directory / "events",
                limits=JournalLimits(
                    record_bytes=8192,
                    commit_bytes=(MAX_GROUP_ARTIFACTS + 2) * 8192,
                    commit_records=MAX_GROUP_ARTIFACTS + 1,
                ),
            )
            _directory(self._data)
            _directory(self._scratch)
            self._index = ArtifactIndex(cache_root)
            self._recover()
        except BaseException:
            if hasattr(self, "_index"):
                self._index.close()
            if hasattr(self, "_journal"):
                self._journal.close()
            self._fixed.close()
            raise

    def _path(self, reference: ArtifactRef) -> Path:
        return self._data / storage_key(reference.id)

    def _open(self, reference: ArtifactRef) -> BinaryIO:
        fd = os.open(self._path(reference), os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            facts = os.fstat(fd)
            if not stat.S_ISREG(facts.st_mode) or facts.st_size != reference.size_bytes:
                raise ValueError("committed artifact payload is missing or invalid")
            return os.fdopen(fd, "rb", buffering=0)
        except BaseException:
            os.close(fd)
            raise

    def _recover(self) -> None:
        rows = self._journal.records()
        first = next(rows, None)
        identity = {"owner_id": self.owner_id}
        if first is None:
            self._journal.append([("artifacts.open", identity)])
        elif first.kind != "artifacts.open" or dict(first.data) != identity:
            raise ValueError("artifact storage belongs to another owner")
        for row in rows:
            if row.kind == "artifact.alias":
                self._index.set_alias(ArtifactAlias.from_dict(dict(row.data)))
                continue
            if row.kind != "artifact.put":
                raise ValueError("unknown committed artifact record")
            reference = ArtifactRef.from_dict(dict(row.data))
            with self._open(reference):
                pass
            self._index.insert(reference)
            self._bytes += reference.size_bytes
            self._count += 1
        if self._bytes > self.max_bytes or self._count > self.max_artifacts:
            raise ValueError("retained artifacts exceed configured storage limits")
        # The exclusive journal writer owns these private directories. Published
        # references are never swept; uncertain commits were reconciled above.
        with os.scandir(self._data) as entries:
            for entry in entries:
                if not re.fullmatch(r"[0-9a-f]{64}", entry.name):
                    raise ValueError("unexpected file in private artifact storage")
                if not self._index.contains(entry.name):
                    Path(entry.path).unlink()
        _sync_directory(self._data)
        with os.scandir(self._scratch) as entries:
            for entry in entries:
                if not entry.name.startswith(
                    ("delivery-", "production-")
                ) or not entry.is_dir(follow_symlinks=False):
                    raise ValueError("unexpected file in private artifact staging")
                shutil.rmtree(entry.path)
        _sync_directory(self._scratch)

    def _usable(self) -> None:
        if self.closed:
            raise RuntimeError("artifact storage is closed")
        if self._broken:
            raise RuntimeError("artifact storage failed; close and reconcile it")

    @property
    def used(self) -> tuple[int, int]:
        with self._mutex:
            return self._bytes, self._count

    def get(self, artifact_id: str) -> ArtifactRef | None:
        identifier(artifact_id, "artifact ID")
        with self._mutex:
            self._usable()
            return self._index.get(artifact_id)

    def page(self, *, after: str = "", limit: int = 100) -> tuple[ArtifactRef, ...]:
        if after != "":
            identifier(after, "artifact cursor")
        if type(limit) is not int or not 1 <= limit <= 128:
            raise ValueError("artifact page limit must be in 1..128")
        with self._mutex:
            self._usable()
            return self._index.page(after, limit)

    def resolve_alias(
        self, session_id: str, run_id: str, name: str
    ) -> ArtifactRef | None:
        scope = alias_scope(session_id, run_id, name)
        with self._mutex:
            self._usable()
            return self._index.resolve_alias(*scope)

    def put(
        self,
        reference: ArtifactRef,
        open_source: Callable[[ArtifactRef], AbstractContextManager[BinaryIO]],
    ) -> ArtifactRef:
        """Publish a verified immutable generation; an exact retry copies nothing."""
        reference_bytes(reference)
        with self._mutex:
            self._usable()
            existing = self._index.get(reference.id)
            if existing is not None:
                if existing != reference:
                    raise ValueError("artifact ID already names different content")
                return existing
            ids = (reference.id,)
            self._capacity(ids, reference.size_bytes)
            delivery = FileDelivery(
                [reference], root=self._scratch, disk=self._disk, memory=self._memory
            )
            pending = self._track(ids, reference.size_bytes, delivery)
        try:
            delivery.receive(open_source)
            with delivery.use() as resolve:
                return self._commit(pending, (reference,), (resolve(reference),))[0]
        finally:
            self._finish(pending)

    def produce(
        self, spec: OutputSpec, write: Callable[[OutputWriter], ArtifactRef]
    ) -> ArtifactRef:
        """Publish one new ID directly from an installed bounded binary encoder.

        Request deduplication happens before invoking the publisher. Without a
        known expected digest, an already committed ID is a conflict, not an
        instruction to invoke the encoder again or replace the immutable file.
        """
        return self.produce_group((spec,), lambda _, writer: write(writer))[0]

    def produce_group(
        self,
        specs: Sequence[OutputSpec],
        write: Callable[[OutputSpec, OutputWriter], ArtifactRef],
        *,
        alias: ArtifactAlias | None = None,
    ) -> tuple[ArtifactRef, ...]:
        """Admit and publish all members and an optional name in one durable commit.

        Encoders run serially in the given order, directly in persistent storage.
        None of the group is discoverable until every producer has confirmed its
        complete output. A final manifest may reference earlier members. Failures
        before commit reclaim all partials; uncertain commits require recovery.
        """
        with self.stage_group(specs, write, alias=alias) as group:
            return group.commit()

    def stage_group(
        self,
        specs: Sequence[OutputSpec],
        write: Callable[[OutputSpec, OutputWriter], ArtifactRef],
        *,
        alias: ArtifactAlias | None = None,
    ) -> StagedArtifacts:
        """Finish verified files without publishing or releasing their ownership.

        The caller must close the returned owner even if its execution receipt
        fails. Storage cannot close while that owner awaits commit or discard.
        """
        if not 1 <= len(specs) <= MAX_GROUP_ARTIFACTS:
            raise ValueError(
                f"artifact group must contain 1..{MAX_GROUP_ARTIFACTS} files"
            )
        specs = tuple(specs)
        ids = tuple(spec.id for spec in specs)
        if len(set(ids)) != len(ids):
            raise ValueError("artifact group requires distinct IDs")
        if alias is not None and alias.artifact_id not in ids:
            raise ValueError("artifact alias must name a member of its new generation")
        size = sum(spec.size_bytes for spec in specs)
        with self._mutex:
            self._usable()
            if any(self._index.get(name) is not None for name in ids):
                raise ValueError("artifact ID is already committed")
            self._capacity(ids, size)
            producer = ProducedArtifacts(
                specs, root=self._scratch, disk=self._disk, memory=self._memory
            )
            pending = self._track(ids, size, producer)
        try:
            references = producer.build(write)
            return StagedArtifacts(self, pending, references, producer.paths, alias)
        except BaseException:
            self._finish(pending)
            raise

    def _capacity(self, ids: tuple[str, ...], size: int) -> None:
        if self._reserved.intersection(ids):
            raise RuntimeError("artifact publication is already in progress")
        if (
            self._bytes + size > self.max_bytes
            or self._count + len(ids) > self.max_artifacts
        ):
            raise ValueError("artifact storage capacity is exhausted")

    def _track(
        self, ids: tuple[str, ...], size: int, owner: FileDelivery | ProducedArtifacts
    ) -> _Publication:
        pending = _Publication(ids, size, owner)
        self._pending[ids[0]] = pending
        self._reserved.update(ids)
        self._receiving.add(ids[0])
        self._bytes += size
        self._count += len(ids)
        return pending

    def _commit(
        self,
        pending: _Publication,
        references: tuple[ArtifactRef, ...],
        paths: tuple[Path, ...],
        alias: ArtifactAlias | None = None,
    ) -> tuple[ArtifactRef, ...]:
        with self._mutex:
            self._usable()
            try:
                # Link only after hashing the complete stream. The same
                # filesystem keeps this bounded and avoids a second copy.
                for reference, path in zip(references, paths, strict=True):
                    with path.open("rb") as stream:
                        os.fsync(stream.fileno())
                    os.link(path, self._path(reference))
                _sync_directory(self._data)
                entries = [
                    ("artifact.put", reference.to_dict()) for reference in references
                ]
                if alias is not None:
                    entries.append(("artifact.alias", alias.to_dict()))
                self._journal.append(entries)
                for reference in references:
                    self._index.insert(reference)
                if alias is not None:
                    self._index.set_alias(alias)
                pending.committed = True
            except BaseException:
                # A visible record may be durable even when acknowledgement
                # failed. Keep its bytes for recovery, never roll it back.
                pending.uncertain = True
                self._broken = True
                raise
        return references

    def _finish(self, pending: _Publication) -> None:
        with self._mutex:
            if self._pending.get(pending.ids[0]) is not pending:
                return  # Storage shutdown may have retried a failed owner cleanup.
            self._receiving.discard(pending.ids[0])
            try:
                self._release(pending)
            except BaseException:
                # Retain the failed owner and its quotas for explicit retry.
                self._broken = True
                raise

    def _release(self, pending: _Publication) -> None:
        pending.owner.close()
        if not pending.committed and not pending.uncertain:
            self._bytes -= pending.size_bytes
            self._count -= len(pending.ids)
        self._reserved.difference_update(pending.ids)
        del self._pending[pending.ids[0]]

    @contextmanager
    def open(self, reference: ArtifactRef) -> Iterator[BinaryIO]:
        """Borrow a committed generation, independently of its producing worker."""
        with self._mutex:
            self._usable()
            if self._index.get(reference.id) != reference:
                raise KeyError("artifact reference is not committed for this owner")
            stream = self._open(reference)
            self._readers += 1
        try:
            yield stream
        finally:
            stream.close()
            with self._mutex:
                self._readers -= 1

    def close(self) -> None:
        with self._mutex:
            if self.closed:
                return
            if self._receiving or self._readers:
                raise RuntimeError("artifact producers or readers still own storage")
            for pending in tuple(self._pending.values()):
                self._release(pending)
            self._index.close()
            self._journal.close()
            self._disk.close()
            self._fixed.close()
            self.closed = True

    def __enter__(self) -> Self:
        self._usable()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
