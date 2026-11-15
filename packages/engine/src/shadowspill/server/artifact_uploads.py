"""Resumable private transfers owned by the existing artifact store and journal."""

from __future__ import annotations

import hashlib
import os
import re
import stat
from _hashlib import HASH
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from time import time
from typing import TYPE_CHECKING, BinaryIO

from shadowspill.orchestration.launch.staging import COPY_BYTES
from shadowspill_client._records import integer, number
from shadowspill_client.buffers import BufferLease
from shadowspill_client.protocol.codec import ProtocolError
from shadowspill_client.protocol.payload_io import BufferReader
from shadowspill_client.protocol.requests import ArtifactRef, identifier
from shadowspill_client.protocol.uploads import MAX_UPLOAD_CHUNK_BYTES, UploadStatus

from .records.artifact_index import reference_bytes, storage_key
from .records.journal import _sync_directory

if TYPE_CHECKING:
    from .artifacts import ArtifactStore

UPLOAD_TTL_SECONDS = 24 * 60 * 60
_HASH_BYTES = 4096


@dataclass
class _Prefix:
    offset: int
    stamp: tuple[int, int, int, int, int]
    digest: HASH
    memory: BufferLease


def _stamp(stream: BinaryIO) -> tuple[int, int, int, int, int]:
    facts = os.fstat(stream.fileno())
    return (
        facts.st_dev,
        facts.st_ino,
        facts.st_size,
        facts.st_mtime_ns,
        facts.st_ctime_ns,
    )


def _complete(reference: ArtifactRef) -> UploadStatus:
    return UploadStatus(reference, reference.size_bytes, reference.sha256, "complete")


class ArtifactUploads:
    """Private files and progress, with the store's locking, quotas and durability.

    No training state or tensor decoder lives here. Only acknowledged, checksum
    verified chunks advance progress. The optional SHA-state cache is bounded by
    transfer count and charged to host memory; eviction/restart rehashes a prefix
    with fixed scratch. Expiry applies only to incomplete transfers.
    """

    def __init__(
        self, store: ArtifactStore, *, max_transfers: int, ttl_seconds: float
    ) -> None:
        self._store = store
        self._directory = store.directory / "uploads"
        self._max_transfers = max_transfers
        self._ttl = number(ttl_seconds, "upload lifetime", minimum=0)
        if not self._ttl:
            raise ValueError("upload lifetime must be positive")
        self._prefixes: OrderedDict[str, _Prefix] = OrderedDict()

    def _path(self, name: str) -> Path:
        return self._directory / storage_key(name)

    def replay(self, kind: str, data: Mapping[str, object]) -> bool:
        store = self._store
        if kind in ("upload.begin", "upload.progress"):
            status = UploadStatus.from_dict(dict(data))
            if status.state != "uploading":
                raise ValueError("upload journal requires incomplete progress")
            previous = store._index.upload(status.reference.id)
            if kind == "upload.begin":
                if previous is not None or store._index.get(status.reference.id):
                    raise ValueError("upload ID already exists")
                if status.offset:
                    raise ValueError("new upload cannot have a received prefix")
                store._bytes += status.reference.size_bytes
                store._count += 1
            elif (
                previous is None
                or previous.reference != status.reference
                or previous.offset >= status.offset
            ):
                raise ValueError("upload journal has inconsistent progress")
            store._index.set_upload(status)
            return True
        if kind == "upload.remove":
            if set(data) != {"id"}:
                raise ValueError("invalid upload removal")
            name = identifier(data["id"], "upload ID")
            previous = store._index.upload(name)
            if previous is None:
                raise ValueError("upload removal has no retained owner")
            store._index.remove_upload(name)
            store._bytes -= previous.reference.size_bytes
            store._count -= 1
            return True
        return False

    def recover(self) -> None:
        """After committed data reconciliation, sweep only private upload files."""
        self.expire()
        with os.scandir(self._directory) as entries:
            for entry in entries:
                if not re.fullmatch(r"[0-9a-f]{64}", entry.name):
                    raise ValueError("unexpected file in private upload storage")
                if self._store._index.upload_key(entry.name) is None:
                    Path(entry.path).unlink()
        _sync_directory(self._directory)
        after = ""
        while page := self._store._index.upload_page(after, 128):
            for name in page:
                status = self._store._index.upload(name)
                assert status is not None
                with self._open(status):
                    pass
            after = page[-1]

    def _open(self, status: UploadStatus) -> BinaryIO:
        fd = os.open(
            self._path(status.reference.id), os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK
        )
        try:
            facts = os.fstat(fd)
            if (
                not stat.S_ISREG(facts.st_mode)
                or facts.st_nlink != 1
                or not status.offset <= facts.st_size <= status.reference.size_bytes
            ):
                raise ValueError("retained upload prefix is missing or invalid")
            if facts.st_size != status.offset:
                os.ftruncate(fd, status.offset)
                os.fsync(fd)
            return os.fdopen(fd, "r+b", buffering=0)
        except BaseException:
            os.close(fd)
            raise

    def _forget(self, name: str) -> None:
        previous = self._prefixes.pop(name, None)
        if previous is not None:
            previous.memory.close()

    def _remember(self, status: UploadStatus, stream: BinaryIO, digest: HASH) -> None:
        name = status.reference.id
        self._forget(name)
        while len(self._prefixes) >= self._max_transfers:
            self._forget(next(iter(self._prefixes)))
        try:
            memory = self._store._memory.reserve(_HASH_BYTES, timeout=0)
        except (TimeoutError, ValueError):
            return  # Optional cache; durable bytes/progress are authoritative.
        self._prefixes[name] = _Prefix(status.offset, _stamp(stream), digest, memory)

    def _digest(self, status: UploadStatus, stream: BinaryIO) -> HASH:
        with self._store._mutex:
            cached = self._prefixes.pop(status.reference.id, None)
        if cached is not None:
            try:
                if cached.offset == status.offset and cached.stamp == _stamp(stream):
                    stream.seek(status.offset)
                    return cached.digest.copy()
            finally:
                cached.memory.close()
        digest = hashlib.sha256()
        remaining = status.offset
        while remaining:
            block = stream.read(min(COPY_BYTES, remaining))
            try:
                if not block:
                    raise ValueError("retained upload prefix is truncated")
                digest.update(block)
                remaining -= len(block)
            finally:
                del block
        if digest.hexdigest() != status.prefix_sha256:
            raise ValueError("retained upload prefix checksum differs")
        return digest

    def expire(self) -> None:
        """Reclaim inactive expired uploads before access or new admission."""
        store = self._store
        with store._mutex:
            store._usable()
            after, now = "", time()
            while page := store._index.expired_uploads(now, after=after, limit=128):
                for name in page:
                    if name not in store._receiving:
                        self._remove(name)
                after = page[-1]

    def _remove(self, name: str) -> None:
        store = self._store
        status = store._index.upload(name)
        assert status is not None
        try:
            store._journal.append([("upload.remove", {"id": name})])
            self._path(name).unlink(missing_ok=True)
            _sync_directory(self._directory)
            store._index.remove_upload(name)
            store._bytes -= status.reference.size_bytes
            store._count -= 1
            self._forget(name)
        except BaseException:
            store._broken = True
            raise

    def begin(self, reference: ArtifactRef) -> UploadStatus:
        reference_bytes(reference)
        store = self._store
        with store._mutex:
            store._usable()
            self.expire()
            status = self.get(reference.id)
            if status is not None:
                if status.reference != reference:
                    raise ValueError("upload ID already names different content")
                return status
            store._capacity((reference.id,), reference.size_bytes)
            status = UploadStatus(
                reference,
                0,
                hashlib.sha256().hexdigest(),
                "uploading",
                time() + self._ttl,
            )
            store._bytes += reference.size_bytes
            store._count += 1
            try:
                with self._path(reference.id).open("xb", buffering=0) as stream:
                    os.fsync(stream.fileno())
                _sync_directory(self._directory)
                store._journal.append([("upload.begin", status.to_dict())])
                store._index.set_upload(status)
            except BaseException:
                store._broken = True
                raise
            return status

    def get(self, name: str) -> UploadStatus | None:
        identifier(name, "upload ID")
        store = self._store
        with store._mutex:
            store._usable()
            self.expire()
            reference = store._index.get(name)
            return _complete(reference) if reference else store._index.upload(name)

    def abort(self, name: str) -> None:
        identifier(name, "upload ID")
        store = self._store
        with store._mutex:
            store._usable()
            if name in store._receiving:
                raise RuntimeError("upload still has an active producer")
            if store._index.upload(name) is not None:
                self._remove(name)

    def chunk(
        self, name: str, offset: int, size: int, sha256: str, source: BufferReader
    ) -> UploadStatus:
        identifier(name, "upload ID")
        integer(offset, "upload offset")
        integer(size, "upload chunk bytes", minimum=1)
        if (
            size > MAX_UPLOAD_CHUNK_BYTES
            or re.fullmatch(r"[0-9a-f]{64}", sha256) is None
        ):
            raise ValueError("upload requires a bounded checksum-labelled chunk")
        store = self._store
        with store._mutex:
            status = self.get(name)
            if status is None:
                raise KeyError("unknown upload")
            if name in store._receiving:
                raise RuntimeError("upload already has an active producer")
            if status.state != "uploading" or status.offset != offset:
                raise ValueError("upload chunk must follow its committed prefix")
            if offset + size > status.reference.size_bytes:
                raise ValueError("upload chunk exceeds its declared artifact")
            transfer = store._disk.reserve(0, timeout=0)
            try:
                memory = store._memory.reserve(COPY_BYTES + _HASH_BYTES, timeout=0)
            except BaseException:
                transfer.close()
                raise
            store._receiving.add(name)
        try:
            with self._open(status) as stream:
                digest = self._digest(status, stream)
                self._copy(source, stream, size, sha256, digest)
                progress = replace(
                    status,
                    offset=offset + size,
                    prefix_sha256=digest.hexdigest(),
                    expires_at=time() + self._ttl,
                )
                os.fsync(stream.fileno())
                with store._mutex:
                    store._usable()
                    try:
                        store._journal.append([("upload.progress", progress.to_dict())])
                        store._index.set_upload(progress)
                    except BaseException:
                        store._broken = True
                        raise
                    self._remember(progress, stream, digest)
                return progress
        finally:
            with store._mutex:
                store._receiving.remove(name)
            memory.close()
            transfer.close()

    @staticmethod
    def _copy(
        source: BufferReader, stream: BinaryIO, size: int, expected: str, digest: HASH
    ) -> None:
        chunk_hash = hashlib.sha256()
        buffer = bytearray(COPY_BYTES)
        try:
            with memoryview(buffer) as view:
                remaining = size
                while remaining:
                    with view[: min(COPY_BYTES, remaining)] as part:
                        count = source.readinto(part)
                        if type(count) is not int or not 0 < count <= len(part):
                            raise ProtocolError("upload chunk is truncated")
                        with part[:count] as filled:
                            digest.update(filled)
                            chunk_hash.update(filled)
                            offset = 0
                            while offset < count:
                                with filled[offset:] as pending:
                                    written = stream.write(pending)
                                if (
                                    type(written) is not int
                                    or not 0 < written <= count - offset
                                ):
                                    raise OSError("upload write made no progress")
                                offset += written
                        remaining -= count
                with view[:1] as extra:
                    if source.readinto(extra) != 0:
                        raise ProtocolError("upload chunk exceeds its declared length")
            if chunk_hash.hexdigest() != expected:
                raise ProtocolError("upload chunk checksum differs")
        finally:
            del buffer

    def complete(
        self, name: str, *, validate: Callable[[Path], None] | None = None
    ) -> UploadStatus:
        store = self._store
        with store._mutex:
            status = self.get(name)
            if status is None:
                raise KeyError("unknown upload")
            if status.state == "complete":
                return status
            if name in store._receiving:
                raise RuntimeError("upload still has an active producer")
            if status.offset != status.reference.size_bytes:
                raise ValueError("upload has not received its complete artifact")
            with (
                store._memory.reserve(COPY_BYTES + _HASH_BYTES, timeout=0),
                self._open(status) as stream,
            ):
                self._digest(status, stream)
                if validate is not None:
                    validate(self._path(name))
                try:
                    os.link(self._path(name), store._path(status.reference))
                    _sync_directory(store._data)
                    store._journal.append(
                        [
                            ("upload.remove", {"id": name}),
                            ("artifact.put", status.reference.to_dict()),
                        ]
                    )
                    store._index.remove_upload(name)
                    store._index.insert(status.reference)
                    self._path(name).unlink()
                    _sync_directory(self._directory)
                    self._forget(name)
                except BaseException:
                    store._broken = True
                    raise
            return _complete(status.reference)

    def close(self) -> None:
        for prefix in self._prefixes.values():
            prefix.memory.close()
        self._prefixes.clear()
