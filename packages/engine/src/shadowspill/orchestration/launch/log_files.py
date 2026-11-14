"""Reserve child logs before launch and retain them through artifact transfer."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import RLock
from types import MappingProxyType
from typing import BinaryIO, Self
from uuid import uuid4

from shadowspill_client._records import integer
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.requests import ArtifactRef

from .admission import NodeLease
from .publication import _open_regular
from .staging import COPY_BYTES


class LogFiles:
    """Two capped streams owned independently of the worker's resource lease.

    The lease confirms every owned producer has stopped before logs can be read
    or deleted. A transfer keeps both quota and files alive until its stream is
    closed. The publisher may retain a separate immutable copy before cleanup.
    """

    def __init__(
        self,
        lease: NodeLease,
        device_id: str | None,
        *,
        root: Path,
        disk: BufferBudget,
        memory: BufferBudget,
        limit_bytes: int = 64 << 20,
    ) -> None:
        if not isinstance(lease, NodeLease) or lease.closed:
            raise ValueError("worker logs require a live node resource lease")
        if device_id is not None and device_id not in lease.request.devices:
            raise ValueError("worker log device is not reserved")
        self.limit_bytes = integer(limit_bytes, "per-stream log bytes")
        self._memory = memory.reserve(4096 + COPY_BYTES, timeout=0)
        try:
            self._disk = disk.reserve(2 * limit_bytes + 12288, timeout=0)
        except BaseException:
            self._memory.close()
            raise
        try:
            self._temporary = TemporaryDirectory(prefix="logs-", dir=root)
        except BaseException:
            self._disk.close()
            self._memory.close()
            raise
        self.directory = Path(self._temporary.name) / "process"
        self._lease, self._device = lease, device_id
        self._id = "logs." + uuid4().hex
        self._lock = RLock()
        self._users = 0
        self._references: dict[str, ArtifactRef] | None = None
        self.closed = False

    def check(self, lease: NodeLease, device_id: str | None) -> None:
        if self.closed or lease is not self._lease or device_id != self._device:
            raise ValueError(
                "worker logs belong to a different or closed resource owner"
            )

    def _finished(self) -> None:
        if self.closed:
            raise RuntimeError("worker log storage is closed")
        if not self._lease.closed:
            raise RuntimeError("worker log producers have not confirmed termination")

    def references(self) -> Mapping[str, ArtifactRef]:
        """Hash finalized capped files with bounded scratch; reuse verified digests."""
        with self._lock:
            self._finished()
            if self._references is None:
                references = {}
                for name in ("stdout", "stderr"):
                    with _open_regular(self.directory / f"{name}.log") as stream:
                        size = os.fstat(stream.fileno()).st_size
                        if not 0 <= size <= self.limit_bytes:
                            raise ValueError(
                                "worker log exceeds its admitted file bound"
                            )
                        digest, remaining = hashlib.sha256(), size
                        while remaining:
                            block = stream.read(min(COPY_BYTES, remaining))
                            try:
                                if not block:
                                    raise ValueError(
                                        "worker log was truncated during publication"
                                    )
                                digest.update(block)
                                remaining -= len(block)
                            finally:
                                del block
                        if stream.read(1):
                            raise ValueError(
                                "worker log grew after producer termination"
                            )
                    references[name] = ArtifactRef(
                        f"{self._id}.{name}", digest.hexdigest(), size, "text/plain"
                    )
                self._references = references
            return MappingProxyType(self._references)

    @contextmanager
    def open(self, reference: ArtifactRef) -> Iterator[BinaryIO]:
        """Borrow an exact finalized stream for an authenticated artifact copy."""
        with self._lock:
            references = self.references()
            names = [name for name, item in references.items() if item == reference]
            if len(names) != 1:
                raise ValueError("worker log reference differs from finalized content")
            stream = _open_regular(self.directory / f"{names[0]}.log")
            try:
                if os.fstat(stream.fileno()).st_size != reference.size_bytes:
                    raise ValueError("finalized worker log size changed")
            except BaseException:
                stream.close()
                raise
            self._users += 1
        try:
            yield stream
        finally:
            try:
                stream.close()
            finally:
                with self._lock:
                    self._users -= 1

    def close(self) -> None:
        with self._lock:
            if self.closed:
                return
            self._finished()
            if self._users:
                raise RuntimeError("worker log transfer still has a live reader")
            # Failed deletion retains both reservations and remains retryable.
            self._temporary.cleanup()
            self._disk.close()
            self._memory.close()
            self.closed = True

    def __enter__(self) -> Self:
        if self.closed:
            raise RuntimeError("worker log storage is closed")
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
