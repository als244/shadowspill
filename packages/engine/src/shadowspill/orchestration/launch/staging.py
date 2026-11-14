"""Owned, quota-bound node-local delivery of opaque binary artifacts."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, RLock
from typing import BinaryIO, Self

from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.codec import MAX_METADATA_BYTES, ProtocolError
from shadowspill_client.protocol.requests import ArtifactRef

COPY_BYTES = 64 << 10


class FileDelivery:
    """An ephemeral agent-owned copy, independent of the sender's file lifetime.

    Disk and retained metadata/copy scratch are admitted before opening a source
    or creating files. Sources are authenticated artifact streams supplied by the
    controller connection, never paths supplied by a request. The transport owns
    read deadlines. This object owns only files below its explicit private root.
    """

    def __init__(
        self,
        references: Sequence[ArtifactRef],
        *,
        root: Path,
        disk: BufferBudget,
        memory: BufferBudget,
    ) -> None:
        if not isinstance(references, (tuple, list)) or len(references) > 65536:
            raise ValueError("artifact delivery requires a bounded reference list")
        if any(not isinstance(reference, ArtifactRef) for reference in references):
            raise TypeError("artifact delivery requires validated references")
        metadata = sum(
            len(reference.id.encode()) + len(reference.media_type.encode()) + 128
            for reference in references
        )
        if metadata > MAX_METADATA_BYTES:
            raise ProtocolError("artifact delivery metadata exceeds its bound")
        retained = 4096 + 8 * metadata + 512 * len(references)
        self.size_bytes = sum(reference.size_bytes for reference in references)
        self._memory = memory.reserve(retained + COPY_BYTES, timeout=0)
        try:
            self._disk = disk.reserve(self.size_bytes, timeout=0)
        except BaseException:
            self._memory.close()
            raise
        try:
            self._references = {reference.id: reference for reference in references}
            if len(self._references) != len(references):
                raise ValueError("delivered artifact IDs must be distinct")
            self._directory = TemporaryDirectory(prefix="delivery-", dir=root)
        except BaseException:
            self._disk.close()
            self._memory.close()
            raise
        self._retained = retained
        self._paths = {
            name: Path(self._directory.name) / str(i)
            for i, name in enumerate(self._references)
        }
        self._mutex = RLock()
        self._cancel = Event()
        self._users = 0
        self._receiving = False
        self._started = self.ready = self.closed = False

    def receive(
        self, open_source: Callable[[ArtifactRef], AbstractContextManager[BinaryIO]]
    ) -> None:
        """Copy once; any failure leaves an unready owner for explicit cleanup."""
        with self._mutex:
            if self.closed or self._started:
                raise RuntimeError("artifact delivery is closed or already started")
            self._started = True
            self._receiving = True
        try:
            for name, reference in self._references.items():
                self._check_cancelled()
                with (
                    open_source(reference) as source,
                    self._paths[name].open("xb", buffering=0) as destination,
                ):
                    self._copy(reference, source, destination)
            with self._mutex:
                self._check_cancelled()
                self.ready = True
        finally:
            with self._mutex:
                # No transfer may resume with this owner after failure.
                self._memory.shrink(self._retained)
                self._receiving = False

    def _copy(
        self, reference: ArtifactRef, source: BinaryIO, destination: BinaryIO
    ) -> None:
        digest = hashlib.sha256()
        remaining = reference.size_bytes
        while remaining:
            self._check_cancelled()
            maximum = min(remaining, COPY_BYTES)
            block = source.read(maximum)
            part = None
            try:
                self._check_cancelled()
                if not isinstance(block, bytes) or not 0 < len(block) <= maximum:
                    raise ProtocolError(
                        "artifact stream is truncated or exceeds its read bound"
                    )
                digest.update(block)
                part = memoryview(block)
                while part:
                    written = destination.write(part)
                    if type(written) is not int or not 0 < written <= len(part):
                        raise OSError("artifact delivery write made no progress")
                    part = part[written:]
                remaining -= len(block)
            finally:
                # Also release on failure before the scratch lease shrinks;
                # a retained exception traceback must not keep our chunk alive.
                del part, block
        self._check_cancelled()
        if source.read(1) != b"":
            raise ProtocolError("artifact stream exceeds its declared size")
        if digest.hexdigest() != reference.sha256:
            raise ProtocolError("artifact delivery checksum differs from its reference")

    def _check_cancelled(self) -> None:
        if self._cancel.is_set():
            raise InterruptedError("artifact delivery was cancelled")

    def cancel(self) -> None:
        """Wake between bounded reads; an active reader retains its quotas/files."""
        with self._mutex:
            if not self.ready:
                self._cancel.set()

    @contextmanager
    def use(self) -> Iterator[Callable[[ArtifactRef], Path]]:
        """Retain delivered files until a worker/transfer consumer has finished."""
        with self._mutex:
            if self.closed or not self.ready:
                raise RuntimeError("artifact delivery is not ready")
            self._users += 1
            active = True

        def resolve(reference: ArtifactRef) -> Path:
            with self._mutex:
                if self.closed or not active:
                    raise RuntimeError("artifact delivery is no longer borrowed")
                if self._references.get(reference.id) != reference:
                    raise ValueError("artifact reference is not owned by this delivery")
                return self._paths[reference.id]

        try:
            yield resolve
        finally:
            with self._mutex:
                active = False
                self._users -= 1

    def close(self) -> None:
        with self._mutex:
            if self.closed:
                return
            if self._users:
                raise RuntimeError(
                    "cannot remove delivered files while consumers borrow them"
                )
            if self._receiving:
                raise RuntimeError(
                    "cannot remove delivered files while a producer owns them"
                )
            # Failed deletion keeps both quotas until cleanup actually succeeds.
            self._directory.cleanup()
            self._paths.clear()
            self._references.clear()
            self._disk.close()
            self._memory.close()
            self.ready = False
            self.closed = True

    def __enter__(self) -> Self:
        if self.closed:
            raise RuntimeError("artifact delivery is closed")
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
