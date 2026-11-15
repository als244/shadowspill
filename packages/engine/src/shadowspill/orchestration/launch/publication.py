"""Pre-execution file reservations retained through output transfer or cleanup."""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import RLock
from typing import BinaryIO, Self

from shadowspill_client._records import Record, integer, text
from shadowspill_client.buffers import BufferBudget, BufferLease
from shadowspill_client.protocol.codec import RecordCodec
from shadowspill_client.protocol.requests import ArtifactRef, identifier

from .staging import COPY_BYTES

PERMIT_BYTES = 4096


@dataclass(frozen=True)
class OutputSpec(Record):
    """Admitted opaque output envelope, known before its producer starts.

    Tensor/checkpoint encoders require the exact default size. Optional reports
    may finish below a reserved maximum; their returned reference still names
    the exact written bytes. Both modes retain the full reservation through
    producer completion and cleanup.
    """

    id: str
    size_bytes: int
    media_type: str
    exact_size: bool = True

    def __post_init__(self) -> None:
        identifier(self.id, "output ID")
        integer(self.size_bytes, "output bytes")
        text(self.media_type, "output media type")
        if len(self.media_type.encode()) > 256:
            raise ValueError("output media type exceeds its bound")
        if type(self.exact_size) is not bool:
            raise TypeError("output exact-size flag must be boolean")

    def check(self, reference: ArtifactRef) -> None:
        if (
            reference.id != self.id
            or reference.size_bytes > self.size_bytes
            or (self.exact_size and reference.size_bytes != self.size_bytes)
            or reference.media_type != self.media_type
        ):
            raise ValueError("output reference differs from its reservation")


_CODEC = RecordCodec({"node.output": OutputSpec})


def _paths(root: Path, spec: OutputSpec) -> tuple[Path, Path]:
    key = hashlib.sha256(spec.id.encode()).hexdigest()
    return root / (key + ".permit"), root / (key + ".data")


def _open_regular(path: Path) -> BinaryIO:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("output storage requires a regular file")
        return os.fdopen(descriptor, "rb", buffering=0)
    except BaseException:
        os.close(descriptor)
        raise


def require_output(root: Path, spec: OutputSpec) -> Path:
    """Check an agent's bounded permit; the root is private launch configuration."""
    permit, payload = _paths(root, spec)
    with _open_regular(permit) as stream:
        raw = stream.read(PERMIT_BYTES + 1)
    if _CODEC.decode(raw, max_bytes=PERMIT_BYTES) != spec:
        raise ValueError("worker output differs from the agent reservation")
    return payload


class OutputWriter:
    """Count bounded borrowed writes without buffering or admitting excess bytes."""

    def __init__(self, stream: BinaryIO, size: int) -> None:
        self._stream, self.remaining = stream, size

    def write(self, data: memoryview, /) -> int:
        if (
            not isinstance(data, memoryview)
            or data.ndim != 1
            or data.itemsize != 1
            or not data.c_contiguous
            or not 0 < data.nbytes <= min(self.remaining, COPY_BYTES)
        ):
            raise ValueError("output writer exceeded its admitted byte bound")
        count = self._stream.write(data)
        if type(count) is not int or not 0 < count <= data.nbytes:
            raise OSError("output file write made no progress")
        self.remaining -= count
        return count


def write_output(
    root: Path,
    spec: OutputSpec,
    write: Callable[[OutputWriter], ArtifactRef],
) -> ArtifactRef:
    """Write a permitted output once; the agent owns even partial failed files.

    The owner must mark its producer active before releasing the worker command.
    No filesystem evidence alone establishes that the producer has stopped.
    """
    path = require_output(root, spec)
    with path.open("xb", buffering=0) as stream:
        writer = OutputWriter(stream, spec.size_bytes)
        reference = write(writer)
        spec.check(reference)
        if spec.size_bytes - writer.remaining != reference.size_bytes:
            raise ValueError("output producer bytes differ from its declared size")
        os.fsync(stream.fileno())
    return reference


class OutputFiles:
    """Node-local storage independent of a worker's resource lease.

    Reserve file bytes, bounded permit metadata and verification scratch before
    exposing a producer permit. The controller transfers verified files through
    the same authenticated artifact streams used by FileDelivery.
    """

    def __init__(self, *, root: Path, disk: BufferBudget, memory: BufferBudget) -> None:
        self._disk, self._memory = disk, memory
        self._lease = memory.reserve(4096, timeout=0)
        try:
            self._directory = TemporaryDirectory(prefix="outputs-", dir=root)
        except BaseException:
            self._lease.close()
            raise
        self.root = Path(self._directory.name)
        self._lock = RLock()
        self._entries: dict[str, OutputFile] = {}
        self._abandoned: set[OutputFile] = set()
        self.closed = False

    def reserve(self, spec: OutputSpec) -> OutputFile:
        raw = _CODEC.encode(spec, max_bytes=PERMIT_BYTES)
        with self._lock:
            if self.closed:
                raise RuntimeError("output storage is closed")
            if spec.id in self._entries:
                raise ValueError("output ID is already reserved")
            memory = self._memory.reserve(4096 + 8 * len(raw) + COPY_BYTES, timeout=0)
            try:
                disk = self._disk.reserve(spec.size_bytes + len(raw), timeout=0)
            except BaseException:
                memory.close()
                raise
            owner = OutputFile(self, spec, disk, memory)
            self._entries[spec.id] = owner
            try:
                permit, _ = _paths(self.root, spec)
                with permit.open("xb", buffering=0) as stream:
                    part = memoryview(raw)
                    while part:
                        count = stream.write(part)
                        if type(count) is not int or not 0 < count <= len(part):
                            raise OSError("output permit write made no progress")
                        part = part[count:]
            except BaseException:
                try:
                    owner.close()
                except BaseException:
                    self._abandoned.add(owner)
                    raise
                raise
            return owner

    def close(self) -> None:
        with self._lock:
            if self.closed:
                return
            for owner in tuple(self._abandoned):
                owner.close()
            if self._entries:
                raise RuntimeError("output files still have storage owners")
            self._directory.cleanup()
            self._lease.close()
            self.closed = True

    def __enter__(self) -> Self:
        if self.closed:
            raise RuntimeError("output storage is closed")
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class OutputFile:
    """Retain through matching completion/abort or confirmed owned-child cleanup."""

    def __init__(
        self,
        owner: OutputFiles,
        spec: OutputSpec,
        disk: BufferLease,
        memory: BufferLease,
    ) -> None:
        self._owner, self.spec = owner, spec
        self._disk, self._memory = disk, memory
        self._started = self._producing = False
        self._users = 0
        self.reference: ArtifactRef | None = None
        self.closed = False

    def begin(self) -> None:
        with self._owner._lock:
            if self.closed or self._started:
                raise RuntimeError("output producer is closed or already started")
            self._started = self._producing = True

    def finished(self) -> None:
        """Agent calls only after confirmed command termination or child cleanup."""
        with self._owner._lock:
            if not self._producing:
                raise RuntimeError("output has no active producer")
            self._producing = False

    def verify(self, reference: ArtifactRef) -> None:
        with self._owner._lock:
            if self.closed or not self._started or self._producing:
                raise RuntimeError("output producer has not finished")
            self.spec.check(reference)
            if self.reference is not None:
                if self.reference != reference:
                    raise ValueError("verified output reference cannot change")
                return
            _, path = _paths(self._owner.root, self.spec)
            digest = hashlib.sha256()
            with _open_regular(path) as stream:
                if os.fstat(stream.fileno()).st_size != reference.size_bytes:
                    raise ValueError("output file differs from its declared size")
                remaining = reference.size_bytes
                while remaining:
                    block = stream.read(min(remaining, COPY_BYTES))
                    try:
                        if not block:
                            raise ValueError("output file is truncated")
                        digest.update(block)
                        remaining -= len(block)
                    finally:
                        del block
                if stream.read(1) or digest.hexdigest() != reference.sha256:
                    raise ValueError("output checksum differs from its reference")
            self.reference = reference

    @contextmanager
    def open(self, reference: ArtifactRef) -> Iterator[BinaryIO]:
        with self._owner._lock:
            if self.closed or self.reference is None or reference != self.reference:
                raise ValueError("output is not a verified owned reference")
            _, path = _paths(self._owner.root, self.spec)
            stream = _open_regular(path)
            self._users += 1
        try:
            with stream:
                yield stream
        finally:
            with self._owner._lock:
                self._users -= 1

    def close(self) -> None:
        with self._owner._lock:
            if self.closed:
                return
            if self._producing or self._users:
                raise RuntimeError(
                    "cannot delete output with an active producer/consumer"
                )
            for path in _paths(self._owner.root, self.spec):
                path.unlink(missing_ok=True)
            self._disk.close()
            self._memory.close()
            del self._owner._entries[self.spec.id]
            self._owner._abandoned.discard(self)
            self.closed = True

    def __enter__(self) -> Self:
        if self.closed:
            raise RuntimeError("output file is closed")
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
