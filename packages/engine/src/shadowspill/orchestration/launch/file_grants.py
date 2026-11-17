"""Process-readable opaque file grants backed by retained storage ownership."""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager, ExitStack, contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import RLock
from typing import Self

from shadowspill_client.buffers import BufferBudget, BufferLease
from shadowspill_client.protocol.codec import MAX_METADATA_BYTES, RecordCodec
from shadowspill_client.protocol.requests import ArtifactRef

from .staging import COPY_BYTES, FileDelivery

_CODEC = RecordCodec({"node.file": ArtifactRef})

type LocalSource = Callable[[ArtifactRef], AbstractContextManager[Path]]


@contextmanager
def _borrow(
    source: FileDelivery | LocalSource,
) -> Iterator[Callable[[ArtifactRef], Path]]:
    if isinstance(source, FileDelivery):
        with source.use() as resolve:
            yield resolve
    else:
        # Only a controller-supplied borrow can expose local paths. Its context
        # retains the original artifact's owner and quota until all links retire.
        with ExitStack() as pins:
            yield lambda reference: pins.enter_context(source(reference))


def _key(reference: ArtifactRef) -> str:
    return hashlib.sha256(_CODEC.encode(reference)).hexdigest()


def resolve_file(root: Path, reference: ArtifactRef) -> Path:
    """Resolve only a complete opaque reference inside an agent-supplied root.

    The root comes from private launch configuration, never a training request.
    The caller must retain the corresponding grant through the consumer's exit or
    confirmed completion. The worker's normal input decoder still checks content.
    """
    path = root / _key(reference)
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_size != reference.size_bytes:
        raise ValueError("granted file differs from the requested artifact")
    return path


def verify_file(root: Path, reference: ArtifactRef) -> Path:
    """Recheck startup artifacts with bounded scratch before decoding/loading.

    The agent retains the immutable grant through this read and the later worker
    consumer lifetime. Input frames have their own prepared payload decoder.
    """
    path = resolve_file(root, reference)
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as stream:
        remaining = reference.size_bytes
        while remaining:
            block = stream.read(min(remaining, COPY_BYTES))
            try:
                if not block:
                    raise ValueError("granted artifact is truncated")
                digest.update(block)
                remaining -= len(block)
            finally:
                del block
        if stream.read(1) or digest.hexdigest() != reference.sha256:
            raise ValueError("granted artifact checksum differs from its reference")
    return path


class FileGrants:
    """A private flat directory of links; grants retain original storage quotas.

    Link names hash the entire reference, not just its ID or content. No incoming
    path or lookup manifest is interpreted. The source and grant directories
    must share a filesystem; there is no hidden, unadmitted copy fallback.
    """

    def __init__(self, *, root: Path, memory: BufferBudget) -> None:
        self._memory = memory
        self._lease = memory.reserve(4096, timeout=0)
        try:
            self._directory = TemporaryDirectory(prefix="grants-", dir=root)
        except BaseException:
            self._lease.close()
            raise
        self.root = Path(self._directory.name)
        self._lock = RLock()
        self._entries: dict[str, tuple[FileDelivery | LocalSource, int]] = {}
        self._owners: set[FileGrant] = set()
        self._abandoned: set[FileGrant] = set()
        self.closed = False

    def lend(
        self, source: FileDelivery | LocalSource, references: Sequence[ArtifactRef]
    ) -> FileGrant:
        if not isinstance(references, (tuple, list)) or len(references) > 65536:
            raise ValueError("file grants require a bounded reference list")
        if any(not isinstance(ref, ArtifactRef) for ref in references):
            raise TypeError("file grants require validated artifact references")
        metadata = sum(
            len(ref.id.encode()) + len(ref.media_type.encode()) + 128
            for ref in references
        )
        if metadata > MAX_METADATA_BYTES:
            raise ValueError("file grant metadata exceeds its bound")
        with self._lock:
            if self.closed:
                raise RuntimeError("file grants are closed")
            lease = self._memory.reserve(
                4096 + 8 * metadata + 512 * len(references), timeout=0
            )
            borrowed = _borrow(source)
            try:
                resolve = borrowed.__enter__()
            except BaseException:
                lease.close()
                raise
            owner = FileGrant(self, borrowed, lease)
            self._owners.add(owner)
            try:
                for reference in references:
                    path = resolve(reference)  # Exact reference pinned by its owner.
                    key = _key(reference)
                    if key in owner._keys:
                        raise ValueError(
                            "one grant cannot repeat an artifact reference"
                        )
                    if key not in self._entries:
                        os.link(path, self.root / key)
                        self._entries[key] = source, 0
                    original, users = self._entries[key]
                    if original is not source:
                        raise ValueError(
                            "artifact is already granted from another source; "
                            "reuse that source until all its consumers finish"
                        )
                    self._entries[key] = original, users + 1
                    owner._keys.add(key)
            except BaseException:
                # On a deletion failure the owner remains reachable in this set,
                # retaining the source and both metadata/storage reservations.
                try:
                    owner.close()
                except BaseException:
                    self._abandoned.add(owner)
                    raise
                raise
            return owner

    def retain(self, reference: ArtifactRef) -> FileGrant | None:
        """Borrow an existing exact grant without copying or transferring ownership.

        Startup artifacts can be used again by later requests. Retain the same
        storage owner under the grant lock so its last prior consumer cannot
        retire the file between lookup and the new borrow.
        """
        if not isinstance(reference, ArtifactRef):
            raise TypeError("file grants require validated artifact references")
        with self._lock:
            if self.closed:
                raise RuntimeError("file grants are closed")
            entry = self._entries.get(_key(reference))
            return None if entry is None else self.lend(entry[0], [reference])

    def _release(self, owner: FileGrant) -> None:
        with self._lock:
            for key in tuple(owner._keys):
                original, users = self._entries[key]
                if users == 1:
                    (self.root / key).unlink()
                    del self._entries[key]
                else:
                    self._entries[key] = original, users - 1
                owner._keys.remove(key)
            owner._borrowed.__exit__(None, None, None)
            owner._lease.close()
            self._owners.remove(owner)
            self._abandoned.discard(owner)
            owner.closed = True

    def close(self) -> None:
        with self._lock:
            if self.closed:
                return
            for owner in tuple(self._abandoned):
                owner.close()
            if self._owners:
                raise RuntimeError("cannot remove file grants while consumers own them")
            self._directory.cleanup()
            self._lease.close()
            self.closed = True

    def __enter__(self) -> Self:
        if self.closed:
            raise RuntimeError("file grants are closed")
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class FileGrant:
    """Hold until process cleanup or confirmed command completion, then close."""

    def __init__(
        self,
        owner: FileGrants,
        borrowed: AbstractContextManager[Callable[[ArtifactRef], Path]],
        lease: BufferLease,
    ) -> None:
        self._owner, self._borrowed, self._lease = owner, borrowed, lease
        self._keys: set[str] = set()
        self.closed = False

    def close(self) -> None:
        with self._owner._lock:
            if not self.closed:
                self._owner._release(self)

    def __enter__(self) -> Self:
        if self.closed:
            raise RuntimeError("file grant is closed")
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
