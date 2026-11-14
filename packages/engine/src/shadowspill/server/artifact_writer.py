"""An admitted sequential producer for an unpublished persistent artifact."""

import hashlib
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import BinaryIO

from shadowspill.orchestration.launch.publication import OutputSpec, OutputWriter
from shadowspill.orchestration.launch.staging import COPY_BYTES
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.requests import ArtifactRef


class _HashingWriter(OutputWriter):
    def __init__(self, stream: BinaryIO, size: int) -> None:
        super().__init__(stream, size)
        self.digest = hashlib.sha256()

    def write(self, data: memoryview, /) -> int:
        count = super().write(data)
        part = data[:count]
        try:
            self.digest.update(part)
        finally:
            part.release()
        return count


class ProducedArtifact:
    """The store owns this through production, commit and temporary-file cleanup.

    Installed binary encoders write through the existing bounded OutputWriter.
    Hashing occurs during those writes; no tensor serializer, full host snapshot,
    second file copy or post-write hashing pass is introduced here.
    """

    def __init__(
        self, spec: OutputSpec, *, root: Path, disk: BufferBudget, memory: BufferBudget
    ) -> None:
        self.spec = spec
        self._memory = memory.reserve((36 << 10) + COPY_BYTES, timeout=0)
        try:
            self._disk = disk.reserve(spec.size_bytes, timeout=0)
        except BaseException:
            self._memory.close()
            raise
        try:
            self._directory = TemporaryDirectory(prefix="production-", dir=root)
        except BaseException:
            self._disk.close()
            self._memory.close()
            raise
        self.path = Path(self._directory.name) / "payload"
        self.closed = False

    def build(self, write: Callable[[OutputWriter], ArtifactRef]) -> ArtifactRef:
        if self.closed:
            raise RuntimeError("artifact producer is closed")
        with self.path.open("xb", buffering=0) as stream:
            writer = _HashingWriter(stream, self.spec.size_bytes)
            reference = write(writer)
            if not isinstance(reference, ArtifactRef):
                raise TypeError("artifact encoder must return its immutable reference")
            self.spec.check(reference)
            if writer.remaining or writer.digest.hexdigest() != reference.sha256:
                raise ValueError("artifact encoder bytes differ from its reference")
            return reference

    def close(self) -> None:
        if not self.closed:
            self._directory.cleanup()
            self._disk.close()
            self._memory.close()
            self.closed = True
