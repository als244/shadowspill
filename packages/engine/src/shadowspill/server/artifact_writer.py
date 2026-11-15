"""An admitted sequential producer for an unpublished persistent artifact."""

import hashlib
from collections.abc import Callable, Sequence
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


class ProducedArtifacts:
    """One admitted group, owned through production, commit and file cleanup.

    Installed binary encoders write through the existing bounded OutputWriter.
    Hashing occurs during those writes; no tensor serializer, full host snapshot,
    second file copy or post-write hashing pass is introduced here.
    """

    def __init__(
        self,
        specs: Sequence[OutputSpec],
        *,
        root: Path,
        disk: BufferBudget,
        memory: BufferBudget,
    ) -> None:
        self.specs = tuple(specs)
        # One serial copy buffer; references, journal encoding and paths are
        # bounded per member. Disk admission covers the entire unpublished group.
        self._memory = memory.reserve((36 << 10) * len(specs) + COPY_BYTES, timeout=0)
        try:
            self._disk = disk.reserve(sum(spec.size_bytes for spec in specs), timeout=0)
        except BaseException:
            self._memory.close()
            raise
        try:
            self._directory = TemporaryDirectory(prefix="production-", dir=root)
        except BaseException:
            self._disk.close()
            self._memory.close()
            raise
        self.paths = tuple(
            Path(self._directory.name) / str(index) for index in range(len(specs))
        )
        self.closed = False

    def build(
        self, write: Callable[[OutputSpec, OutputWriter], ArtifactRef]
    ) -> tuple[ArtifactRef, ...]:
        if self.closed:
            raise RuntimeError("artifact producer is closed")
        references = []
        for spec, path in zip(self.specs, self.paths, strict=True):
            with path.open("xb", buffering=0) as stream:
                writer = _HashingWriter(stream, spec.size_bytes)
                reference = write(spec, writer)
                if not isinstance(reference, ArtifactRef):
                    raise TypeError(
                        "artifact encoder must return its immutable reference"
                    )
                spec.check(reference)
                if (
                    spec.size_bytes - writer.remaining != reference.size_bytes
                    or writer.digest.hexdigest() != reference.sha256
                ):
                    raise ValueError("artifact encoder bytes differ from its reference")
                references.append(reference)
        return tuple(references)

    def close(self) -> None:
        if not self.closed:
            self._directory.cleanup()
            self._disk.close()
            self._memory.close()
            self.closed = True
