"""Immutable server references with explicit laptop downloads."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ._async import CallFuture
from .protocol.requests import ArtifactRef

if TYPE_CHECKING:
    from .service import ServiceClient


@dataclass(frozen=True)
class ArtifactHandle:
    reference: ArtifactRef
    _service: ServiceClient

    @property
    def id(self) -> str:
        return self.reference.id

    def download(self, path: str | Path, *, resume: bool = False) -> CallFuture[Path]:
        """Stream and verify into a new local file; never overwrite an existing one."""
        if type(resume) is not bool:
            raise TypeError("download resume must be boolean")
        destination = Path(path)

        def transfer() -> Path:
            if resume:
                from .downloads import resume_download

                return resume_download(
                    self._service._transport,
                    self.reference,
                    destination,
                    memory=self._service._memory,
                )
            with destination.open("xb") as stream:
                try:
                    self._service._transport.download(
                        "/v1/artifacts/" + self.id, self.reference, stream
                    )
                except BaseException:
                    destination.unlink()
                    raise
            return destination

        return self._service._work.submit(transfer)


class ArtifactTransfers:
    """Explicit transfers into this authenticated service's artifact namespace."""

    def __init__(self, service: ServiceClient) -> None:
        self._service = service

    def upload(
        self, path: str | Path, *, reference: ArtifactRef | None = None
    ) -> CallFuture[ArtifactHandle]:
        """Register a file, checkpoint or weights directory with verified resume.

        Repeating this call explicitly resumes an incomplete transfer. Exchanges
        are never retried automatically. Keep local files unchanged until the
        future finishes. Supply ``reference`` to preserve an existing dependency
        identity when copying between services. No workers start on upload.
        """
        from .uploads import upload

        if reference is not None and not isinstance(reference, ArtifactRef):
            raise TypeError("upload reference requires an immutable artifact")
        source = Path(path)
        return self._service._work.submit(
            lambda: upload(self._service, source, reference)
        )
