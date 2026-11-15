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

    def download(self, path: str | Path) -> CallFuture[Path]:
        """Stream and verify into a new local file; never overwrite an existing one."""
        destination = Path(path)

        def transfer() -> Path:
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
