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
    """A stored artifact, reachable through one connection.

    `ServiceClient.artifact` and the calls that return stored files, such as
    `ArtifactTransfers.upload`, return one.

    Attributes
    ----------
    reference : ArtifactRef
        The artifact.
    """

    reference: ArtifactRef
    _service: ServiceClient

    @property
    def id(self) -> str:
        """The artifact's ID, as in ``reference``."""
        return self.reference.id

    def download(self, path: str | Path, *, resume: bool = False) -> CallFuture[Path]:
        """Download the artifact into a new local file.

        The bytes are checked against the reference's size and SHA-256. An
        existing file is never overwritten.

        Parameters
        ----------
        path : str or pathlib.Path
            The local file to create.
        resume : bool, default False
            Keep the progress of a failed download beside ``path``, in
            ``<path>.shadowspill-download``, and continue a download kept that
            way. Without it, a failed download removes its partial file.

        Returns
        -------
        CallFuture of pathlib.Path
            ``path``, once the complete file is in place. It fails with
            `FileExistsError` if ``path`` exists, and with `OSError` for a
            local file failure.

        Raises
        ------
        TypeError
            ``resume`` is not a bool.
        """
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
    """Uploads into a service's artifact storage, as `ServiceClient.artifacts`."""

    def __init__(self, service: ServiceClient) -> None:
        self._service = service

    def upload(
        self, path: str | Path, *, reference: ArtifactRef | None = None
    ) -> CallFuture[ArtifactHandle]:
        """Upload a local file, or a downloaded checkpoint or weights directory.

        Calling again with the same file continues an incomplete upload from
        the part the service has verified; nothing is retried automatically.
        Keep the local files unchanged until the future finishes.

        Parameters
        ----------
        path : str or pathlib.Path
            The local file or directory.
        reference : ArtifactRef, optional
            The artifact's existing reference, kept so that settings or
            manifests naming it still resolve, for example when copying
            between services. By default, a file gets an ID derived from its
            content.

        Returns
        -------
        CallFuture of ArtifactHandle
            The stored artifact: a `CheckpointHandle` or `WeightsHandle` for
            those kinds.

        Raises
        ------
        TypeError
            ``reference`` is not an `ArtifactRef`.
        """
        from .uploads import upload

        if reference is not None and not isinstance(reference, ArtifactRef):
            raise TypeError("upload reference requires an immutable artifact")
        source = Path(path)
        return self._service._work.submit(
            lambda: upload(self._service, source, reference)
        )
